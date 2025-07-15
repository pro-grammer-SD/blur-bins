import vapoursynth as vs
from vapoursynth import core
import blur.utils as u

cur_interp = None
dupe_last_good_idx = 0
dupe_next_good_idx = 0
duped_frames = 0


def get_interp(
    clip,
    duplicate_index,
    threshold: float,
    max_frames: int | None,
    model_path: str,
    gpu_index: int,
):
    global cur_interp
    global dupe_last_good_idx
    global dupe_next_good_idx
    global duped_frames

    duped_frame = clip[duplicate_index]

    def find_next_good_frame() -> int:
        index = duplicate_index + 1
        last_possible_index = (
            len(clip) - 1
        )  # for clarity (this shit always trips me up)

        if max_frames:
            max_permitted = duplicate_index + max_frames
            if last_possible_index > max_permitted:
                last_possible_index = max_permitted

        while index <= last_possible_index:
            test_frame = clip[index]
            diffclip = core.std.PlaneStats(test_frame, duped_frame)

            for frame_index2, frame2 in enumerate(diffclip.frames()):
                if frame2.props["PlaneStatsDiff"] >= threshold:
                    return index

            index += 1

        return None

    dupe_last_good_idx = duplicate_index - 1

    # find the next non-duplicate frame
    dupe_next_good_idx = find_next_good_frame()

    if not dupe_next_good_idx:
        # don't dedupe
        cur_interp = None
        return

    duped_frames = dupe_next_good_idx - dupe_last_good_idx

    # generate fake clip which includes the two good frames. this will be used to interpolate between them.
    # todo: possibly including more frames will result in better results?
    good_frames = clip[dupe_last_good_idx] + clip[dupe_next_good_idx]

    cur_interp = core.rife.RIFE(
        good_frames,
        fps_num=duped_frames,
        fps_den=1,
        model_path=model_path,
        gpu_id=gpu_index,
    )

    cur_interp = cur_interp[1 : 1 + duped_frames]  # first frame is a duplicate

    cur_interp = core.std.AssumeFPS(cur_interp, fpsnum=1, fpsden=1)


def interpolate_dupes(
    clip,
    frame_index,
    threshold: float,
    max_frames: int | None,
    model_path: str,
    gpu_index: int,
):
    global cur_interp
    global dupe_last_good_idx
    global dupe_next_good_idx

    clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)

    if cur_interp is None:
        # haven't interpolated yet
        get_interp(clip1, frame_index, threshold, max_frames, model_path, gpu_index)

    if cur_interp is None:
        # interpolated but no dedupe solution. get out
        return clip

    # combine the good frames with the interpolated ones so that vapoursynth can use them by indexing
    # (i hate how you have to do this, there might be nicer way idk)
    good_before = core.std.Trim(clip1, first=0, last=dupe_last_good_idx)
    good_after = core.std.Trim(clip1, first=dupe_next_good_idx)

    joined = good_before + cur_interp + good_after

    return core.std.AssumeFPS(joined, src=clip)


def fill_drops_rife(
    clip: vs.VideoNode,
    model_path: str,
    gpu_index: int,
    threshold: float = 0.1,
    max_frames: int | None = None,
    debug=False,
):
    u.check_model_path(model_path)

    def handle_frames(n, f):
        global cur_interp

        if f.props["PlaneStatsDiff"] >= threshold or n == 0:
            cur_interp = None
            return clip

        # duplicate frame
        interp = interpolate_dupes(
            clip, n, threshold, max_frames, model_path, gpu_index
        )

        if debug:
            return core.text.Text(
                clip=interp,
                text=f"duplicate, {duped_frames - 1} gap, diff: {f.props['PlaneStatsDiff']:.4f}",
                alignment=8,
            )

        return interp

    orig_format = clip.format
    needs_conversion = orig_format.id != vs.RGBS

    if needs_conversion:
        # Convert to RGBS for RIFE
        clip = core.resize.Bicubic(
            clip,
            format=vs.RGBS,
            matrix_in_s="709" if orig_format.color_family == vs.YUV else None,
        )

    diffclip = core.std.PlaneStats(clip, clip[0] + clip)
    out = core.std.FrameEval(clip, handle_frames, prop_src=diffclip)

    if needs_conversion:
        # Convert back to original format
        out = core.resize.Bicubic(
            out,
            format=orig_format.id,
            matrix_s="709" if orig_format.color_family == vs.YUV else None,
        )

    return out
