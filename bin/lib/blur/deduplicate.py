import vapoursynth as vs
from vapoursynth import core

import blur.interpolate

cur_interp = None
dupe_last_good_idx = 0
dupe_next_good_idx = 0
duped_frames = 0


def get_interp(
    clip,
    duplicate_index,
    threshold: float,
    max_frames: int | None,
    svp_preset,
    svp_algorithm,
    svp_blocksize,
    svp_masking,
    svp_gpu,
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

    duped_frames = dupe_next_good_idx - duplicate_index

    # generate fake clip which includes the two good frames. this will be used to interpolate between them.
    # todo: possibly including more frames will result in better results?
    good_frames = clip[dupe_last_good_idx] + clip[dupe_next_good_idx]

    [super_string, vectors_string, smooth_string] = (
        blur.interpolate.generate_svp_strings(
            new_fps=duped_frames + 1,
            preset=svp_preset,
            algorithm=svp_algorithm,
            blocksize=svp_blocksize,
            # overlap=2,
            # speed="medium",
            masking=svp_masking,
            gpu=svp_gpu,
        )
    )

    super = core.svp1.Super(good_frames, super_string)
    vectors = core.svp1.Analyse(
        super["clip"], super["data"], good_frames, vectors_string
    )

    cur_interp = core.svp2.SmoothFps(
        good_frames,
        super["clip"],
        super["data"],
        vectors["clip"],
        vectors["data"],
        smooth_string,
        src=good_frames,
        fps=good_frames.fps,
    )

    # trim edges (they're just the input frames)
    cur_interp = cur_interp[1:-1]

    cur_interp = core.std.AssumeFPS(cur_interp, fpsnum=1, fpsden=1)


def interpolate_dupes(
    clip,
    frame_index,
    threshold: float,
    max_frames: int | None,
    svp_preset,
    svp_algorithm,
    svp_blocksize,
    svp_masking,
    svp_gpu,
):
    global cur_interp
    global dupe_last_good_idx
    global dupe_next_good_idx

    clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)

    if cur_interp is None:
        # haven't interpolated yet
        get_interp(
            clip1,
            frame_index,
            threshold,
            max_frames,
            svp_preset,
            svp_algorithm,
            svp_blocksize,
            svp_masking,
            svp_gpu,
        )

    if cur_interp is None:
        # interpolated but no dedupe solution. get out
        return clip

    # combine the good frames with the interpolated ones so that vapoursynth can use them by indexing
    # (i hate how you have to do this, there might be nicer way idk)
    good_before = core.std.Trim(clip1, first=0, last=dupe_last_good_idx)
    good_after = core.std.Trim(clip1, first=dupe_next_good_idx)

    joined = good_before + cur_interp + good_after

    return core.std.AssumeFPS(joined, src=clip)


def fill_drops_multiple(
    video: vs.VideoNode,
    threshold: float = 0.1,
    max_frames: int | None = None,
    svp_preset=blur.interpolate.DEFAULT_PRESET,
    svp_algorithm=blur.interpolate.DEFAULT_ALGORITHM,
    svp_blocksize=blur.interpolate.DEFAULT_BLOCKSIZE,
    svp_masking=blur.interpolate.DEFAULT_MASKING,
    svp_gpu=blur.interpolate.DEFAULT_GPU,
    debug=False,
):
    def handle_frames(n, f):
        global cur_interp

        if f.props["PlaneStatsDiff"] >= threshold or n == 0:
            cur_interp = None
            return video

        # duplicate frame
        interp = interpolate_dupes(
            video,
            n,
            threshold,
            max_frames,
            svp_preset,
            svp_algorithm,
            svp_blocksize,
            svp_masking,
            svp_gpu,
        )

        if debug:
            return core.text.Text(
                clip=interp,
                text=f"duplicate, {duped_frames} gap, diff: {f.props['PlaneStatsDiff']:.4f}",
                alignment=8,
            )

        return interp

    orig_format = video.format
    needs_conversion = (
        orig_format.id != vs.YUV420P8
    )  # svp only accepts yv12 (SVSuper: Clip must be YV12)

    if needs_conversion:
        video = core.resize.Bicubic(video, format=vs.YUV420P8)

    diffclip = core.std.PlaneStats(video, video[0] + video)
    out = core.std.FrameEval(video, handle_frames, prop_src=diffclip)

    if needs_conversion:
        # Convert back to original format
        out = core.resize.Bicubic(
            out,
            format=orig_format.id,
            matrix_s="709" if orig_format.color_family == vs.YUV else None,
        )

    return out


def fill_drops_old(clip, threshold=0.1, debug=False):
    if not isinstance(clip, vs.VideoNode):
        raise ValueError("This is not a clip")

    differences = core.std.PlaneStats(clip, clip[0] + clip)

    super = core.mv.Super(clip)
    forward_vectors = core.mv.Analyse(super, isb=False)
    backwards_vectors = core.mv.Analyse(super, isb=True)
    filldrops = core.mv.FlowInter(
        clip, super, mvbw=backwards_vectors, mvfw=forward_vectors, ml=1
    )

    def selectFunc(n, f):
        if f.props["PlaneStatsDiff"] < threshold:
            if debug:
                return core.text.Text(
                    filldrops,
                    f"interpolated, diff: {f.props['PlaneStatsDiff']:.3f}",
                    alignment=8,
                )

            return filldrops
        else:
            return clip

    return core.std.FrameEval(clip, selectFunc, prop_src=differences)


def fill_drops_svp(
    video,
    threshold: float = 0.1,
    svp_preset=blur.interpolate.DEFAULT_PRESET,
    svp_algorithm=blur.interpolate.DEFAULT_ALGORITHM,
    svp_blocksize=blur.interpolate.DEFAULT_BLOCKSIZE,
    svp_masking=blur.interpolate.DEFAULT_MASKING,
    svp_gpu=blur.interpolate.DEFAULT_GPU,
    debug=False,
):
    if not isinstance(video, vs.VideoNode):
        raise ValueError("This is not a video")

    [super_string, vectors_string, smooth_string] = (
        blur.interpolate.generate_svp_strings(
            new_fps=video.fps,
            preset=svp_preset,
            algorithm=svp_algorithm,
            blocksize=svp_blocksize,
            masking=svp_masking,
            gpu=svp_gpu,
        )
    )

    super = core.svp1.Super(video, super_string)
    vectors = core.svp1.Analyse(super["clip"], super["data"], video, vectors_string)
    filldrops = core.svp2.SmoothFps(
        video,
        super["clip"],
        super["data"],
        vectors["clip"],
        vectors["data"],
        smooth_string,
        src=video,
        fps=video.fps,
    )

    def selectFunc(n, f):
        if f.props["PlaneStatsDiff"] >= threshold or n == 0:
            return video

        clip_1fps = core.std.AssumeFPS(video, fpsnum=1, fpsden=1)

        good_frames = clip_1fps[n - 1] + clip_1fps[n + 1]

        [super_string, vectors_string, smooth_string] = (
            blur.interpolate.generate_svp_strings(
                new_fps=3,
                preset=svp_preset,
                algorithm=svp_algorithm,
                blocksize=svp_blocksize,
                # overlap=2,
                # speed="medium",
                masking=svp_masking,
                gpu=svp_gpu,
            )
        )

        super = core.svp1.Super(good_frames, super_string)
        vectors = core.svp1.Analyse(
            super["clip"], super["data"], good_frames, vectors_string
        )

        cur_interp = core.svp2.SmoothFps(
            good_frames,
            super["clip"],
            super["data"],
            vectors["clip"],
            vectors["data"],
            smooth_string,
            src=good_frames,
            fps=good_frames.fps,
        )

        # trim edges (they're just the input frames)
        cur_interp = cur_interp[1:-1]

        # combine the good frames with the interpolated ones so that vapoursynth can use them by indexing
        # (i hate how you have to do this, there might be nicer way idk)
        good_before = core.std.Trim(clip_1fps, first=0, last=n - 1)
        good_after = core.std.Trim(clip_1fps, first=n + 1)

        joined = good_before + cur_interp + good_after

        out_video = core.std.AssumeFPS(joined, src=video)

        if debug:
            return core.text.Text(
                out_video,
                f"interpolated, diff: {f.props['PlaneStatsDiff']:.3f}",
                alignment=8,
            )

        return out_video

    differences = core.std.PlaneStats(video, video[0] + video)
    return core.std.FrameEval(video, selectFunc, prop_src=differences)


def fill_drops_mvtools(clip, threshold=0.1, debug=False):
    if not isinstance(clip, vs.VideoNode):
        raise ValueError("This is not a clip")

    differences = core.std.PlaneStats(clip, clip[0] + clip)

    pel = 4
    rfilter = 4
    sharp = 0
    blksize = 4
    overlap = 2
    search = 5
    searchparam = 3
    dct = 5

    super = core.mv.Super(
        clip, hpad=blksize, vpad=blksize, pel=pel, rfilter=rfilter, sharp=sharp
    )

    analyse_args = dict(
        blksize=blksize,
        overlap=overlap,
        search=search,
        searchparam=searchparam,
        dct=dct,
    )

    bv = core.mv.Analyse(super, isb=True, **analyse_args)
    fv = core.mv.Analyse(super, isb=False, **analyse_args)

    filldrops = core.mv.FlowInter(clip, super, mvbw=bv, mvfw=fv, ml=200)

    def selectFunc(n, f):
        if f.props["PlaneStatsDiff"] < threshold:
            if debug:
                return core.text.Text(
                    filldrops,
                    f"interpolated, diff: {f.props['PlaneStatsDiff']:.3f}",
                    alignment=8,
                )

            return filldrops
        else:
            return clip

    return core.std.FrameEval(clip, selectFunc, prop_src=differences)
