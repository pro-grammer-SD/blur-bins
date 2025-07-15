# credit to InterFrame - https://www.spirton.com/uploads/InterFrame/InterFrame2.html and https://github.com/HomeOfVapourSynthEvolution/havsfunc

import vapoursynth as vs
from vapoursynth import core

import json
import math

import blur.utils as u

LEGACY_PRESETS = ["weak", "film", "smooth", "animation"]
NEW_PRESETS = ["default", "test"]

DEFAULT_PRESET = "weak"
DEFAULT_ALGORITHM = 13
DEFAULT_BLOCKSIZE = 8
DEFAULT_OVERLAP = 2
DEFAULT_SPEED = "medium"
DEFAULT_MASKING = 50
DEFAULT_GPU = True


def generate_svp_strings(
    new_fps,
    preset=DEFAULT_PRESET,
    algorithm=DEFAULT_ALGORITHM,
    blocksize=DEFAULT_BLOCKSIZE,
    overlap=DEFAULT_OVERLAP,
    speed=DEFAULT_SPEED,
    masking=DEFAULT_MASKING,
    gpu=DEFAULT_GPU,
    scene_detect=False,
):
    # build super json
    super_json = {
        "pel": 1,
        "gpu": gpu,
    }

    # build vectors json
    vectors_json = {
        "block": {
            "w": blocksize,
            "overlap": overlap,
        }
    }

    match preset:
        case "test":
            vectors_json["main"] = {
                "search": {"type": 3, "satd": True, "coarse": {"type": 3}}
            }
        case _ if preset in LEGACY_PRESETS:
            vectors_json["main"] = {"search": {"distance": 0, "coarse": {}}}

            if preset == "weak":
                vectors_json["main"]["search"]["coarse"] = {
                    "distance": -1,
                    "trymany": True,
                    "bad": {"sad": 2000},
                }
            else:
                vectors_json["main"]["search"]["coarse"] = {"distance": -10}

    # build smooth json
    smooth_json = {
        "rate": {"num": int(new_fps), "abs": True},
        "algo": algorithm,
        "mask": {
            "area": masking,
            "area_sharp": 1.2,  # test if this does anything
        },
    }

    if not scene_detect:
        # dont want any scene detection stuff when normally blurring (i think?)
        smooth_json["scene"] = {
            "blend": False,
            "mode": 0,
            "limits": {"blocks": 9999999},
        }

    return [json.dumps(obj) for obj in [super_json, vectors_json, smooth_json]]


def interpolate_svp(
    video,
    new_fps,
    preset=DEFAULT_PRESET,
    algorithm=DEFAULT_ALGORITHM,
    blocksize=DEFAULT_BLOCKSIZE,
    overlap=DEFAULT_OVERLAP,
    speed=DEFAULT_SPEED,
    masking=DEFAULT_MASKING,
    gpu=DEFAULT_GPU,
):
    if not isinstance(video, vs.VideoNode):
        raise vs.Error("interpolate: input not a video")

    preset = preset.lower()

    if preset not in LEGACY_PRESETS and preset not in NEW_PRESETS:
        raise vs.Error(f"interpolate: '{preset}' is not a valid preset")

    # generate svp strings
    [super_string, vectors_string, smooth_string] = generate_svp_strings(
        new_fps, preset, algorithm, blocksize, overlap, speed, masking, gpu
    )

    # interpolate
    super = core.svp1.Super(video, super_string)
    vectors = core.svp1.Analyse(super["clip"], super["data"], video, vectors_string)

    return core.svp2.SmoothFps(
        video,
        super["clip"],
        super["data"],
        vectors["clip"],
        vectors["data"],
        smooth_string,
        src=video,
        fps=video.fps,
    )


def change_fps(clip, fpsnum, fpsden=1):  # this is just directly from havsfunc
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ChangeFPS: This is not a clip")

    factor = (fpsnum / fpsden) * (clip.fps_den / clip.fps_num)

    def frame_adjuster(n):
        real_n = math.floor(n / factor)
        one_frame_clip = clip[real_n] * (len(clip) + 100)
        return one_frame_clip

    attribute_clip = clip.std.BlankClip(
        length=math.floor(len(clip) * factor), fpsnum=fpsnum, fpsden=fpsden
    )
    return attribute_clip.std.FrameEval(eval=frame_adjuster)


def interpolate_mvtools(
    clip,
    new_fps,
    blocksize=DEFAULT_BLOCKSIZE,
    masking=100,
    pel=1,
    sharp=0,
    overlap=DEFAULT_OVERLAP,
    search=5,
    searchparam=3,
    pelsearch=1,
    dct=3,
    blend=False,
):
    super = core.mv.Super(
        clip, hpad=blocksize, vpad=blocksize, pel=pel, rfilter=1, sharp=sharp
    )

    analyse_args = dict(
        blksize=blocksize,
        overlap=overlap,
        search=search,
        searchparam=searchparam,
        pelsearch=pelsearch,
        dct=dct,
    )

    bv = core.mv.Analyse(super, isb=True, **analyse_args)
    fv = core.mv.Analyse(super, isb=False, **analyse_args)

    return core.mv.FlowFPS(
        clip, super, bv, fv, num=new_fps, den=1, blend=blend, ml=max(masking, 1)
    )


def interpolate_rife(video, new_fps: int, model_path: str, gpu_index=int):
    u.check_model_path(model_path)

    orig_format = video.format
    needs_conversion = orig_format.id != vs.RGBS

    if needs_conversion:
        video = core.resize.Bicubic(
            video,
            format=vs.RGBS,
            matrix_in_s="709" if orig_format.color_family == vs.YUV else None,
        )

    video = core.rife.RIFE(
        video,
        fps_num=new_fps,
        fps_den=1,
        model_path=model_path,
        gpu_id=gpu_index,
    )

    if needs_conversion:
        video = core.resize.Bicubic(
            video,
            format=orig_format.id,
            matrix_s="709" if orig_format.color_family == vs.YUV else None,
        )

    return video
