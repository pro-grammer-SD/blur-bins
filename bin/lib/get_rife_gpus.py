import vapoursynth as vs
from vapoursynth import core

import sys
from pathlib import Path

if vars().get("macos_bundled") == "true":
    # load plugins
    plugin_dir = Path("../vapoursynth-plugins")
    ignored = {
        "libbestsource.dylib",
    }

    for dylib in plugin_dir.glob("*.dylib"):
        if dylib.name not in ignored:
            print("loading", dylib.name)
            core.std.LoadPlugin(path=str(dylib))

# add blur.py folder to path so it can reference scripts
sys.path.insert(1, str(Path(__file__).parent))

video = core.std.BlankClip(
    width=1, height=1, length=2, fpsnum=1, fpsden=1, format=vs.RGBS
)

core.rife.RIFE(video, list_gpu=True)
