"""Example 3 -- you only have an animation skeleton (.bvh, or .fbx through Blender).

What it does:

1. writes a BVH clip in centimetres on a rig with the common humanoid joint names (Hips, Spine,
   LeftUpLeg, LeftArm, ...), each joint with a Z-X-Y rotation triple;
2. converts it with ``smpl18 convert bvh`` and the shipped humanoid correspondence;
3. compares the stored poses with the motion the clip was written from.

Run from the repository root::

    python examples/03_bvh_to_smpl18.py

With your own data: give the clip's length unit (``--length-unit``) and up axis (``--up-axis``)
explicitly; BVH files do not state them. For FBX, export BVH first with
``smpl18 fbx2bvh --input clip.fbx --out clip.bvh --blender <path to blender>``, which writes
Blender's frame (Z up, metres). If your rig's joint names differ, copy
configs/correspondence/bvh_humanoid.yaml and edit the names, prefixes or aliases.
"""

import numpy as np
from _common import CONFIGS, SETTINGS, arguments, compare, prepare, run

from smpl18 import synthetic
from smpl18.skeleton.kinematics import fk_batch, rest_joints

FPS = 60.0


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    inputs, models, model = prepare(args, "03_bvh")
    rest = rest_joints(model, np.array([-0.4, 0.3, 0.2, -0.2, 0.1, 0.2, 0.0, 0.1, 0.0, 0.0]))
    truth, files = {}, []
    for seed in (3, 4):
        local, trans = synthetic.walking_motion(args.frames, FPS, seed=seed)
        path = synthetic.write_bvh(inputs / f"clip_{seed}.bvh", rest, local, trans, FPS,
                                   units_per_metre=100.0)
        files.append(path)
        truth[path.stem] = fk_batch(rest, local, trans)[0]
        print(f"wrote {path} ({args.frames} frames at {FPS:g} Hz)")

    corpus = args.work / "03_bvh" / "corpus"
    run(["convert", "bvh",
         "--input", *files,
         "--correspondence", CONFIGS / "correspondence" / "bvh_humanoid.yaml",
         "--up-axis", "y",
         "--length-unit", "cm",
         "--subject-id", "S03", "--gender", "neutral",
         "--settings", SETTINGS,
         "--models", models,
         "--out", corpus])
    run(["info", corpus])
    compare(corpus, "S03", model, truth)


if __name__ == "__main__":
    main()
