"""Example 7 -- taking a corpus into Blender, and back to plain SMPL on the way.

A corpus stores 18 joints; renderers and viewers read the original 24-joint SMPL structure. Both
ways out are here. What it does:

1. converts a synthetic trial into a corpus (as example 5 does), with a body that has a surface;
2. writes the trial back as ordinary SMPL parameters with ``smpl18 export-smpl``: ``poses``
   [T, 72] axis-angle, ``betas``, ``trans``, ``mocap_framerate`` -- what any SMPL reader wants;
3. builds a Blender scene plan with ``smpl18 blender``, and runs Blender on it when you name one.

Run from the repository root::

    python examples/07_render_in_blender.py
    python examples/07_render_in_blender.py --blender "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"

Without ``--blender`` it stops at the plan and prints the command to run, which is the useful
half on a machine with no Blender. With it, Blender builds the scene, checks its own skinning
against this package's, saves a ``.blend`` and renders the frames.

With your own data: extract your SMPL models with ``smpl18 extract-model --with-mesh`` (the
skinning weights and the pose blend shapes live there, and a render needs them), then point
``--models`` at that directory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from _common import CONFIGS, SETTINGS, compare, prepare, run

from smpl18 import synthetic
from smpl18.blender.plan import read_plan
from smpl18.skeleton.kinematics import fk_batch, rest_joints

FPS = 60.0
NAME = "07_blender"


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, default=Path("example-output"),
                        help="where inputs, the corpus and the scene are written")
    parser.add_argument("--models", type=Path,
                        help="directory with SMPL models extracted --with-mesh; without it the "
                             "stand-in body (a mannequin of blocks) is written and used")
    parser.add_argument("--frames", type=int, default=120, help="frames in the synthetic trial")
    parser.add_argument("--blender", help="the Blender executable; without it the plan is written "
                                          "and the command printed")
    parser.add_argument("--correctives", action="store_true",
                        help="carry SMPL's pose blend shapes into the scene as 207 shape keys")
    return parser.parse_args()


def main() -> None:
    args = options()
    inputs, models, model = prepare(args, NAME, with_mesh=True)
    if not model.has_mesh:
        raise SystemExit(f"the models in {models} carry no surface; extract them again with "
                         "`smpl18 extract-model --with-mesh`")
    betas = np.array([0.4, -0.3, 0.2, 0.1, 0, 0, 0, 0, 0, 0])
    rest = rest_joints(model, betas)
    local, trans = synthetic.walking_motion(args.frames, FPS, seed=7)
    path = inputs / "walk.npz"
    np.savez(path, poses=synthetic.rotations_to_axis_angle(local), trans=trans, betas=betas,
             mocap_framerate=np.float64(FPS))
    print(f"wrote {path} ({args.frames} frames of walking, SMPL parameters)")

    work = args.work / NAME
    corpus = work / "corpus"
    run(["convert", "smpl", "--input", path, "--up-axis", "y",
         "--subject-id", "S07", "--gender", "neutral",
         "--settings", SETTINGS, "--models", models, "--out", corpus])
    compare(corpus, "S07", model, {"walk": fk_batch(rest, local, trans)[0]})

    # Out again as ordinary SMPL parameters: 24 joints, the frozen ones at the subject's constants.
    exported = work / "smpl"
    run(["export-smpl", "--corpus", corpus, "--subject", "S07", "--out", exported])
    with np.load(exported / "S07_walk.npz", allow_pickle=False) as data:
        print(f"  poses {data['poses'].shape}, betas {data['betas'].shape}, "
              f"trans {data['trans'].shape} at {float(data['mocap_framerate']):g} Hz")

    # And into Blender: the plan holds the surface, the weights and each joint's transform.
    scene = work / "scene"
    command = ["blender", "--corpus", corpus, "--subject", "S07", "--trial", "walk",
               "--models", models, "--out", scene,
               "--render-settings", CONFIGS / "render" / "default.yaml", "--blend", "--render"]
    if args.correctives:
        command.append("--correctives")
    if args.blender:
        command += ["--blender", args.blender]
    run(command)

    plan = read_plan(scene / "S07_walk.plan.npz")
    print(f"  the plan holds {plan.frames} frames of {plan.num_vertices} vertices "
          f"({plan.transforms.nbytes / 1e6:.1f} MB of joint transforms), checked at frames "
          f"{plan.sample_frames.tolist()} to {plan.tolerance * 1000:.3f} mm")
    if not args.blender:
        print("\nno --blender given: the scene was not built. Install Blender and pass its "
              "executable to see the body move.", file=sys.stderr)


if __name__ == "__main__":
    main()
