"""Example 5 -- you already have SMPL / SMPL-H / SMPL-X parameters (.npz).

Nothing needs fitting: the 22 body joints are kept, the up axis is changed, and the 18-joint
reduction is applied. What it does:

1. writes an SMPL-H style ``.npz`` (``poses`` [T, 156], ``trans``, ``betas`` [16],
   ``mocap_framerate``) in a Z-up world;
2. converts it with ``smpl18 convert smpl``;
3. checks that the stored poses reproduce the joints of the original parameters (the only
   difference is what the 18-joint freeze costs, which the subject record reports).

Run from the repository root::

    python examples/05_smpl_parameters_to_smpl18.py

With your own data: name the arrays with ``--poses-key``, ``--trans-key``, ``--betas-key`` if
they differ, and give ``--fps`` when the file stores no rate.
"""

import numpy as np
from _common import SETTINGS, arguments, compare, prepare, run

from smpl18 import synthetic
from smpl18.skeleton.frames import frame_change
from smpl18.skeleton.kinematics import fk_batch, rest_joints

FPS = 120.0


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    inputs, models, model = prepare(args, "05_smpl")
    betas = np.zeros(16)
    betas[:4] = [0.3, -0.2, 0.1, 0.4]
    rest = rest_joints(model, betas)
    local, trans = synthetic.walking_motion(args.frames, FPS, seed=6)
    # Express the motion in a Z-up world, the way the parameters would have been stored.
    to_z_up = frame_change("y", "z")
    z_local, z_trans = to_z_up.apply_to_pose(local, trans, pelvis_rest=rest[0])
    poses = np.zeros((args.frames, 52, 3))
    poses[:, :24] = synthetic.rotations_to_axis_angle(z_local)
    poses[:, 22:24] = 0.0                               # SMPL-H: joints 22+ are the fingers
    path = inputs / "dance_6_poses.npz"
    np.savez(path, poses=poses.reshape(args.frames, -1), trans=z_trans, betas=betas,
             mocap_framerate=np.float64(FPS), gender=np.array("neutral"))
    print(f"wrote {path} (SMPL-H layout, {args.frames} frames)")

    corpus = args.work / "05_smpl" / "corpus"
    run(["convert", "smpl",
         "--input", path,
         "--up-axis", "z",
         "--subject-id", "S05", "--gender", "neutral",
         "--settings", SETTINGS,
         "--models", models,
         "--out", corpus])
    run(["info", corpus])
    # The corpus is Y-up, like the motion before it was stored Z-up. Hands are not compared:
    # an SMPL-H file's joints 22 and 23 are fingers, not SMPL's hands.
    truth_local = local.copy()
    truth_local[:, 22:24] = np.eye(3)
    compare(corpus, "S05", model, {"dance_6_poses": fk_batch(rest, truth_local, trans)[0]})


if __name__ == "__main__":
    main()
