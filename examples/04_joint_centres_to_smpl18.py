"""Example 4 -- you have joint-centre trajectories (LHJC, LKJC, ... as a .trc, .c3d or .npz).

Motion-analysis software often exports the joint centres it computed rather than (or beside)
the markers. What it does:

1. writes the centres of a walking body as a Z-up millimetre ``.trc`` and, for the same trial,
   as an ``.npz`` (names, positions, fps, units);
2. converts the ``.trc`` with ``smpl18 convert centres`` and the shipped label table;
3. compares the stored poses with the motion the centres came from.

Run from the repository root::

    python examples/04_joint_centres_to_smpl18.py

With your own data: edit configs/correspondence/joint_centre_labels.yaml (a copy of it) so each
SMPL joint names your export's label, and lower the weight of centres your software places
differently from SMPL (trunk, neck, head). If the export also stores segment rotations, put
them in the ``.npz`` (``segment_names``, ``segment_rotations`` [T, M, 3, 3]) and name them in the
table with ``segment:``; they then steer each segment's twist.
"""

import numpy as np
from _common import CONFIGS, SETTINGS, arguments, compare, prepare, run

from smpl18 import synthetic
from smpl18.formats import trc
from smpl18.skeleton.kinematics import fk_batch, rest_joints

FPS = 120.0
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    inputs, models, model = prepare(args, "04_centres")
    rest = rest_joints(model, np.array([0.2, 0.5, -0.1, 0.3, 0.0, -0.2, 0.1, 0.0, -0.3, 0.2]))
    local, trans = synthetic.walking_motion(args.frames, FPS, seed=5)
    labels, centres = synthetic.joint_centres(rest, local, trans)
    z_up_mm = (centres @ Y_UP_TO_Z_UP.T) * 1000.0
    trc_path = trc.write(inputs / "run_5.trc", labels, z_up_mm, rate_hz=FPS, units="mm")
    npz_path = inputs / "run_5_centres.npz"
    np.savez(npz_path, names=np.array(labels), positions=z_up_mm, fps=np.float64(FPS),
             units=np.array("mm"))
    print(f"wrote {trc_path} and {npz_path} ({len(labels)} centres, {args.frames} frames)")

    corpus = args.work / "04_centres" / "corpus"
    run(["convert", "centres",
         "--input", trc_path,
         "--correspondence", CONFIGS / "correspondence" / "joint_centre_labels.yaml",
         "--up-axis", "z",
         "--subject-id", "S04", "--gender", "neutral",
         "--settings", SETTINGS,
         "--models", models,
         "--out", corpus])
    run(["info", corpus])
    compare(corpus, "S04", model, {"run_5": fk_batch(rest, local, trans)[0]})


if __name__ == "__main__":
    main()
