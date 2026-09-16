"""Example 2 -- you only have kinematics from OpenSim (.osim + .mot).

What it does:

1. writes an OpenSim 4 model with Rajagopal-style joint and body names, and a walking
   coordinate file for it in degrees, as OpenSim's inverse kinematics writes them;
2. converts it with ``smpl18 convert opensim`` and the shipped Rajagopal correspondence;
3. compares the stored poses with the model's own joint centres, which is what the conversion
   aims at.

Run from the repository root::

    python examples/02_opensim_to_smpl18.py

With your own data: ``--osim`` is the scaled model your IK used, ``--mot`` the IK results (one
file per trial), and the correspondence names your model's joints (the shipped one covers the
Rajagopal and gait2392 families; copy and edit it for another model). The up axis comes from the
model's gravity, and degrees or radians from the file's ``inDegrees`` header.
"""

import numpy as np
from _common import CONFIGS, SETTINGS, arguments, prepare, run

from smpl18 import synthetic
from smpl18.corpus import read_corpus
from smpl18.formats import mot
from smpl18.skeleton.definition import JOINT_NAMES
from smpl18.skeleton.kinematics import rest_joints
from smpl18.sources.skeleton import opensim

FPS = 100.0
#: The model's joints that stand for SMPL joints, to compare against.
CENTRES = {
    "left_hip": "hip_l", "right_hip": "hip_r", "left_knee": "walker_knee_l",
    "right_knee": "walker_knee_r", "left_ankle": "ankle_l", "right_ankle": "ankle_r",
    "left_shoulder": "acromial_l", "right_shoulder": "acromial_r",
    "left_elbow": "elbow_l", "right_elbow": "elbow_r",
    "left_wrist": "radius_hand_l", "right_wrist": "radius_hand_r",
}


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    inputs, models, model = prepare(args, "02_opensim")
    # The OpenSim body is a little different from the SMPL mean body, as a scaled model is.
    source_rest = rest_joints(model, np.array([0.8, 0.4, -0.3, 0.2, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0]))
    osim_path = synthetic.write_opensim_model(inputs / "subject02_scaled.osim", source_rest)
    print(f"wrote {osim_path}")
    mot_paths = []
    for seed in (1, 2):
        values = synthetic.opensim_motion(args.frames, FPS, seed=seed)
        path = synthetic.write_mot(inputs / f"walk_{seed}_ik.mot", values, FPS, in_degrees=True)
        mot_paths.append(path)
        print(f"wrote {path} ({len(values)} coordinates, {args.frames} frames)")

    corpus = args.work / "02_opensim" / "corpus"
    run(["convert", "opensim",
         "--osim", osim_path,
         "--mot", *mot_paths,
         "--correspondence", CONFIGS / "correspondence" / "opensim_rajagopal.yaml",
         "--subject-id", "S02", "--gender", "neutral",
         "--settings", SETTINGS,
         "--models", models,
         "--out", corpus])
    run(["info", corpus])

    # The model's own joint centres, turned from its Y-up frame into the corpus frame (also Y-up
    # here, so the gravity rotation is the identity).
    skeleton = opensim.read(osim_path)
    subject = read_corpus(corpus).subject("S02")
    joints = [JOINT_NAMES.index(name) for name in CENTRES]
    for trial in subject.trials():
        values, _ = opensim.coordinates_from_mot(skeleton, mot.read(inputs / f"{trial.id}.mot"))
        placed = skeleton.motion(values)
        source = np.stack([placed.joint_centres[CENTRES[name]] for name in CENTRES], axis=1)
        stored = trial.joints_world(model)[:, joints]
        distance = np.linalg.norm(stored - source, axis=2)
        print(f"  {trial.id}: stored pose vs the model's joint centres: median "
              f"{np.median(distance) * 1000:.1f} mm, 95th percentile "
              f"{np.percentile(distance, 95) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
