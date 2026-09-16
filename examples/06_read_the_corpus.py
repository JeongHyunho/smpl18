"""Example 6 -- reading a corpus back.

Run one of the conversion examples first, then::

    python examples/06_read_the_corpus.py example-output/01_markers/corpus

It prints what a corpus stores per subject and trial, rebuilds the 24-joint pose from the 18
stored joints and the subject's frozen constants, and runs forward kinematics on it.
"""

import argparse
from pathlib import Path

import numpy as np
from _common import read_corpus

from smpl18.model.load import Model
from smpl18.skeleton.definition import FROZEN_JOINT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--models", type=Path,
                        help="the model directory the corpus was made with (default: the "
                             "example's own models directory beside the corpus)")
    args = parser.parse_args()
    corpus = read_corpus(args.corpus)
    models = args.models or args.corpus.parent / "models"
    for subject in corpus.subjects():
        record = subject.record
        print(f"subject {subject.id}: gender {subject.gender}, betas {np.round(subject.betas, 2)}")
        reduced = record["reduced_model"]
        for name, vector in zip(FROZEN_JOINT_NAMES, reduced["constants"]):
            print(f"  frozen {name:13s} constant {np.degrees(np.linalg.norm(vector)):5.1f} deg "
                  f"-> absorbed by {reduced['absorbed_by'][name]}")
        print(f"  the freeze costs {reduced['fit']['residual_rms_m'] * 1000:.1f} mm RMS "
              f"(max {reduced['fit']['residual_max_m'] * 1000:.1f} mm)")
        model = Model.for_gender(subject.gender, root=models)
        for trial in subject.trials():
            print(f"  trial {trial.id}: poses {trial.poses.shape} ({', '.join(trial.joint_names[:3])}"
                  f", ...), trans {trial.trans.shape}, {trial.fps:g} Hz, {trial.up_axis}-up")
            print(f"    provenance: {dict(zip(trial.joint_names, trial.joint_provenance))}")
            local = trial.local_rotations_24()           # (T, 24, 3, 3)
            joints = trial.joints_world(model)           # (T, 24, 3)
            print(f"    rebuilt 24-joint rotations {local.shape}, joint centres {joints.shape}; "
                  f"pelvis travels {np.linalg.norm(joints[-1, 0] - joints[0, 0]):.2f} m")


if __name__ == "__main__":
    main()
