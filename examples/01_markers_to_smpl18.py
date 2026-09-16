"""Example 1 -- you only have labelled markers (.trc or .c3d).

What it does:

1. walks a body for two trials and places the 36 markers of the conventional full-body set on
   it, then writes them as Z-up, millimetre ``.trc`` files, the way lab exports usually are;
2. writes a subject file with the measurements the marker set asks for;
3. converts both trials with ``smpl18 convert markers``;
4. compares the stored 18-joint poses with the motion the markers were made from.

Run from the repository root::

    python examples/01_markers_to_smpl18.py
    python examples/01_markers_to_smpl18.py --models ~/smpl18-models   # your SMPL models

With your own data, only step 3 matters: point ``--input`` at your files, write your subject's
measurements, and use a marker set that matches your labels (see configs/markersets/).
The synthetic markers here obey the marker set's rules exactly; skin-mounted markers do not,
so expect larger errors on real data.
"""

import numpy as np
import yaml
from _common import CONFIGS, SETTINGS, arguments, compare, prepare, run

from smpl18 import synthetic
from smpl18.formats import trc
from smpl18.skeleton.kinematics import fk_batch, rest_joints

FPS = 100.0
#: SMPL's frame is Y-up; these files are Z-up, as most motion-capture exports are.
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def main() -> None:
    args = arguments(__doc__.splitlines()[0])
    inputs, models, model = prepare(args, "01_markers")
    rest = rest_joints(model, np.array([0.5, -0.3, 0.2, 0.4, -0.4, 0.3, 0.1, -0.2, 0.3, -0.1]))

    truth, files = {}, []
    for seed in (1, 2):
        local, trans = synthetic.walking_motion(args.frames, FPS, seed=seed)
        labels, markers = synthetic.conventional_markers(rest, local, trans)
        path = inputs / f"walk_{seed}.trc"
        trc.write(path, labels, (markers @ Y_UP_TO_Z_UP.T) * 1000.0, rate_hz=FPS, units="mm")
        files.append(path)
        truth[path.stem] = fk_batch(rest, local, trans)[0]
        print(f"wrote {path} ({len(labels)} markers, {args.frames} frames)")

    subject = inputs / "S01.yaml"
    subject.write_text(yaml.safe_dump({
        "schema": "smpl18_subject_v1",
        "id": "S01",
        "gender": "neutral",
        "measurements": synthetic.MEASUREMENTS,
    }, sort_keys=False), encoding="utf-8")
    print(f"wrote {subject}")

    corpus = args.work / "01_markers" / "corpus"
    run(["convert", "markers",
         "--input", *files,
         "--markerset", CONFIGS / "markersets" / "conventional_full_body.yaml",
         "--up-axis", "z",
         "--subject", subject,
         "--settings", SETTINGS,
         "--models", models,
         "--out", corpus])
    run(["info", corpus])
    compare(corpus, "S01", model, truth)


if __name__ == "__main__":
    main()
