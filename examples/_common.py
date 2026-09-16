"""Shared plumbing for the examples: arguments, the work directory, the body model, running a
command and checking the corpus it wrote against the motion the example started from.

Each example writes synthetic input files, then converts them with exactly the ``smpl18``
command a user would type (printed before it runs), then reads the corpus back.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs"
SETTINGS = CONFIGS / "settings" / "default.yaml"

sys.path.insert(0, str(REPOSITORY / "src"))  # run from a checkout without installing

from smpl18.cli import main as smpl18  # noqa: E402
from smpl18.corpus import read_corpus  # noqa: E402
from smpl18.model.demo import write_models  # noqa: E402
from smpl18.model.load import Model  # noqa: E402


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--work", type=Path, default=Path("example-output"),
                        help="where inputs and the corpus are written (default: ./example-output)")
    parser.add_argument("--models", type=Path,
                        help="directory with your extracted SMPL models; without it the "
                             "stand-in body is written and used")
    parser.add_argument("--frames", type=int, default=300, help="frames per synthetic trial")
    return parser.parse_args()


def prepare(args: argparse.Namespace, name: str) -> tuple[Path, Path, Model]:
    """``(inputs directory, models directory, neutral model)`` for one example."""
    work = args.work / name
    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    models = args.models
    if models is None:
        models = work / "models"
        write_models(models)
        print(f"using the stand-in body in {models} (pass --models for real SMPL models)")
    return inputs, models, Model.for_gender("neutral", root=models)


def run(command: list[str]) -> None:
    """Print the command as a user would type it, then run it in this process."""
    printable = ["smpl18"] + [str(part) for part in command]
    print("\n$ " + " ".join(shlex.quote(part) for part in printable) + "\n")
    code = smpl18([str(part) for part in command])
    if code != 0:
        raise SystemExit(f"smpl18 exited with {code}")


def compare(corpus_dir: Path, subject_id: str, model: Model, truth: dict[str, np.ndarray],
            joints=range(22)) -> None:
    """Joint-centre distance between the stored 18-joint poses (rebuilt to 24) and the truth."""
    subject = read_corpus(corpus_dir).subject(subject_id)
    joints = list(joints)
    for trial in subject.trials():
        if trial.id not in truth:
            continue
        rebuilt = trial.joints_world(model)[:, joints]
        distance = np.linalg.norm(rebuilt - truth[trial.id][:, joints], axis=2)
        print(f"  {trial.id}: stored pose vs the motion it came from: median "
              f"{np.median(distance) * 1000:.1f} mm, 95th percentile "
              f"{np.percentile(distance, 95) * 1000:.1f} mm")
