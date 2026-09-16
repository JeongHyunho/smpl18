"""The examples are part of the documentation, so they run as tests: each one, on a few frames,
in a scratch directory, with the stand-in body."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(p for p in EXAMPLES.glob("0*.py") if not p.name.startswith("06_"))


def run(arguments):
    """Run a script in a child interpreter that writes UTF-8 whatever the console's encoding."""
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        check=False, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_the_example_runs(script, tmp_path) -> None:
    result = run([script, "--work", tmp_path, "--frames", "40"])
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "$ smpl18 convert" in result.stdout
    assert "stored pose vs" in result.stdout


def test_the_reading_example_runs(tmp_path) -> None:
    first = EXAMPLES / "05_smpl_parameters_to_smpl18.py"
    assert run([first, "--work", tmp_path, "--frames", "20"]).returncode == 0
    result = run([EXAMPLES / "06_read_the_corpus.py", tmp_path / "05_smpl" / "corpus"])
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "subject S05" in result.stdout and "rebuilt 24-joint rotations" in result.stdout


def test_every_example_is_listed_in_the_examples_readme() -> None:
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for script in EXAMPLES.glob("0*.py"):
        assert script.name in readme, script.name
