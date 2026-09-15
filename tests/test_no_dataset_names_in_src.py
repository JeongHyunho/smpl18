"""No dataset may be named in library code: a dataset is a profile, never a module.

Dataset ids belong in configs/ and in test fixtures. Their appearance anywhere under src/smpl24
(code, docstrings or comments) is the defect the plan's section 8 warns about.
"""

import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "smpl24"
DATASET_IDS = ("addbio", "addbiomechanics", "gaitex", "prism", "amass", "hknu")
PATTERN = re.compile("|".join(DATASET_IDS), re.IGNORECASE)
SOURCES = sorted(SRC.rglob("*.py"))


def test_the_source_tree_was_found() -> None:
    assert SOURCES, f"no Python files under {SRC}"


@pytest.mark.parametrize("path", SOURCES, ids=[str(p.relative_to(SRC)) for p in SOURCES])
def test_no_dataset_id_appears_in_src(path: pathlib.Path) -> None:
    offending = [
        f"{path.relative_to(SRC)}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if PATTERN.search(line)
    ]
    assert not offending, "dataset names belong in configs/ and tests/, not in src/:\n" + "\n".join(offending)
