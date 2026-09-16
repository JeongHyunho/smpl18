"""No dataset may be named in library code: a dataset is a profile, never a module.

The ids come from the profiles in configs/, so the guard grows with them and names no dataset
of its own. Their appearance anywhere under src/smpl24 (code, docstrings or comments) is the
risk the plan's Risks section warns about.
"""

import pathlib
import re

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src" / "smpl24"
#: Every shipped profile id, plus the stems of the tables the profiles point at, so a name that
#: belongs to a dataset cannot reach the library through either route.
DATASET_IDS = tuple(sorted({
    part
    for directory in ("profiles", "correspondence", "offsets", "settings", "markersets")
    for path in (PACKAGE / "configs" / directory).glob("*.yaml")
    for part in path.stem.split("_")
    if len(part) > 3 and part not in {"default", "landmarks", "rajagopal", "opensim"}
}))
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
