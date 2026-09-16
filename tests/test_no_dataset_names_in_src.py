"""No dataset may be named in library code: a dataset is a profile, never a module.

The ids come from the shipped dataset profiles and the settings files named after them, so the
guard grows with them and names no dataset of its own. Marker sets and correspondence tables are
not read: they are named after capture protocols and naming schemes shared by many datasets.
Their appearance anywhere under src/smpl18 (code, docstrings or comments) is the risk the plan's
Risks section warns about. Whole words are matched, so an id cannot hide inside a longer word
and a shorter word cannot trip the guard.
"""

import pathlib
import re

import pytest
import yaml

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
SRC = PACKAGE / "src" / "smpl18"
#: Settings files every profile shares; their ids name no dataset.
SHARED_SETTINGS = {"default"}


def _ids() -> tuple[str, ...]:
    found = set()
    for directory in ("profiles", "settings"):
        for path in (PACKAGE / "configs" / directory).glob("*.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            found.add(str(data["id"]))
    return tuple(sorted(found - SHARED_SETTINGS))


DATASET_IDS = _ids()
PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, DATASET_IDS)) + r")\b", re.IGNORECASE)
SOURCES = sorted(SRC.rglob("*.py"))


def test_the_source_tree_and_the_ids_were_found() -> None:
    assert SOURCES, f"no Python files under {SRC}"
    assert DATASET_IDS, "no dataset profile found under configs/"


@pytest.mark.parametrize("path", SOURCES, ids=[str(p.relative_to(SRC)) for p in SOURCES])
def test_no_dataset_id_appears_in_src(path: pathlib.Path) -> None:
    offending = [
        f"{path.relative_to(SRC)}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if PATTERN.search(line)
    ]
    assert not offending, "dataset names belong in configs/ and tests/, not in src/:\n" + "\n".join(offending)
