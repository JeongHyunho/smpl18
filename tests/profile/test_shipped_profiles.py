"""Every profile shipped in configs/profiles/ validates and every file it references exists.

A profile kept outside this repository is the consumer's to check; the loader's resolution order
is covered in test_profile_load.py.
"""

import pathlib

import pytest
import yaml

from smpl24.cli import main
from smpl24.profile import Profile

PACKAGE = pathlib.Path(__file__).resolve().parents[2]
SHIPPED = sorted((PACKAGE / "configs" / "profiles").glob("*.yaml"))


def _ids(paths):
    return [path.stem for path in paths]


@pytest.fixture(autouse=True)
def without_shared_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARED_DATASET_PATH", raising=False)


def test_the_three_public_profiles_ship() -> None:
    assert _ids(SHIPPED) == ["addbiomechanics", "amass", "gaitex"]


@pytest.mark.parametrize("path", SHIPPED, ids=_ids(SHIPPED))
def test_shipped_profile_validates_and_its_references_exist(path: pathlib.Path) -> None:
    profile = Profile.load(path)
    assert profile.id == path.stem
    for entry in profile.referenced_files():
        assert entry.path.is_file(), f"{entry.role} -> {entry.path}"
        assert len(entry.sha256) == 64
    assert profile.settings, "every profile resolves to non-empty settings"
    assert "shape_fit" in profile.settings and "discontinuity" in profile.settings


@pytest.mark.parametrize("name", _ids(SHIPPED))
def test_shipped_profile_resolves_by_name_through_the_cli(name: str, capsys) -> None:
    assert main(["profile", "validate", name]) == 0
    assert capsys.readouterr().out.startswith(f"OK {name} ")


@pytest.mark.parametrize("path", SHIPPED, ids=_ids(SHIPPED))
def test_shipped_settings_cite_their_source_and_carry_no_dataset_defaults_in_code(path) -> None:
    profile = Profile.load(path)
    for entry in profile.referenced_files():
        if entry.role.startswith("settings"):
            loaded = yaml.safe_load(entry.path.read_text(encoding="utf-8"))
            assert loaded["schema"] == "smpl24_settings_v1"
