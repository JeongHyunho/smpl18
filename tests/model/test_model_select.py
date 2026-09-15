"""Model selection: three exact gender names, one root from the argument or the environment, no default path."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from smpl24.model import (
    ENV_MODELS,
    GENDERS,
    Model,
    ModelRootUnset,
    UnresolvedGender,
    model_filename,
    model_path_for_gender,
    model_root,
    set_hashes,
)

SOURCES = Path(__file__).resolve().parents[2] / "src" / "smpl24"


@pytest.fixture(autouse=True)
def no_models_env(monkeypatch):
    monkeypatch.delenv(ENV_MODELS, raising=False)


def test_the_three_genders_map_to_the_set_file_names():
    assert GENDERS == ("male", "female", "neutral")
    assert model_filename("male") == "SMPL_MALE_clean.npz"
    assert model_filename("female") == "SMPL_FEMALE_clean.npz"
    assert model_filename("neutral") == "SMPL_NEUTRAL_clean.npz"


@pytest.mark.parametrize("gender", ["M", "F", "Male", "unknown", "", "x"])
def test_anything_but_the_three_names_is_unresolved(gender, tmp_path):
    with pytest.raises(UnresolvedGender, match="no body model"):
        model_path_for_gender(gender, tmp_path)


def test_with_neither_argument_nor_environment_the_call_fails(tmp_path):
    with pytest.raises(ModelRootUnset) as info:
        model_root()
    assert "--models" in str(info.value) and ENV_MODELS in str(info.value)
    with pytest.raises(ModelRootUnset):
        model_path_for_gender("male")
    with pytest.raises(ModelRootUnset):
        Model.for_gender("male")
    with pytest.raises(ModelRootUnset):
        set_hashes()


def test_an_empty_environment_value_counts_as_unset(monkeypatch):
    monkeypatch.setenv(ENV_MODELS, "")
    with pytest.raises(ModelRootUnset):
        model_root()


def test_the_root_argument_wins_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_MODELS, str(tmp_path / "from_env"))
    assert model_root() == tmp_path / "from_env"
    assert model_root(tmp_path / "given") == tmp_path / "given"
    assert model_path_for_gender("female", tmp_path / "given") == tmp_path / "given" / "SMPL_FEMALE_clean.npz"
    assert model_path_for_gender("neutral") == tmp_path / "from_env" / "SMPL_NEUTRAL_clean.npz"


def test_paths_are_not_checked_for_existence(tmp_path):
    assert not model_path_for_gender("male", tmp_path).exists()


def test_set_hashes_reports_every_file_of_the_set(model_dir, monkeypatch):
    hashes = set_hashes(model_dir)
    assert set(hashes) == {"SMPL_MALE_clean.npz", "SMPL_FEMALE_clean.npz", "SMPL_NEUTRAL_clean.npz"}
    for name in ("SMPL_MALE_clean.npz", "SMPL_FEMALE_clean.npz"):
        assert hashes[name] == hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
    assert hashes["SMPL_NEUTRAL_clean.npz"] is None
    monkeypatch.setenv(ENV_MODELS, str(model_dir))
    assert set_hashes() == hashes


def test_model_for_gender_loads_the_selected_file(model_dir, monkeypatch):
    model = Model.for_gender("male", model_dir)
    assert model.gender == "male" and model.path == model_dir / "SMPL_MALE_clean.npz"
    assert model.sha256 == set_hashes(model_dir)["SMPL_MALE_clean.npz"]
    monkeypatch.setenv(ENV_MODELS, str(model_dir))
    assert Model.for_gender("female").path == model_dir / "SMPL_FEMALE_clean.npz"
    with pytest.raises(FileNotFoundError):
        Model.for_gender("neutral")


def test_no_machine_path_and_no_dataset_name_in_the_moved_sources():
    files = [
        SOURCES / "formats" / "pickle_safe.py",
        *(SOURCES / "model").glob("*.py"),
        *(SOURCES / "skeleton").glob("*.py"),
    ]
    drive = re.compile(r"[A-Za-z]:\\")
    datasets = re.compile(r"addbio|addbiomechanics|gaitex|prism|amass|hknu", re.IGNORECASE)
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not drive.search(text), f"machine path in {path.name}"
        assert not datasets.search(text), f"dataset name in {path.name}"
