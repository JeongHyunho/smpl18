"""Shared fixtures: a minimal valid profile of each kind, written to a temporary directory."""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

SETTINGS = {"schema": "smpl24_settings_v1", "id": "t", "shape_fit": {"n_betas": 10},
            "discontinuity": {"gyro_abs_max_deg_s": 8000.0}}

CORRESPONDENCE = {"schema": "smpl24_correspondence_v1", "id": "t", "names": "opensim_joints",
                  "joints": {"pelvis": {"source": ["ground_pelvis"]}}, "trunk_body": "torso",
                  "aliases": {}, "root_translation": []}

PARAMETERS_PROFILE = {
    "schema": "smpl24_profile_v1",
    "id": "params_fixture",
    "source_kind": "smpl_parameters",
    "format": "npz",
    "layout": {"subject": "{study}/{subject}", "trial": "{study}/{subject}/{trial}_poses.npz",
               "exclude": ["shape.npz"]},
    "bindings": {
        "poses": {"field": "poses", "layout": "T,J*3"},
        "betas": {"field": "betas"},
        "trans": {"field": "trans"},
        "fps": {"field": "mocap_framerate", "aliases": ["mocap_frame_rate"],
                "fallback_key": "study", "fallback": {"A": 120.0, "B": 60.0}},
        "gender": {"field": "gender", "map": {"male": "male", "female": "female"},
                   "when_absent": "neutral"},
    },
    "conventions": {"up_axis": "z", "length_unit": "m", "angle_unit": "rad"},
    "shape": {"method": "parameters"},
    "root": {"placement": "source_translation"},
    "repairs": {"resample": {"fps": 100}},
    "settings": "settings.yaml",
}

SKELETON_PROFILE = {
    "schema": "smpl24_profile_v1",
    "id": "skeleton_fixture",
    "source_kind": "skeleton_motion",
    "format": "b3d",
    "layout": {"subject": "{study}/{subject}/*.b3d", "trial": "within_container",
               "skip_empty_files": True},
    "bindings": {
        "gender": {"field": "biological_sex",
                   "map": {"female": "female", "male": "male", "unknown": "neutral"},
                   "default": "neutral"},
        "frames": {"pass": "dynamics", "unfiltered_pass": "kinematics"},
        "root_translation": ["pelvis_tx", "pelvis_ty", "pelvis_tz"],
    },
    "conventions": {"up_axis": "y", "length_unit": "m", "angle_unit": "rad"},
    "correspondence": "correspondence.yaml",
    "shape": {"method": "bone_lengths", "landmark_offsets": "none"},
    "root": {"placement": "pelvis_centre", "alignment": "child_offsets"},
    "pose": {"method": "segment_rotation_transfer"},
    "repairs": {"wrap": {"coordinates": "any", "filter": "declared_by_source",
                         "declared_pass": "lowPassFilter"}},
    "skip": {"trials_without_pass": "dynamics"},
    "settings": ["settings.yaml"],
}


def write_yaml(path: pathlib.Path, data) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def parameters_profile() -> dict:
    return copy.deepcopy(PARAMETERS_PROFILE)


@pytest.fixture
def skeleton_profile() -> dict:
    return copy.deepcopy(SKELETON_PROFILE)


@pytest.fixture
def profile_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A directory holding the fixture profiles and the files they reference."""
    root = tmp_path / "profiles"
    write_yaml(root / "settings.yaml", SETTINGS)
    write_yaml(root / "correspondence.yaml", CORRESPONDENCE)
    write_yaml(root / "params.yaml", PARAMETERS_PROFILE)
    write_yaml(root / "skeleton.yaml", SKELETON_PROFILE)
    return root


@pytest.fixture
def no_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARED_DATASET_PATH", raising=False)
