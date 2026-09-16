import copy
import hashlib
import pathlib

import pytest

from smpl18.profile import (
    EnvironmentVariableUnset,
    Profile,
    ProfileNotFound,
    ProfileSchemaError,
    ReferenceNotFound,
    resolve_profile_path,
)
from smpl18.profile import load as load_module
from smpl18.profile.load import substitute_env
from smpl18.sources import Format, SourceKind

from .conftest import SETTINGS, write_yaml


@pytest.fixture
def fake_package(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """A stand-in for the package's configs/ directory, so tests do not depend on shipped files."""
    configs = tmp_path / "package_configs"
    (configs / "profiles").mkdir(parents=True)
    monkeypatch.setattr(load_module, "package_configs_dir", lambda: configs)
    return configs


# ---- resolution order ------------------------------------------------------------------------


def test_an_existing_path_wins(profile_dir, fake_package, no_shared) -> None:
    resolved = resolve_profile_path(profile_dir / "params.yaml")
    assert resolved == (profile_dir / "params.yaml").resolve()


def test_a_name_resolves_in_the_package_before_the_shared_drive(
    profile_dir, fake_package, monkeypatch, tmp_path
) -> None:
    write_yaml(fake_package / "profiles" / "demo.yaml", {"from": "package"})
    shared = tmp_path / "shared"
    write_yaml(shared / "smpl18" / "profiles" / "demo.yaml", {"from": "shared"})
    monkeypatch.setenv("SHARED_DATASET_PATH", str(shared))
    assert resolve_profile_path("demo") == (fake_package / "profiles" / "demo.yaml").resolve()
    assert resolve_profile_path("demo.yaml") == (fake_package / "profiles" / "demo.yaml").resolve()


def test_a_name_falls_back_to_the_shared_drive(fake_package, monkeypatch, tmp_path) -> None:
    shared = tmp_path / "shared"
    write_yaml(shared / "smpl18" / "profiles" / "only_shared.yaml", {"from": "shared"})
    monkeypatch.setenv("SHARED_DATASET_PATH", str(shared))
    assert (
        resolve_profile_path("only_shared")
        == (shared / "smpl18" / "profiles" / "only_shared.yaml").resolve()
    )


def test_not_found_lists_every_place_tried_and_the_unset_variable(fake_package, no_shared) -> None:
    with pytest.raises(ProfileNotFound) as info:
        resolve_profile_path("nowhere")
    message = str(info.value)
    assert "as a path" in message
    assert "package examples" in message
    assert "SHARED_DATASET_PATH is not set" in message


# ---- environment substitution ----------------------------------------------------------------


def test_env_substitution_inside_string_values(monkeypatch) -> None:
    monkeypatch.setenv("SHARED_DATASET_PATH", "/drive")
    data = {"a": "${SHARED_DATASET_PATH}/x", "b": ["${SHARED_DATASET_PATH}"], "c": {"d": 3}}
    assert substitute_env(data) == {"a": "/drive/x", "b": ["/drive"], "c": {"d": 3}}


def test_unset_variable_names_the_key(no_shared) -> None:
    with pytest.raises(EnvironmentVariableUnset) as info:
        substitute_env({"layout": {"subject": "${SHARED_DATASET_PATH}/{subject}"}})
    assert "layout.subject" in str(info.value)
    assert "SHARED_DATASET_PATH" in str(info.value)


def test_profile_needing_the_variable_fails_clearly_when_unset(
    profile_dir, fake_package, no_shared, parameters_profile
) -> None:
    parameters_profile["settings"] = "${SHARED_DATASET_PATH}/smpl18/settings/default.yaml"
    path = write_yaml(profile_dir / "needs_env.yaml", parameters_profile)
    with pytest.raises(EnvironmentVariableUnset) as info:
        Profile.load(path)
    assert "settings" in str(info.value) and "SHARED_DATASET_PATH" in str(info.value)


# ---- referenced files ------------------------------------------------------------------------


def test_relative_references_resolve_against_the_profile_directory_and_are_hashed(
    profile_dir, fake_package, no_shared
) -> None:
    profile = Profile.load(profile_dir / "skeleton.yaml")
    roles = {entry.role: entry for entry in profile.referenced_files()}
    assert set(roles) == {"correspondence", "settings[0]"}
    correspondence = roles["correspondence"]
    assert correspondence.given == "correspondence.yaml"
    assert correspondence.path == (profile_dir / "correspondence.yaml").resolve()
    expected = hashlib.sha256((profile_dir / "correspondence.yaml").read_bytes()).hexdigest()
    assert correspondence.sha256 == expected
    assert profile.sha256 == hashlib.sha256((profile_dir / "skeleton.yaml").read_bytes()).hexdigest()


def test_a_reference_falls_back_to_the_package_then_the_shared_drive(
    profile_dir, fake_package, monkeypatch, tmp_path, parameters_profile
) -> None:
    write_yaml(fake_package / "settings" / "engine.yaml", {**SETTINGS, "id": "package"})
    shared = tmp_path / "shared"
    write_yaml(shared / "smpl18" / "settings" / "engine.yaml", {**SETTINGS, "id": "shared"})
    write_yaml(shared / "smpl18" / "settings" / "only_shared.yaml", {**SETTINGS, "id": "only"})
    monkeypatch.setenv("SHARED_DATASET_PATH", str(shared))

    parameters_profile["settings"] = ["settings/engine.yaml", "settings/only_shared.yaml"]
    path = write_yaml(profile_dir / "layered.yaml", parameters_profile)
    profile = Profile.load(path)
    paths = [entry.path for entry in profile.referenced_files()]
    assert paths[0] == (fake_package / "settings" / "engine.yaml").resolve()
    assert paths[1] == (shared / "smpl18" / "settings" / "only_shared.yaml").resolve()
    assert profile.settings["id"] == "only"      # later files override earlier ones


def test_a_missing_reference_lists_every_place_tried(
    profile_dir, fake_package, no_shared, skeleton_profile
) -> None:
    skeleton_profile["correspondence"] = "missing.yaml"
    path = write_yaml(profile_dir / "broken.yaml", skeleton_profile)
    with pytest.raises(ReferenceNotFound) as info:
        Profile.load(path)
    message = str(info.value)
    assert "correspondence" in message and "missing.yaml" in message
    assert "relative to the profile" in message and "package configs" in message


def test_settings_files_deep_merge_in_order(profile_dir, fake_package, no_shared,
                                           parameters_profile) -> None:
    write_yaml(profile_dir / "override.yaml",
               {"schema": "smpl18_settings_v1", "shape_fit": {"regularisation": 0.5}})
    parameters_profile["settings"] = ["settings.yaml", "override.yaml"]
    path = write_yaml(profile_dir / "merged.yaml", parameters_profile)
    profile = Profile.load(path)
    assert profile.settings["shape_fit"] == {"n_betas": 10, "regularisation": 0.5}
    assert profile.settings["discontinuity"]["gyro_abs_max_deg_s"] == 8000.0


def test_a_referenced_yaml_without_a_schema_key_is_refused(
    profile_dir, fake_package, no_shared, skeleton_profile
) -> None:
    write_yaml(profile_dir / "bare.yaml", {"joints": {}})
    skeleton_profile["correspondence"] = "bare.yaml"
    path = write_yaml(profile_dir / "bare_profile.yaml", skeleton_profile)
    with pytest.raises(load_module.ProfileLoadError) as info:
        Profile.load(path)
    assert "'schema'" in str(info.value)


# ---- the loaded object -----------------------------------------------------------------------


def test_schema_errors_name_the_file(profile_dir, fake_package, no_shared,
                                     parameters_profile) -> None:
    parameters_profile["bindings"]["poses"]["layouts"] = "T,J,3"
    path = write_yaml(profile_dir / "typo.yaml", parameters_profile)
    with pytest.raises(ProfileSchemaError) as info:
        Profile.load(path)
    assert info.value.path == "bindings.poses.layouts"
    assert "typo.yaml" in info.value.message


def test_typed_accessors(profile_dir, fake_package, no_shared) -> None:
    profile = Profile.load(profile_dir / "params.yaml")
    assert profile.id == "params_fixture"
    assert profile.source_kind is SourceKind.SMPL_PARAMETERS
    assert profile.format is Format.NPZ
    assert profile.layout.trial.endswith("_poses.npz")
    assert profile.bindings["fps"]["fallback"]["A"] == 120.0
    assert profile.conventions["up_axis"] == "z"
    assert profile.section("repairs")["resample"]["fps"] == 100
    assert profile.section("provenance") == {}


def test_resolved_is_a_plain_dict_with_files_and_hashes(profile_dir, fake_package,
                                                         no_shared) -> None:
    profile = Profile.load(profile_dir / "skeleton.yaml")
    plain = profile.resolved()
    assert plain["profile"]["sha256"] == profile.sha256
    assert plain["id"] == "skeleton_fixture"
    roles = {entry["role"]: entry for entry in plain["referenced_files"]}
    assert set(roles["correspondence"]) == {"role", "given", "path", "sha256"}
    assert plain["settings_resolved"]["shape_fit"]["n_betas"] == 10
    assert copy.deepcopy(plain) == plain      # nothing exotic inside
