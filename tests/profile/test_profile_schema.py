import pytest

from smpl18.profile.schema import ProfileSchemaError, validate_profile


def test_fixture_profiles_validate(parameters_profile, skeleton_profile) -> None:
    validate_profile(parameters_profile)
    validate_profile(skeleton_profile)


def test_unknown_top_level_key_is_reported_with_its_path(parameters_profile) -> None:
    parameters_profile["shapes"] = {"method": "parameters"}
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "shapes"
    assert "unknown key" in info.value.message


def test_unknown_nested_key_is_reported_with_its_dotted_path(parameters_profile) -> None:
    parameters_profile["bindings"]["gender"]["mapp"] = {}
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "bindings.gender.mapp"


def test_unknown_key_inside_a_list_item_names_the_index(skeleton_profile) -> None:
    skeleton_profile["root"]["alignment"] = "directions"
    skeleton_profile["root"]["directions"] = [
        {"smpl": ["left_hip", "right_hip"], "source": ["LHJC", "RHJC"]},
        {"smpl": ["neck", "pelvis"], "source": ["NECK", "PELVIS"], "extra": 1},
    ]
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "root.directions[1].extra"


def test_bad_enum_value_lists_the_allowed_values(skeleton_profile) -> None:
    skeleton_profile["root"]["placement"] = "hips"
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "root.placement"
    assert "pelvis_centre" in info.value.message and "'hips'" in info.value.message


def test_bad_gender_map_target_is_refused(parameters_profile) -> None:
    parameters_profile["bindings"]["gender"]["map"]["x"] = "unknown"
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "bindings.gender.map.x"


def test_wrong_schema_id_is_refused(parameters_profile) -> None:
    parameters_profile["schema"] = "smpl18_profile_v2"
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "schema"


def test_missing_required_key_is_reported_at_the_parent(parameters_profile) -> None:
    del parameters_profile["conventions"]["up_axis"]
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "conventions"
    assert "up_axis" in info.value.message


def test_parameters_kind_needs_its_bindings(parameters_profile) -> None:
    del parameters_profile["bindings"]["trans"]
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "bindings.trans"


def test_parameters_kind_refuses_a_correspondence(parameters_profile) -> None:
    parameters_profile["correspondence"] = "x.yaml"
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "correspondence"


def test_skeleton_kind_needs_root_and_pose(skeleton_profile) -> None:
    del skeleton_profile["pose"]
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "pose"


def test_gender_binding_is_field_or_constant_not_both(skeleton_profile) -> None:
    skeleton_profile["bindings"]["gender"]["constant"] = "neutral"
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "bindings.gender"


def test_fps_fallback_needs_its_key(parameters_profile) -> None:
    del parameters_profile["bindings"]["fps"]["fallback_key"]
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(parameters_profile)
    assert info.value.path == "bindings.fps"


def test_directions_alignment_needs_two_pairs(skeleton_profile) -> None:
    skeleton_profile["root"]["alignment"] = "directions"
    skeleton_profile["root"]["directions"] = [
        {"smpl": ["left_hip", "right_hip"], "source": ["LHJC", "RHJC"]}
    ]
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "root.directions"


def test_reference_flags_need_a_reference_trial(skeleton_profile) -> None:
    skeleton_profile["pose"]["lumbar_zero_from_reference"] = True
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "pose.lumbar_zero_from_reference"


def test_companion_files_conflict_with_within_container(skeleton_profile) -> None:
    skeleton_profile["layout"]["files"] = {"osim": "{subject}.osim"}
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "layout.files"


def test_one_of_failure_lists_every_alternative(skeleton_profile) -> None:
    skeleton_profile["repairs"]["wrap"] = "sometimes"
    with pytest.raises(ProfileSchemaError) as info:
        validate_profile(skeleton_profile)
    assert info.value.path == "repairs.wrap"
    assert "'none'" in info.value.message and "mapping" in info.value.message


def test_non_mapping_profile_is_refused() -> None:
    with pytest.raises(ProfileSchemaError):
        validate_profile(["not", "a", "profile"])
