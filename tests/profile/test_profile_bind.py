import numpy as np
import pytest

from smpl18.profile.bind import (
    BindingError,
    MissingField,
    UnresolvedGender,
    bind_parameters,
    read_field,
    resolve_fps,
    resolve_gender,
)
from smpl18.sources import SmplParameters, Trial, UpAxis


def tables(frames: int = 5, joints: int = 52, **extra) -> dict:
    rng = np.random.default_rng(0)
    data = {
        "poses": rng.normal(size=(frames, joints * 3)),
        "betas": rng.normal(size=16),
        "trans": rng.normal(size=(frames, 3)),
        "gender": np.array("male"),
        "mocap_framerate": np.array(120.0),
    }
    data.update(extra)
    return data


def trial(**groups) -> Trial:
    return Trial(id="t1", subject="s1", groups={"study": "A", **groups})


def test_bind_parameters_reshapes_flat_poses_and_maps_gender(parameters_profile) -> None:
    bound = bind_parameters(parameters_profile["bindings"], tables(), up_axis="z",
                            trial=trial())
    assert isinstance(bound, SmplParameters)
    assert bound.poses.shape == (5, 52, 3)
    assert bound.trans.shape == (5, 3)
    assert bound.betas.shape == (16,)
    assert bound.fps == 120.0 and bound.fps_provenance == "field"
    assert bound.subject.gender == "male" and bound.subject.gender_provenance == "field"
    assert bound.up_axis is UpAxis.Z
    assert bound.frame_valid.all() and bound.frames == 5 and bound.joints == 52


def test_fps_alias_then_fallback_by_group(parameters_profile) -> None:
    binding = parameters_profile["bindings"]["fps"]
    assert resolve_fps(binding, {"mocap_frame_rate": 60.0}) == (60.0, "field")
    assert resolve_fps(binding, {}, {"study": "B"}) == (60.0, "fallback")
    with pytest.raises(BindingError, match="no fallback for study 'Z'"):
        resolve_fps(binding, {}, {"study": "Z"})
    with pytest.raises(BindingError, match="did not match"):
        resolve_fps(binding, {}, {})


def test_fps_without_fallback_is_a_missing_field() -> None:
    with pytest.raises(MissingField, match="fps"):
        resolve_fps({"field": "rate"}, {})
    assert resolve_fps({"constant": 100}, {}) == (100.0, "constant")


def test_gender_map_default_when_absent_and_errors() -> None:
    binding = {"field": "gender", "map": {"female": "female", "f": "female", "male": "male"}}
    assert resolve_gender(binding, {"gender": " F "}) == ("female", "field")
    assert resolve_gender(binding, {"gender": b"male"}) == ("male", "field")
    assert resolve_gender(binding, {"gender": np.array(["Female"])}) == ("female", "field")
    with pytest.raises(UnresolvedGender, match="'unknown' is not in the profile's map"):
        resolve_gender(binding, {"gender": "unknown"})
    assert resolve_gender({**binding, "default": "neutral"}, {"gender": "unknown"}) == (
        "neutral", "default")
    with pytest.raises(UnresolvedGender, match="absent"):
        resolve_gender(binding, {})
    assert resolve_gender({**binding, "when_absent": "neutral"}, {}) == ("neutral", "when_absent")
    assert resolve_gender({"constant": "neutral"}, {}) == ("neutral", "constant")


def test_gender_cross_check_on_the_first_initial() -> None:
    binding = {"field": ["info", "subj_info", "gender"], "map": {"m": "male", "f": "female"},
               "cross_check": ["smpl_params", "gender"]}
    nested = {"info": {"subj_info": {"gender": "M"}}, "smpl_params": {"gender": "male"}}
    assert resolve_gender(binding, nested) == ("male", "field")
    nested["smpl_params"]["gender"] = "female"
    with pytest.raises(UnresolvedGender, match="disagrees"):
        resolve_gender(binding, nested)


def test_nested_field_paths_and_missing_field_message() -> None:
    nested = {"smpl_params": {"poses": np.zeros((3, 72))}}
    assert read_field(nested, ["smpl_params", "poses"]).shape == (3, 72)
    with pytest.raises(MissingField) as info:
        read_field(nested, ["smpl_params", "trans"], what="trans")
    assert str(info.value) == "trans: no 'smpl_params/trans' in the source tables"


def test_betas_first_frame_and_nested_container(parameters_profile) -> None:
    bindings = {
        "poses": {"field": ["smpl_params", "poses"], "layout": "T,J*3"},
        "betas": {"field": ["smpl_params", "betas"], "frame": "first"},
        "trans": {"field": ["smpl_params", "trans"]},
        "fps": {"field": ["info", "data_info", "fps"]},
        "gender": {"field": ["info", "subj_info", "gender"], "map": {"m": "male", "f": "female"}},
    }
    per_frame_betas = np.tile(np.arange(10.0), (4, 1))
    per_frame_betas[1:] += 100.0
    nested = {
        "smpl_params": {"poses": np.zeros((4, 72)), "betas": per_frame_betas,
                        "trans": np.zeros((4, 3))},
        "info": {"data_info": {"fps": 100}, "subj_info": {"gender": "F"}},
    }
    bound = bind_parameters(bindings, nested, up_axis="z", trial=trial())
    assert bound.poses.shape == (4, 24, 3)
    np.testing.assert_array_equal(bound.betas, np.arange(10.0))
    assert bound.fps == 100.0
    assert bound.subject.gender == "female"


def test_non_finite_frames_are_marked_invalid_not_repaired(parameters_profile) -> None:
    data = tables()
    data["poses"][2, 0] = np.nan
    data["trans"][4, 1] = np.inf
    bound = bind_parameters(parameters_profile["bindings"], data, up_axis="z", trial=trial())
    assert bound.frame_valid.tolist() == [True, True, False, True, False]
    assert np.isnan(bound.poses[2, 0, 0])


def test_subject_fields_can_come_from_another_table(parameters_profile) -> None:
    bindings = {**parameters_profile["bindings"], "stature_m": {"field": "height"},
                "mass_kg": {"field": "weight"}}
    data = tables()
    del data["gender"]
    bound = bind_parameters(bindings, data, up_axis="z", trial=trial(),
                            subject_tables={"gender": "female", "height": 1.7, "weight": -1})
    assert bound.subject.gender == "female"
    assert bound.subject.stature_m == 1.7
    assert bound.subject.mass_kg is None       # a placeholder is not a measurement


def test_bad_pose_layout_is_a_binding_error(parameters_profile) -> None:
    data = tables()
    data["poses"] = data["poses"][:, :70]
    with pytest.raises(BindingError, match="T,J\\*3"):
        bind_parameters(parameters_profile["bindings"], data, up_axis="z", trial=trial())
