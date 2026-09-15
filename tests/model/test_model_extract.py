"""pkl -> clean npz: exactly the listed keys, the tree with root -1, the shape space cut to width."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from smpl24.formats.pickle_safe import BlockedGlobal
from smpl24.model import (
    MESH_KEYS,
    REQUIRED_KEYS,
    Extracted,
    UnresolvedGender,
    extract_clean,
    load,
)
from smpl24.model.extract import main
from smpl24.skeleton.definition import PARENTS


@pytest.fixture
def pkl(tmp_path, rng, make_fake_pickle):
    path = tmp_path / "basicmodel_f_lbs_10_207_0_v1.0.0.pkl"
    plain = make_fake_pickle(path, rng)
    return path, plain


def test_skeleton_only_extract_writes_exactly_the_required_keys(tmp_path, pkl):
    path, plain = pkl
    result = extract_clean(path, tmp_path / "models", gender="female", num_betas=10, with_mesh=False)

    assert isinstance(result, Extracted)
    assert result.path == tmp_path / "models" / "SMPL_FEMALE_clean.npz"
    assert set(result.shapes) == set(REQUIRED_KEYS)
    with np.load(result.path, allow_pickle=False) as data:
        assert set(data.files) == set(REQUIRED_KEYS)
        np.testing.assert_array_equal(data["v_template"], plain["v_template"])
        np.testing.assert_array_equal(data["shapedirs"], plain["shapedirs"][:, :, :10])
        np.testing.assert_array_equal(data["J_regressor"], plain["J_regressor"])
        assert data["v_template"].dtype == np.float64 and data["shapedirs"].shape == (30, 3, 10)
        assert data["kintree_parents"].dtype == np.int64
        assert tuple(data["kintree_parents"]) == PARENTS and data["kintree_parents"][0] == -1


def test_with_mesh_adds_the_skinning_trio_with_faces_taken_from_f(tmp_path, pkl):
    path, plain = pkl
    result = extract_clean(path, tmp_path, gender="male", num_betas=10, with_mesh=True)

    assert set(result.shapes) == set(REQUIRED_KEYS) | set(MESH_KEYS)
    with np.load(result.path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["weights"], plain["weights"])
        np.testing.assert_array_equal(data["posedirs"], plain["posedirs"])
        np.testing.assert_array_equal(data["faces"], plain["faces"])
        assert data["faces"].dtype == np.int64
        assert "f" not in data.files


def test_an_npz_path_is_written_as_given_and_num_betas_is_honoured(tmp_path, pkl):
    path, plain = pkl
    target = tmp_path / "somewhere" / "custom.npz"
    result = extract_clean(path, target, gender="neutral", num_betas=4, with_mesh=False)
    assert result.path == target
    assert result.shapes["shapedirs"] == (30, 3, 4)


def test_the_extract_loads_back_as_a_model_with_the_gender_of_its_name(tmp_path, pkl):
    path, plain = pkl
    result = extract_clean(path, tmp_path, gender="female", num_betas=10, with_mesh=True)
    model = load(result.path)
    assert model.gender == "female" and model.num_betas == 10 and model.has_mesh
    assert model.path == result.path


def test_a_narrower_shape_space_than_asked_is_refused(tmp_path, pkl):
    path, plain = pkl
    with pytest.raises(ValueError, match="shape directions"):
        extract_clean(path, tmp_path, gender="male", num_betas=301, with_mesh=False)
    assert not (tmp_path / "SMPL_MALE_clean.npz").exists()


def test_an_unknown_gender_is_refused_before_the_pickle_is_read(tmp_path):
    with pytest.raises(UnresolvedGender):
        extract_clean(tmp_path / "missing.pkl", tmp_path, gender="M", num_betas=10, with_mesh=False)


def test_a_pickle_with_the_wrong_joint_count_is_refused_and_nothing_is_written(tmp_path, rng, make_arrays):
    arrays = make_arrays(rng, with_mesh=False)
    arrays["J_regressor"] = arrays["J_regressor"][:23]
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as handle:
        pickle.dump(arrays, handle, protocol=2)
    with pytest.raises(ValueError, match="J_regressor"):
        extract_clean(path, tmp_path, gender="male", num_betas=3, with_mesh=False)
    assert not (tmp_path / "SMPL_MALE_clean.npz").exists()


def test_a_non_dict_pickle_is_refused(tmp_path):
    path = tmp_path / "list.pkl"
    with open(path, "wb") as handle:
        pickle.dump([1, 2, 3], handle, protocol=2)
    with pytest.raises(TypeError, match="expected dict"):
        extract_clean(path, tmp_path, gender="male", num_betas=10, with_mesh=False)


def test_a_pickle_that_names_foreign_code_is_refused(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(b"cos\nsystem\n(S'echo pwned'\ntR.")
    with pytest.raises(BlockedGlobal, match="os.system"):
        extract_clean(path, tmp_path, gender="male", num_betas=10, with_mesh=False)


def test_main_runs_the_extract_model_command(tmp_path, pkl, capsys):
    path, plain = pkl
    out = tmp_path / "set"
    code = main(["--pkl", str(path), "--gender", "male", "--out", str(out), "--num-betas", "4", "--with-mesh"])
    assert code == 0
    printed = capsys.readouterr().out
    assert f"wrote {out / 'SMPL_MALE_clean.npz'}" in printed
    assert "posedirs" in printed and "(30, 3, 4)" in printed
    assert load(out / "SMPL_MALE_clean.npz").num_betas == 4


def test_main_refuses_a_gender_outside_the_set(tmp_path, pkl):
    path, plain = pkl
    with pytest.raises(SystemExit):
        main(["--pkl", str(path), "--gender", "m", "--out", str(tmp_path)])
