"""Loading a clean npz: keys and shapes are checked, arrays are float64 and read-only, the file is hashed."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from smpl18.model import Model, load


def test_load_gives_typed_read_only_arrays_and_the_file_hash(tmp_path, tiny):
    path = tmp_path / "SMPL_MALE_clean.npz"
    np.savez(path, **{k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in tiny.items()})

    model = load(path)

    assert model.gender == "male" and model.path == path
    assert model.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert model.num_betas == 3 and model.num_vertices == 30 and model.has_mesh
    for name in ("v_template", "shapedirs", "J_regressor", "weights", "posedirs"):
        array = getattr(model, name)
        assert array.dtype == np.float64 and not array.flags.writeable
    assert model.kintree_parents.dtype == np.int64 and model.faces.dtype == np.int64
    np.testing.assert_array_equal(model.v_template, tiny["v_template"].astype(np.float32))
    with pytest.raises(ValueError):
        model.v_template[0, 0] = 1.0


def test_an_explicit_gender_overrides_the_file_name_and_an_unknown_name_gives_none(tmp_path, tiny):
    path = tmp_path / "SMPL_MALE_clean.npz"
    np.savez(path, **tiny)
    assert load(path, gender="neutral").gender == "neutral"
    other = tmp_path / "body.npz"
    np.savez(other, **tiny)
    assert load(other).gender is None


def test_skeleton_only_files_load_without_a_mesh(tmp_path, rng, make_arrays):
    path = tmp_path / "SMPL_FEMALE_clean.npz"
    np.savez(path, **make_arrays(rng, with_mesh=False))
    model = load(path)
    assert not model.has_mesh and model.weights is None and model.faces is None


def test_the_legacy_f_key_is_accepted_as_faces(tmp_path, tiny):
    faces = tiny.pop("faces")
    path = tmp_path / "SMPL_FEMALE_clean.npz"
    np.savez(path, f=faces, bs_style="lbs", **tiny)
    model = load(path)
    np.testing.assert_array_equal(model.faces, faces)
    assert model.has_mesh


def test_a_missing_required_key_is_named(tmp_path, tiny):
    del tiny["J_regressor"]
    path = tmp_path / "broken.npz"
    np.savez(path, **tiny)
    with pytest.raises(ValueError, match="J_regressor"):
        load(path)


def test_a_file_whose_tree_is_not_smpl18_is_refused(tmp_path, tiny):
    tiny["kintree_parents"] = np.zeros(24, dtype=np.int64)
    path = tmp_path / "tree.npz"
    np.savez(path, **tiny)
    with pytest.raises(ValueError, match="SMPL-24 tree"):
        load(path)


def test_pickled_object_arrays_never_load(tmp_path, tiny):
    tiny["v_template"] = np.array([None] * 30, dtype=object)
    path = tmp_path / "objects.npz"
    np.savez(path, **tiny)
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize(
    "key,shape",
    [
        ("v_template", (30, 2)),
        ("shapedirs", (30, 2, 3)),
        ("J_regressor", (24, 31)),
        ("kintree_parents", (23,)),
        ("weights", (30, 23)),
        ("posedirs", (30, 3, 200)),
        ("faces", (5, 4)),
    ],
)
def test_in_memory_models_check_every_shape(tiny, key, shape):
    tiny[key] = np.zeros(shape, dtype=tiny[key].dtype)
    with pytest.raises(ValueError, match=key):
        Model(**tiny)


def test_in_memory_models_copy_their_inputs(tiny):
    source = tiny["v_template"]
    model = Model(**tiny)
    source[0, 0] = 99.0
    assert model.v_template[0, 0] != 99.0
    assert model.gender is None and model.path is None and model.sha256 is None
