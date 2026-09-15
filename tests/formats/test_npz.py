"""Reading an ``.npz``: every array by key, and never a pickled object."""

import numpy as np
import pytest

from smpl24.formats import npz


def test_every_array_comes_back_by_its_key(tmp_path):
    path = tmp_path / "trial.npz"
    poses = np.arange(24.0).reshape(2, 12)
    np.savez(path, poses=poses, fps=np.array(100.0), gender=np.array("female"))
    tables = npz.read(path)
    assert set(tables) == {"poses", "fps", "gender"}
    np.testing.assert_array_equal(tables["poses"], poses)
    assert tables["fps"] == 100.0
    assert str(tables["gender"]) == "female"


def test_arrays_are_usable_after_the_archive_is_closed(tmp_path):
    path = tmp_path / "trial.npz"
    np.savez(path, betas=np.zeros(10))
    betas = npz.read(path)["betas"]
    assert betas.shape == (10,)
    assert betas.sum() == 0.0


def test_a_pickled_object_array_is_refused(tmp_path):
    path = tmp_path / "objects.npz"
    np.savez(path, meta=np.array({"a": 1}, dtype=object))
    with pytest.raises(ValueError, match="allow_pickle"):
        npz.read(path)
