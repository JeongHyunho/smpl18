"""Synthetic body models for the model tests: small random arrays with the SMPL-24 tree.

``fake_smpl_pickle`` writes a pickle laid out like a licensed ``basicmodel_*.pkl``: arrays
wrapped in a ``chumpy.ch.Ch`` class (a throwaway stub registered in ``sys.modules`` only while
pickling), a ``scipy.sparse`` joint regressor, a 300-wide shape space, the triangle list under
``f`` and the tree under ``kintree_table`` with an unsigned -1 for the root's parent.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

from smpl24.skeleton.definition import PARENTS

VERTICES = 30
POSE_FEATURES = 23 * 9


def tiny_arrays(rng: np.random.Generator, *, num_betas: int = 3, with_mesh: bool = True) -> dict:
    regressor = rng.random((24, VERTICES))
    arrays = {
        "v_template": rng.standard_normal((VERTICES, 3)),
        "shapedirs": rng.standard_normal((VERTICES, 3, num_betas)) * 0.05,
        "J_regressor": regressor / regressor.sum(axis=1, keepdims=True),
        "kintree_parents": np.array(PARENTS, dtype=np.int64),
    }
    if with_mesh:
        weights = rng.random((VERTICES, 24)) + 0.05
        arrays["weights"] = weights / weights.sum(axis=1, keepdims=True)
        arrays["posedirs"] = rng.standard_normal((VERTICES, 3, POSE_FEATURES)) * 0.02
        arrays["faces"] = rng.integers(0, VERTICES, size=(12, 3)).astype(np.int64)
    return arrays


def fake_smpl_pickle(path: Path, rng: np.random.Generator, *, protocol: int = 2, shape_width: int = 300) -> dict:
    """Write a licensed-model look-alike pickle at ``path``; return the plain arrays it wraps."""
    plain = tiny_arrays(rng, num_betas=shape_width, with_mesh=True)
    kintree = np.zeros((2, 24), dtype=np.uint32)
    kintree[0] = np.array(PARENTS, dtype=np.int64).astype(np.uint32)   # -1 wraps to 4294967295
    kintree[1] = np.arange(24, dtype=np.uint32)

    chumpy = types.ModuleType("chumpy")
    chumpy_ch = types.ModuleType("chumpy.ch")

    class Ch:
        def __init__(self, x):
            self.x = x
            self.dterms = ("x",)

    Ch.__module__ = "chumpy.ch"
    Ch.__qualname__ = "Ch"
    chumpy_ch.Ch = Ch
    chumpy.ch = chumpy_ch
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = chumpy_ch
    try:
        payload = {
            "v_template": Ch(plain["v_template"]),
            "shapedirs": Ch(plain["shapedirs"]),
            "posedirs": Ch(plain["posedirs"]),
            "J_regressor": scipy.sparse.csc_matrix(plain["J_regressor"]),
            "weights": plain["weights"],
            "f": plain["faces"].astype(np.uint32),
            "kintree_table": kintree,
            "J": plain["J_regressor"] @ plain["v_template"],
            "bs_style": "lbs",
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=protocol)
    finally:
        del sys.modules["chumpy"]
        del sys.modules["chumpy.ch"]
    return plain


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def tiny(rng) -> dict:
    return tiny_arrays(rng)


@pytest.fixture
def make_arrays():
    return tiny_arrays


@pytest.fixture
def make_fake_pickle():
    return fake_smpl_pickle


@pytest.fixture
def model_dir(tmp_path, rng) -> Path:
    """A model set with male and female files present and the neutral one absent."""
    for seed, name in ((1, "SMPL_MALE_clean.npz"), (2, "SMPL_FEMALE_clean.npz")):
        np.savez(tmp_path / name, **tiny_arrays(np.random.default_rng(seed)))
    return tmp_path
