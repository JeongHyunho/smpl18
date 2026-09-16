"""The extracted SMPL body model in memory, and loading it from a clean npz.

A clean npz holds ``v_template``, ``shapedirs``, ``J_regressor`` and ``kintree_parents``, and
optionally the mesh trio ``weights``, ``posedirs``, ``faces`` (``docs/primer.md`` section 2.4).
Nothing is unpickled: ``np.load`` runs with ``allow_pickle=False``. Arrays are float64 (indices
int64) and read-only, so a model shared between callers cannot be edited under them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smpl18.skeleton.definition import NUM_JOINTS, PARENTS

from .select import MODEL_FILENAMES, file_sha256, model_path_for_gender

__all__ = ["MESH_KEYS", "REQUIRED_KEYS", "STAND_IN_KEY", "Model", "load"]

REQUIRED_KEYS: tuple[str, ...] = ("v_template", "shapedirs", "J_regressor", "kintree_parents")
MESH_KEYS: tuple[str, ...] = ("weights", "posedirs", "faces")
#: Present in a file written by ``smpl18.model.demo``: a stand-in, not an SMPL model.
STAND_IN_KEY = "stand_in"

_POSE_FEATURES = (NUM_JOINTS - 1) * 9


def _read_only(values, dtype) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class Model:
    """An SMPL body model reduced to what the skeleton (and optionally the mesh) needs.

    ``gender``, ``path`` and ``sha256`` are ``None`` for a model built in memory rather than
    loaded from a file. ``stand_in`` is true for the demonstration body of
    :mod:`smpl18.model.demo`, which only borrows the SMPL-24 tree.
    """

    v_template: np.ndarray
    shapedirs: np.ndarray
    J_regressor: np.ndarray
    kintree_parents: np.ndarray
    weights: np.ndarray | None = None
    posedirs: np.ndarray | None = None
    faces: np.ndarray | None = None
    gender: str | None = None
    path: Path | None = None
    sha256: str | None = None
    stand_in: bool = False

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "v_template", _read_only(self.v_template, np.float64))
        set_(self, "shapedirs", _read_only(self.shapedirs, np.float64))
        set_(self, "J_regressor", _read_only(self.J_regressor, np.float64))
        set_(self, "kintree_parents", _read_only(self.kintree_parents, np.int64))
        for key, dtype in (("weights", np.float64), ("posedirs", np.float64), ("faces", np.int64)):
            if getattr(self, key) is not None:
                set_(self, key, _read_only(getattr(self, key), dtype))
        self._validate()

    def _validate(self) -> None:
        vertices = self.v_template.shape[0]
        if self.shapedirs.ndim != 3 or self.shapedirs.shape[:2] != (vertices, 3):
            raise ValueError(f"shapedirs has shape {self.shapedirs.shape}, expected ({vertices}, 3, B)")
        expected = {
            "v_template": (vertices, 3),
            "J_regressor": (NUM_JOINTS, vertices),
            "kintree_parents": (NUM_JOINTS,),
            "weights": (vertices, NUM_JOINTS),
            "posedirs": (vertices, 3, _POSE_FEATURES),
        }
        for key, shape in expected.items():
            array = getattr(self, key)
            if array is not None and array.shape != shape:
                raise ValueError(f"{key} has shape {array.shape}, expected {shape}")
        if self.faces is not None and (self.faces.ndim != 2 or self.faces.shape[1] != 3):
            raise ValueError(f"faces has shape {self.faces.shape}, expected (F, 3)")
        if tuple(int(p) for p in self.kintree_parents) != PARENTS:
            raise ValueError("kintree_parents is not the SMPL-24 tree")

    @property
    def num_betas(self) -> int:
        return int(self.shapedirs.shape[2])

    @property
    def num_vertices(self) -> int:
        return int(self.v_template.shape[0])

    @property
    def has_mesh(self) -> bool:
        return self.weights is not None and self.posedirs is not None and self.faces is not None

    @classmethod
    def for_gender(cls, gender: str, root: str | Path | None = None) -> Model:
        """Load the model :func:`smpl18.model.select.model_path_for_gender` names."""
        return load(model_path_for_gender(gender, root), gender=gender)


def _gender_from_name(path: Path) -> str | None:
    for gender, filename in MODEL_FILENAMES.items():
        if path.name == filename:
            return gender
    return None


def load(path: str | Path, *, gender: str | None = None) -> Model:
    """Read a clean npz into a :class:`Model`, hashing the file and checking keys and shapes.

    ``gender`` defaults to what the file name says when it is one of the set's names.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [key for key in REQUIRED_KEYS if key not in keys]
        if missing:
            raise ValueError(f"{path} lacks required model keys {missing}")
        arrays = {key: data[key] for key in REQUIRED_KEYS}
        for key in MESH_KEYS:
            if key in keys:
                arrays[key] = data[key]
        stand_in = STAND_IN_KEY in keys
        if "faces" not in arrays and "f" in keys:
            arrays["faces"] = data["f"]     # the SMPL pickle's own spelling of the triangle list
    return Model(
        **arrays,
        gender=gender if gender is not None else _gender_from_name(path),
        path=path,
        sha256=file_sha256(path),
        stand_in=stand_in,
    )
