"""Which extracted body-model file serves a gender, where the set lives, and what it hashes to.

The set is three files, ``SMPL_MALE_clean.npz``, ``SMPL_FEMALE_clean.npz`` and
``SMPL_NEUTRAL_clean.npz``, in one directory. That directory comes from the ``root`` argument
or the ``SMPL24_MODELS`` environment variable and from nowhere else: there is no default path,
so a run on another machine fails at once instead of quietly reading a different model.

Gender is one of ``male``, ``female``, ``neutral``. Mapping a source's own labels (``f``,
``M``, ``unknown``) onto those is a profile's job, not this module's.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = [
    "ENV_MODELS",
    "GENDERS",
    "GENDER_FEMALE",
    "GENDER_MALE",
    "GENDER_NEUTRAL",
    "MODEL_FILENAMES",
    "ModelRootUnset",
    "UnresolvedGender",
    "file_sha256",
    "model_filename",
    "model_path_for_gender",
    "model_root",
    "set_hashes",
]

GENDER_MALE = "male"
GENDER_FEMALE = "female"
GENDER_NEUTRAL = "neutral"
GENDERS: tuple[str, ...] = (GENDER_MALE, GENDER_FEMALE, GENDER_NEUTRAL)

#: The environment variable naming the directory that holds the model set.
ENV_MODELS = "SMPL24_MODELS"

MODEL_FILENAMES: dict[str, str] = {gender: f"SMPL_{gender.upper()}_clean.npz" for gender in GENDERS}


class UnresolvedGender(ValueError):
    """The gender given selects no body model."""


class ModelRootUnset(LookupError):
    """Neither a ``root`` argument nor ``SMPL24_MODELS`` says where the model set lives."""


def model_filename(gender: str) -> str:
    """``SMPL_<GENDER>_clean.npz`` for one of :data:`GENDERS`."""
    try:
        return MODEL_FILENAMES[gender]
    except KeyError:
        raise UnresolvedGender(
            f"no body model is defined for gender {gender!r}; expected one of {GENDERS}"
        ) from None


def model_root(root: str | Path | None = None) -> Path:
    """The model-set directory: ``root`` if given, else ``$SMPL24_MODELS``, else an error."""
    if root is not None:
        return Path(root)
    from_env = os.environ.get(ENV_MODELS)
    if from_env:
        return Path(from_env)
    raise ModelRootUnset(
        f"no body-model directory: pass --models <dir> (the root argument) or set {ENV_MODELS}"
    )


def model_path_for_gender(gender: str, root: str | Path | None = None) -> Path:
    """Path of the extracted model for ``gender`` under :func:`model_root`; existence is not checked."""
    return model_root(root) / model_filename(gender)


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_hashes(root: str | Path | None = None) -> dict[str, str | None]:
    """``{filename: sha256}`` for the whole set under :func:`model_root`; ``None`` where a file is absent."""
    base = model_root(root)
    return {
        filename: (file_sha256(base / filename) if (base / filename).is_file() else None)
        for filename in MODEL_FILENAMES.values()
    }
