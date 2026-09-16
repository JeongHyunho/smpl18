"""Fail-closed extraction of a licensed SMPL ``.pkl`` into a chumpy-free clean npz.

The pickle is read through :mod:`smpl18.formats.pickle_safe`, so no code from it runs and the
licensed original is never modified. The output holds exactly what the skeleton needs
(``v_template``, ``shapedirs`` cut to ``num_betas``, ``J_regressor``, ``kintree_parents`` with
the root's parent at -1) and, with ``with_mesh``, the skinning trio ``weights``, ``posedirs``,
``faces``. Every array is validated as a :class:`~smpl18.model.load.Model` before it is written,
so nothing the loader would refuse ever reaches disk.

``main`` is the body of the ``extract-model`` command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smpl18.formats import pickle_safe
from .load import MESH_KEYS, REQUIRED_KEYS, Model
from .select import GENDERS, model_filename

__all__ = ["Extracted", "extract_clean", "main"]


@dataclass(frozen=True)
class Extracted:
    """Where the clean model was written and the shape of every array in it."""

    path: Path
    shapes: dict[str, tuple[int, ...]]


def _parents_from(raw: dict) -> np.ndarray:
    if "kintree_parents" in raw:
        parents = pickle_safe.to_array(raw["kintree_parents"]).astype(np.int64)
    elif "kintree_table" in raw:
        parents = pickle_safe.to_array(raw["kintree_table"])[0].astype(np.int64).copy()
    else:
        raise ValueError("model has neither kintree_table nor kintree_parents")
    parents[0] = -1                      # the pickle stores the root's parent as unsigned -1
    return parents


def _output_path(out: str | Path, gender: str) -> Path:
    out = Path(out).expanduser()
    if out.suffix == ".npz":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    return out / model_filename(gender)


def extract_clean(
    pkl_path: str | Path,
    out: str | Path,
    *,
    gender: str,
    num_betas: int,
    with_mesh: bool,
) -> Extracted:
    """Write ``SMPL_<GENDER>_clean.npz`` under directory ``out`` (or to ``out`` itself when it ends in ``.npz``).

    ``shapedirs`` is cut to its first ``num_betas`` directions; a model with fewer is refused
    rather than silently written narrower than asked.
    """
    model_filename(gender)               # refuse an unknown gender before reading anything
    if num_betas < 1:
        raise ValueError(f"num_betas must be positive, got {num_betas}")
    raw = pickle_safe.load(pkl_path)
    if not isinstance(raw, dict):
        raise TypeError(f"{pkl_path}: top-level object is {type(raw).__name__}, expected dict")

    shapedirs = pickle_safe.to_array(raw["shapedirs"]).astype(np.float64)
    if shapedirs.ndim != 3 or shapedirs.shape[2] < num_betas:
        raise ValueError(
            f"shapedirs has shape {shapedirs.shape}; cannot take {num_betas} shape directions"
        )
    arrays: dict[str, np.ndarray] = {
        "v_template": pickle_safe.to_array(raw["v_template"]).astype(np.float64),
        "shapedirs": np.ascontiguousarray(shapedirs[:, :, :num_betas]),
        "J_regressor": pickle_safe.to_array(raw["J_regressor"]).astype(np.float64),
        "kintree_parents": _parents_from(raw),
    }
    if with_mesh:
        faces_key = "faces" if "faces" in raw else "f"
        arrays["weights"] = pickle_safe.to_array(raw["weights"]).astype(np.float64)
        arrays["posedirs"] = pickle_safe.to_array(raw["posedirs"]).astype(np.float64)
        arrays["faces"] = pickle_safe.to_array(raw[faces_key]).astype(np.int64)

    Model(**arrays)                      # the loader's shape checks, before anything is written
    path = _output_path(out, gender)
    np.savez(path, **arrays)
    return Extracted(path=path, shapes={key: value.shape for key, value in arrays.items()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smpl18 extract-model",
        description="Extract a licensed SMPL .pkl into a chumpy-free clean npz without running code from it.",
    )
    parser.add_argument("--pkl", required=True, help="the basicmodel_*.pkl to read")
    parser.add_argument("--gender", required=True, choices=GENDERS)
    parser.add_argument("--out", required=True, help="directory for SMPL_<GENDER>_clean.npz, or a .npz path")
    parser.add_argument("--num-betas", type=int, default=10, help="shape directions to keep (default 10)")
    parser.add_argument(
        "--with-mesh", action="store_true",
        help=f"also write {', '.join(MESH_KEYS)} (needed for skinning, volume and marker fitting)",
    )
    return parser


def main(argv: list[str]) -> int:
    """The ``extract-model`` command: parse ``argv``, extract, print what was written."""
    args = build_parser().parse_args(argv)
    result = extract_clean(
        args.pkl, args.out, gender=args.gender, num_betas=args.num_betas, with_mesh=args.with_mesh
    )
    print(f"wrote {result.path}")
    for key in (*REQUIRED_KEYS, *MESH_KEYS):
        if key in result.shapes:
            print(f"    {key:16s} {result.shapes[key]}")
    return 0
