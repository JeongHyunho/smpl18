"""A whitelisting unpickler: read numpy, scipy.sparse and chumpy-wrapped payloads, run nothing.

The published SMPL body-model pickles are Python 2, protocol 0, and mix numpy arrays, a
``scipy.sparse`` joint regressor and ``chumpy.Ch`` leaves. A plain ``pickle.load`` would import
whatever module the stream names and call it, and ``chumpy`` no longer imports on a current
numpy anyway. Here ``find_class`` resolves only the globals listed in :data:`BUILTIN_ALLOW`
(array and dtype reconstruction, sparse matrices, ``copyreg``, a few builtin containers,
``collections.OrderedDict``), maps every ``chumpy.*`` class onto the passive :class:`Ch` stub,
and refuses everything else by name before any import happens. A caller who really needs
another global lists it in ``allow``.
"""

from __future__ import annotations

import builtins
import codecs
import collections
import copyreg
import importlib
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse

__all__ = [
    "BUILTIN_ALLOW",
    "BlockedGlobal",
    "Ch",
    "SafeUnpickler",
    "load",
    "to_array",
]


class BlockedGlobal(pickle.UnpicklingError):
    """The stream named a global outside the whitelist. Nothing was imported or executed."""


class Ch:
    """Passive stand-in for any ``chumpy`` class: keeps the pickled state and runs no code.

    ``to_array`` returns the numpy payload the leaf carried, looking first at the attribute
    names chumpy uses for it (``x``, ``_x``, ``r``, ``_result``) and otherwise at the largest
    array in the captured state.
    """

    _PAYLOAD_KEYS = ("x", "_x", "r", "_result")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2:
            # copyreg's (instance dict, slot state) pair
            for part in state:
                if isinstance(part, dict):
                    self.__dict__.update(part)
        elif isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    def to_array(self) -> np.ndarray:
        for key in self._PAYLOAD_KEYS:
            value = self.__dict__.get(key)
            if isinstance(value, np.ndarray):
                return value
        arrays = [v for v in self.__dict__.values() if isinstance(v, np.ndarray)]
        if arrays:
            return max(arrays, key=lambda a: a.size)
        raise TypeError(f"chumpy leaf carries no array payload; attributes: {sorted(self.__dict__)}")


def _numpy_module(new: str, old: str):
    """numpy >= 2 keeps the reconstruction helpers under ``numpy._core``; 1.x under ``numpy.core``."""
    try:
        return importlib.import_module(new)
    except ModuleNotFoundError:
        return importlib.import_module(old)


def _builtin_allow() -> dict[tuple[str, str], Any]:
    multiarray = _numpy_module("numpy._core.multiarray", "numpy.core.multiarray")
    numeric = _numpy_module("numpy._core.numeric", "numpy.core.numeric")
    table: dict[tuple[str, str], Any] = {
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        # numpy < 3 pickles of protocol < 3 route the raw buffer through a latin1 str and
        # back with _codecs.encode; it is part of ndarray's own reconstruction chain.
        ("_codecs", "encode"): codecs.encode,
        ("collections", "OrderedDict"): collections.OrderedDict,
    }
    for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
        table[(module, "_reconstruct")] = multiarray._reconstruct
        table[(module, "scalar")] = multiarray.scalar
    for module in ("numpy.core.numeric", "numpy._core.numeric"):
        table[(module, "_frombuffer")] = numeric._frombuffer
    for kind in ("csr", "csc", "coo"):
        for suffix in ("matrix", "array"):
            name = f"{kind}_{suffix}"
            target = getattr(scipy.sparse, name)
            for module in ("scipy.sparse", f"scipy.sparse.{kind}", f"scipy.sparse._{kind}"):
                table[(module, name)] = target
    for module in ("copy_reg", "copyreg"):
        table[(module, "_reconstructor")] = copyreg._reconstructor
    for module in ("__builtin__", "builtins"):
        for name in ("object", "set", "frozenset", "list", "dict", "tuple", "bytearray"):
            table[(module, name)] = getattr(builtins, name)
    return table


#: Every ``(module, name)`` the unpickler resolves without being told to, and what it resolves to.
BUILTIN_ALLOW: dict[tuple[str, str], Any] = _builtin_allow()


class SafeUnpickler(pickle.Unpickler):
    """``pickle.Unpickler`` whose ``find_class`` consults the whitelist instead of importing."""

    def __init__(self, file, *, allow: frozenset[tuple[str, str]] = frozenset(), **kwargs):
        super().__init__(file, **kwargs)
        self._allow = frozenset(allow)

    def find_class(self, module: str, name: str):
        if module.split(".")[0] == "chumpy":
            return Ch
        target = BUILTIN_ALLOW.get((module, name))
        if target is not None:
            return target
        if (module, name) in self._allow:
            return super().find_class(module, name)
        raise BlockedGlobal(f"refusing to load {module}.{name}: not in the unpickling whitelist")


def load(path: str | Path, *, allow: frozenset[tuple[str, str]] = frozenset()) -> Any:
    """Unpickle ``path`` through the whitelist. ``allow`` adds ``(module, name)`` pairs to it.

    The file is read with ``encoding="latin1"`` so Python 2 byte strings survive. Legacy
    pickles rebuild dtypes with an idiom numpy deprecates; that warning is about the asset's
    encoding, not about this code, and is silenced for the duration of the load only.
    """
    with open(path, "rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", np.exceptions.VisibleDeprecationWarning)
        return SafeUnpickler(handle, allow=allow, encoding="latin1").load()


def to_array(value: Any) -> np.ndarray:
    """A plain numpy array from a loaded value: unwrap a :class:`Ch` leaf, densify a sparse matrix."""
    if isinstance(value, Ch):
        return np.asarray(value.to_array())
    if scipy.sparse.issparse(value):
        return np.asarray(value.toarray())
    return np.asarray(value)
