"""The whitelisting unpickler: chumpy becomes a stub, numpy and scipy load, everything else is named and refused."""

from __future__ import annotations

import collections
import pickle
import sys

import numpy as np
import pytest
import scipy.sparse

from smpl18.formats import pickle_safe
from smpl18.formats.pickle_safe import BlockedGlobal, Ch, SafeUnpickler


def _dump(obj, path, protocol):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=protocol)


@pytest.mark.parametrize("protocol", [0, 2, 4, 5])
def test_chumpy_classes_become_the_passive_stub_without_importing_chumpy(
    tmp_path, rng, protocol, make_fake_pickle
):
    path = tmp_path / "model.pkl"
    plain = make_fake_pickle(path, rng, protocol=protocol)
    assert "chumpy" not in sys.modules

    loaded = pickle_safe.load(path)

    assert "chumpy" not in sys.modules
    assert isinstance(loaded["v_template"], Ch)
    np.testing.assert_array_equal(pickle_safe.to_array(loaded["v_template"]), plain["v_template"])
    np.testing.assert_array_equal(pickle_safe.to_array(loaded["shapedirs"]), plain["shapedirs"])
    np.testing.assert_array_equal(pickle_safe.to_array(loaded["J_regressor"]), plain["J_regressor"])
    assert loaded["kintree_table"][0, 0] == 4294967295
    assert loaded["bs_style"] == "lbs"


@pytest.mark.parametrize("protocol", [0, 2, 4, 5])
def test_numpy_scipy_and_containers_round_trip(tmp_path, protocol):
    payload = {
        "floats": np.arange(6.0).reshape(2, 3),
        "ints": np.arange(4, dtype=np.int32),
        "bools": np.array([True, False]),
        "scalar": np.float64(2.5),
        "dtype": np.dtype("float32"),
        "set": {1, 2},
        "frozen": frozenset({"a"}),
        "ordered": collections.OrderedDict([("k", 1)]),
        "csr": scipy.sparse.csr_matrix(np.eye(3)),
        "csc": scipy.sparse.csc_matrix(np.eye(3)),
        "coo": scipy.sparse.coo_matrix(np.eye(3)),
        "bytes": bytearray(b"xy"),
    }
    path = tmp_path / f"p{protocol}.pkl"
    _dump(payload, path, protocol)

    loaded = pickle_safe.load(path)

    np.testing.assert_array_equal(loaded["floats"], payload["floats"])
    np.testing.assert_array_equal(loaded["ints"], payload["ints"])
    np.testing.assert_array_equal(loaded["bools"], payload["bools"])
    assert loaded["scalar"] == 2.5 and loaded["dtype"] == np.dtype("float32")
    assert loaded["set"] == {1, 2} and loaded["frozen"] == frozenset({"a"})
    assert loaded["ordered"] == collections.OrderedDict([("k", 1)])
    for key in ("csr", "csc", "coo"):
        np.testing.assert_array_equal(pickle_safe.to_array(loaded[key]), np.eye(3))
    assert loaded["bytes"] == bytearray(b"xy")


@pytest.mark.parametrize(
    "stream,name",
    [
        (b"cos\nsystem\n(S'echo pwned'\ntR.", "os.system"),
        (b"cbuiltins\neval\n(S'1+1'\ntR.", "builtins.eval"),
        (b"csubprocess\nPopen\n(S'x'\ntR.", "subprocess.Popen"),
    ],
)
def test_a_global_outside_the_whitelist_is_refused_by_name(tmp_path, stream, name):
    path = tmp_path / "bad.pkl"
    path.write_bytes(stream)
    with pytest.raises(BlockedGlobal, match=name.replace(".", r"\.")):
        pickle_safe.load(path)


def test_allow_extends_the_whitelist_for_one_call(tmp_path):
    path = tmp_path / "deque.pkl"
    _dump(collections.deque([1, 2]), path, 2)
    with pytest.raises(BlockedGlobal, match="collections.deque"):
        pickle_safe.load(path)
    assert pickle_safe.load(path, allow=frozenset({("collections", "deque")})) == collections.deque([1, 2])
    with pytest.raises(BlockedGlobal):
        pickle_safe.load(path)


def test_find_class_resolves_legacy_and_current_spellings():
    unpickler = SafeUnpickler.__new__(SafeUnpickler)
    unpickler._allow = frozenset()
    assert SafeUnpickler.find_class(unpickler, "chumpy.ch", "Ch") is Ch
    assert SafeUnpickler.find_class(unpickler, "chumpy", "Ch") is Ch
    legacy = SafeUnpickler.find_class(unpickler, "numpy.core.multiarray", "_reconstruct")
    current = SafeUnpickler.find_class(unpickler, "numpy._core.multiarray", "_reconstruct")
    assert legacy is current
    assert SafeUnpickler.find_class(unpickler, "scipy.sparse.csc", "csc_matrix") is scipy.sparse.csc_matrix
    assert SafeUnpickler.find_class(unpickler, "copy_reg", "_reconstructor") is not None
    assert SafeUnpickler.find_class(unpickler, "__builtin__", "object") is object
    with pytest.raises(BlockedGlobal):
        SafeUnpickler.find_class(unpickler, "os", "system")
    with pytest.raises(BlockedGlobal):
        SafeUnpickler.find_class(unpickler, "builtins", "eval")
    with pytest.raises(BlockedGlobal):
        SafeUnpickler.find_class(unpickler, "numpy", "load")


def test_stub_state_forms_and_payload_preference():
    leaf = Ch()
    leaf.__setstate__({"r": np.ones(2), "x": np.arange(6).reshape(2, 3)})
    np.testing.assert_array_equal(leaf.to_array(), np.arange(6).reshape(2, 3))

    leaf = Ch()
    leaf.__setstate__(({"_x": np.zeros(3)}, None))       # copyreg (dict, slots) pair
    np.testing.assert_array_equal(leaf.to_array(), np.zeros(3))

    leaf = Ch()
    leaf.__setstate__({"small": np.zeros(2), "big": np.ones(5)})
    np.testing.assert_array_equal(leaf.to_array(), np.ones(5))

    leaf = Ch()
    leaf.__setstate__([1, 2])
    assert leaf.__dict__["_state"] == [1, 2]
    with pytest.raises(TypeError, match="no array payload"):
        leaf.to_array()


def test_to_array_unwraps_stub_sparse_and_plain_values():
    leaf = Ch()
    leaf.__setstate__({"x": np.arange(3.0)})
    np.testing.assert_array_equal(pickle_safe.to_array(leaf), np.arange(3.0))
    np.testing.assert_array_equal(pickle_safe.to_array(scipy.sparse.csr_matrix(np.eye(2))), np.eye(2))
    np.testing.assert_array_equal(pickle_safe.to_array([1, 2]), np.array([1, 2]))
