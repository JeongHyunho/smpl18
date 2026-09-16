"""Reading a ``.mat``: variables by name, structs as structs, and a plain-dict view on request."""

import numpy as np
import scipy.io
from scipy.io.matlab import mat_struct

from smpl18.formats import mat


def write_mat(path):
    scipy.io.savemat(
        str(path),
        {
            "centres": np.arange(12.0).reshape(2, 2, 3),
            "info": {
                "name": "s001",
                "height_cm": 171.5,
                "count": 3,
                "nested": {"eye": np.eye(2)},
            },
            "trials": np.array([{"id": 1}, {"id": 2}], dtype=object),
        },
    )
    return path


def test_variables_come_back_by_name_without_the_bookkeeping_keys(tmp_path):
    variables = mat.read(write_mat(tmp_path / "trial.mat"))
    assert set(variables) == {"centres", "info", "trials"}
    np.testing.assert_array_equal(variables["centres"], np.arange(12.0).reshape(2, 2, 3))


def test_structs_arrive_squeezed_with_fields_as_attributes(tmp_path):
    info = mat.read(write_mat(tmp_path / "trial.mat"))["info"]
    assert isinstance(info, mat_struct)
    assert info.name == "s001"
    assert info.height_cm == 171.5
    assert info.count == 3
    np.testing.assert_array_equal(info.nested.eye, np.eye(2))


def test_struct_to_dict_nests_structs_and_lists_struct_arrays(tmp_path):
    variables = mat.read(write_mat(tmp_path / "trial.mat"))
    info = mat.struct_to_dict(variables["info"])
    assert info["name"] == "s001"
    assert info["height_cm"] == 171.5
    np.testing.assert_array_equal(info["nested"]["eye"], np.eye(2))
    assert mat.struct_to_dict(variables["trials"]) == [{"id": 1}, {"id": 2}]


def test_struct_to_dict_leaves_plain_values_alone():
    array = np.arange(3.0)
    assert mat.struct_to_dict(array) is array
    assert mat.struct_to_dict("text") == "text"
    assert mat.struct_to_dict(2.5) == 2.5
