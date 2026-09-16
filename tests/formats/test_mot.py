"""Reading an OpenSim ``.mot``: the header flag is reported, the numbers are not touched."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smpl18.formats import FormatError, mot

DT_S = 0.01
FRAMES = 5

# One value per coordinate, held constant: this file is about columns and units, not motion.
COORDINATES = {
    "pelvis_tilt": 3.0,
    "pelvis_tx": 0.5,
    "hip_flexion_r": 30.0,
    "knee_angle_r": 12.5,
    "knee_angle_r_beta": 0.1466979,
}


def write_mot(path: Path, *, in_degrees: str | None = "yes", columns=None) -> Path:
    """A ``.mot`` with the usual header, declaring degrees the way OpenSim writes it."""
    values = COORDINATES if columns is None else columns
    names = ["time", *values]
    header = []
    if in_degrees is not None:
        header.append(f"inDegrees={in_degrees}")
    header += [
        f"name={path.stem}",
        "DataType=double",
        "version=3",
        "OpenSimVersion=4.5.2-2025-05-03-6a4c6ec41",
        "endheader",
    ]
    lines = [*header, "\t".join(names)]
    for frame in range(FRAMES):
        row = [repr(frame * DT_S), *(repr(float(v)) for v in values.values())]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_columns_and_values_come_back_as_written(tmp_path):
    table = mot.read(write_mot(tmp_path / "ik.mot"))
    assert table.columns == ("time", *COORDINATES)
    assert table.values.shape == (FRAMES, 1 + len(COORDINATES))
    assert table.values.dtype == np.float64
    assert table.frame_count == FRAMES
    # degrees stay degrees: 30.0, not pi/6
    assert table.column("hip_flexion_r")[0] == 30.0
    assert table.column("pelvis_tx")[0] == 0.5


def test_the_degrees_flag_is_reported_not_applied(tmp_path):
    assert mot.read(write_mot(tmp_path / "deg.mot", in_degrees="yes")).in_degrees is True
    assert mot.read(write_mot(tmp_path / "rad.mot", in_degrees="no")).in_degrees is False
    assert mot.read(write_mot(tmp_path / "none.sto", in_degrees=None)).in_degrees is None


def test_the_header_lines_are_kept_verbatim(tmp_path):
    table = mot.read(write_mot(tmp_path / "ik.mot"))
    assert table.header["inDegrees"] == "yes"
    assert table.header["name"] == "ik"
    assert table.header["OpenSimVersion"] == "4.5.2-2025-05-03-6a4c6ec41"


def test_time_is_the_time_column_or_else_the_first_column(tmp_path):
    table = mot.read(write_mot(tmp_path / "ik.mot"))
    np.testing.assert_allclose(table.time_s, np.arange(FRAMES) * DT_S)
    path = tmp_path / "notime.sto"
    path.write_text("endheader\na\tb\n1\t2\n3\t4\n", encoding="utf-8")
    np.testing.assert_array_equal(mot.read(path).time_s, [1.0, 3.0])


def test_a_dependent_coordinate_column_is_still_a_column(tmp_path):
    """Whether ``knee_angle_r_beta`` belongs in a pose is a model question; the file has it."""
    table = mot.read(write_mot(tmp_path / "ik.mot"))
    assert "knee_angle_r_beta" in table.columns
    with pytest.raises(KeyError, match="no column named"):
        table.column("absent")


def test_a_file_without_endheader_is_refused(tmp_path):
    path = tmp_path / "broken.mot"
    path.write_text("inDegrees=yes\ntime\ta\n0\t1\n", encoding="utf-8")
    with pytest.raises(FormatError, match="endheader"):
        mot.read(path)


def test_a_column_count_mismatch_is_refused(tmp_path):
    path = tmp_path / "narrow.mot"
    path.write_text("endheader\ntime\ta\tb\n0\t1\n0.01\t2\n", encoding="utf-8")
    with pytest.raises(FormatError, match="numeric columns against"):
        mot.read(path)


def test_a_ragged_or_non_numeric_row_is_refused(tmp_path):
    path = tmp_path / "ragged.mot"
    path.write_text("endheader\ntime\ta\n0\t1\n0.01\n", encoding="utf-8")
    with pytest.raises(FormatError, match="not numeric or not rectangular"):
        mot.read(path)


def test_a_table_with_no_rows_keeps_its_columns(tmp_path):
    path = tmp_path / "empty.mot"
    path.write_text("endheader\ntime\ta\n", encoding="utf-8")
    table = mot.read(path)
    assert table.columns == ("time", "a")
    assert table.values.shape == (0, 2)
