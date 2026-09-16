"""Reading a ``.c3d`` through ezc3d into the same table a ``.trc`` gives."""

from __future__ import annotations

import numpy as np
import pytest

from smpl18.formats import FormatError, TrcTable, c3d

ezc3d = pytest.importorskip("ezc3d", reason="ezc3d is not installed")

LABELS = ["A", "B"]
RATE_HZ = 100.0
FRAMES = 3


def write_c3d(path, *, units: str = "mm", invalid: tuple[tuple[int, int], ...] = ()):
    """Two markers, three frames, a distinct number in every cell; ``invalid`` marks samples
    by a negative residual, the C3D convention."""
    container = ezc3d.c3d()
    container["parameters"]["POINT"]["RATE"]["value"] = [RATE_HZ]
    container["parameters"]["POINT"]["LABELS"]["value"] = LABELS
    container["parameters"]["POINT"]["UNITS"]["value"] = [units]
    points = np.ones((4, len(LABELS), FRAMES))
    points[:3] = np.arange(3 * len(LABELS) * FRAMES, dtype=np.float64).reshape(
        3, len(LABELS), FRAMES
    ) * 10.0 + 1.0
    container["data"]["points"] = points
    residuals = np.zeros((1, len(LABELS), FRAMES))
    for marker, frame in invalid:
        residuals[0, marker, frame] = -1.0
    container["data"]["meta_points"]["residuals"] = residuals
    container.write(str(path))
    return path, np.transpose(points[:3], (2, 1, 0))


def test_labels_rate_units_and_points_come_from_the_parameter_block(tmp_path):
    path, expected = write_c3d(tmp_path / "m.c3d")
    table = c3d.read(path)
    assert isinstance(table, TrcTable)
    assert table.labels == tuple(LABELS)
    assert table.data_rate_hz == RATE_HZ
    assert table.units == "mm"
    assert table.positions.shape == (FRAMES, len(LABELS), 3)
    np.testing.assert_allclose(table.positions, expected)
    assert table.valid.all()


def test_units_are_reported_not_converted(tmp_path):
    path, expected = write_c3d(tmp_path / "metres.c3d", units="m")
    table = c3d.read(path)
    assert table.units == "m"
    np.testing.assert_allclose(table.positions, expected)


def test_a_negative_residual_marks_the_sample_invalid_and_nan(tmp_path):
    path, _ = write_c3d(tmp_path / "gap.c3d", invalid=((1, 2),))
    table = c3d.read(path)
    assert not table.valid[2, 1]
    assert np.isnan(table.positions[2, 1]).all()
    assert table.valid.sum() == FRAMES * len(LABELS) - 1
    assert np.isfinite(table.positions[table.valid]).all()


def test_times_count_from_the_first_stored_frame_at_the_point_rate(tmp_path):
    path, _ = write_c3d(tmp_path / "m.c3d")
    table = c3d.read(path)
    np.testing.assert_allclose(table.times_s, np.arange(FRAMES) / RATE_HZ)
    assert table.frame_numbers.shape == (FRAMES,)
    assert table.frame_numbers[0] == table.metadata["first_frame"]


def test_a_file_that_is_not_a_c3d_is_refused_by_the_library(tmp_path):
    path = tmp_path / "not.c3d"
    path.write_bytes(b"not a c3d")
    with pytest.raises((FormatError, RuntimeError, ValueError, OSError)):
        c3d.read(path)
