"""Reading an OpenSim-format ``.trc``: every row the file says it has, in the file's own units.

The header's end is found rather than assumed. An earlier reader started at a fixed line and so
dropped frame 0 of every trial -- silently, because the remaining frames were all valid. The
header's own ``NumFrames`` is the check that would have caught it, so it is asserted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smpl24.formats import FormatError, TrcTable, trc

DT_S = 0.01
FRAMES = 6
LABELS = ["R_IAS", "L_IAS", "R_FAL"]


def positions_mm(frames: int = FRAMES) -> np.ndarray:
    """A distinct number in every cell, so a shifted column or row is caught."""
    return (
        np.arange(frames * len(LABELS) * 3, dtype=np.float64).reshape(frames, len(LABELS), 3)
        * 10.0
        + 1.0
    )


def write_trc(
    path: Path,
    *,
    positions: np.ndarray,
    blank_after_axis_row: bool = False,
    declared_frames: int | None = None,
    blank: tuple[tuple[int, int], ...] = (),
    units: str = "mm",
) -> Path:
    """The archive-style OpenSim ``.trc``: five header lines, tab separated."""
    frames = positions.shape[0]
    name_row = ["Frame#", "Time"]
    axis_row = ["", ""]
    for slot, name in enumerate(LABELS, start=1):
        name_row.extend([name, "", ""])
        axis_row.extend([f"X{slot}", f"Y{slot}", f"Z{slot}"])
    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
        "OrigDataStartFrame\tOrigNumFrames",
        f"100\t120\t{frames if declared_frames is None else declared_frames}\t{len(LABELS)}"
        f"\t{units}\t100\t1\t{frames}",
        "\t".join(name_row),
        "\t".join(axis_row),
    ]
    if blank_after_axis_row:
        lines.append("")
    for frame in range(frames):
        row = [str(frame + 1), f"{frame * DT_S:.6f}"]
        for marker in range(len(LABELS)):
            if (frame, marker) in blank:
                row.extend(["", "", ""])
            else:
                row.extend(f"{value:.6f}" for value in positions[frame, marker])
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_labels_rates_and_units_are_read_from_the_header(tmp_path):
    table = trc.read(write_trc(tmp_path / "m.trc", positions=positions_mm()))
    assert isinstance(table, TrcTable)
    assert table.labels == tuple(LABELS)
    assert table.data_rate_hz == 100.0
    assert table.camera_rate_hz == 120.0
    assert table.units == "mm"
    assert table.metadata["OrigDataStartFrame"] == "1"


def test_positions_are_the_files_own_numbers_in_the_files_own_units(tmp_path):
    """Millimetres stay millimetres; a unit change belongs to whoever knows what they want."""
    expected = positions_mm()
    table = trc.read(write_trc(tmp_path / "m.trc", positions=expected))
    assert table.positions.shape == (FRAMES, len(LABELS), 3)
    assert table.positions.dtype == np.float64
    np.testing.assert_allclose(table.positions, expected)
    assert table.valid.all()
    np.testing.assert_allclose(table.positions_of("R_FAL"), expected[:, 2])


def test_frame_numbers_and_times_are_the_first_two_columns(tmp_path):
    table = trc.read(write_trc(tmp_path / "m.trc", positions=positions_mm()))
    np.testing.assert_array_equal(table.frame_numbers, np.arange(1, FRAMES + 1))
    np.testing.assert_allclose(table.times_s, np.arange(FRAMES) * DT_S)
    assert table.frame_count == FRAMES


def test_frame_zero_is_kept_whether_or_not_a_blank_line_follows_the_axis_row(tmp_path):
    expected = positions_mm()
    plain = trc.read(write_trc(tmp_path / "plain.trc", positions=expected))
    spaced = trc.read(
        write_trc(tmp_path / "spaced.trc", positions=expected, blank_after_axis_row=True)
    )
    for table in (plain, spaced):
        assert table.frame_count == FRAMES
        assert table.times_s[0] == 0.0
        np.testing.assert_allclose(table.positions[0], expected[0])


def test_a_num_frames_that_disagrees_with_the_rows_is_refused(tmp_path):
    path = write_trc(tmp_path / "short.trc", positions=positions_mm(), declared_frames=FRAMES + 1)
    with pytest.raises(FormatError, match="declares NumFrames"):
        trc.read(path)


def test_blank_fields_are_invalid_samples_and_nan(tmp_path):
    blank = ((2, 0), (4, 2))
    table = trc.read(write_trc(tmp_path / "gaps.trc", positions=positions_mm(), blank=blank))
    for frame, marker in blank:
        assert not table.valid[frame, marker]
        assert np.isnan(table.positions[frame, marker]).all()
    assert table.valid.sum() == FRAMES * len(LABELS) - len(blank)
    assert np.isfinite(table.positions[table.valid]).all()


def test_a_zero_triplet_is_a_number_not_a_verdict(tmp_path):
    """Some writers park a lost marker at the origin. Reading that as lost is a policy, and
    policies are bound by profiles; the reader reports what the file wrote."""
    positions = positions_mm()
    positions[1, 1] = 0.0
    table = trc.read(write_trc(tmp_path / "zero.trc", positions=positions))
    assert table.valid[1, 1]
    assert table.positions[1, 1].tolist() == [0.0, 0.0, 0.0]


def test_a_file_without_an_axis_row_is_refused(tmp_path):
    path = tmp_path / "noaxis.trc"
    path.write_text(
        "PathFileType\t4\t(X/Y/Z)\tx\nDataRate\tUnits\n100\tmm\nFrame#\tTime\tA\n"
        "1\t0\t1\t2\t3\n",
        encoding="utf-8",
    )
    with pytest.raises(FormatError, match="no axis row"):
        trc.read(path)


def test_a_file_that_declares_no_units_is_refused(tmp_path):
    path = write_trc(tmp_path / "nounits.trc", positions=positions_mm(), units="")
    with pytest.raises(FormatError, match="Units"):
        trc.read(path)


def test_an_unknown_label_is_a_key_error_naming_it(tmp_path):
    table = trc.read(write_trc(tmp_path / "m.trc", positions=positions_mm()))
    with pytest.raises(KeyError, match="R_IPS"):
        table.positions_of("R_IPS")
