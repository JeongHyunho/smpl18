"""Frame addressing in .b3d, exercised against a file this test builds itself.

The payload is a flat run of fixed-width records with no index: a trial's frame is found by
adding up every preceding trial's `length * (raw_sensor_frame_size + passes *
processing_pass_frame_size)`. Trials differ in both length *and* pass count, so an arithmetic
slip lands on a neighbouring frame and still decodes -- it returns a real pose, just the wrong
one. Every trial here is given a distinct pass count and a distinct value pattern so that
landing on the wrong record fails loudly.
"""

import struct

import numpy as np
import pytest

from smpl24.formats import FormatError, b3d
from smpl24.formats._vendor.nimblephysics import SubjectOnDisk_pb2 as pb

NUM_DOFS = 3
NUM_CENTRES = 2
TRIALS = (
    # (length, pass types)
    (4, ("kinematics", "lowPassFilter", "dynamics")),
    (3, ("kinematics", "dynamics")),
    (2, ("kinematics", "lowPassFilter", "dynamics")),
)


def _sensor_frame(trial: int, frame: int) -> bytes:
    frame_message = pb.SubjectOnDiskSensorFrame()
    tag = 100 * trial + 10 * frame
    frame_message.marker_obs.extend([tag + 0.5, 0.0, 0.0])
    return frame_message.SerializeToString()


def _pass_frame(trial: int, frame: int, which: int) -> bytes:
    """A frame whose numbers encode exactly where it lives."""
    payload = pb.SubjectOnDiskProcessingPassFrame()
    tag = 100 * trial + 10 * frame + which
    payload.pos.extend([tag + 0.1, tag + 0.2, tag + 0.3])
    payload.world_frame_joint_centers.extend(
        [tag + 1.0, tag + 2.0, tag + 3.0, tag + 4.0, tag + 5.0, tag + 6.0]
    )
    payload.ground_contact_force.extend([tag, 0.0, 0.0])
    payload.ground_contact_center_of_pressure.extend([0.0, tag, 0.0])
    return payload.SerializeToString()


def _build(path, *, biological_sex: str = "female", trailing: bytes = b"") -> None:
    header = pb.SubjectOnDiskHeader()
    header.num_dofs = NUM_DOFS
    header.version = 4
    header.biological_sex = biological_sex
    header.height_m = 1.7
    header.mass_kg = 62.0
    header.href = "https://example.invalid/subject"
    header.subject_tag.extend(["healthy"])
    header.marker_name.extend(["M1"])
    for name in ("kinematics", "lowPassFilter", "dynamics"):
        entry = header.passes.add()
        entry.pass_type = pb.ProcessingPassType.Value(name)
        entry.model_osim_text = _MINI_OSIM if name != "lowPassFilter" else ""
    header.ground_contact_body.extend(["calcn_r", "calcn_l"])

    header.raw_sensor_frame_size = len(_sensor_frame(0, 0))
    header.processing_pass_frame_size = len(_pass_frame(0, 0, 0))

    for length, passes in TRIALS:
        trial = header.trial_header.add()
        trial.trial_length = length
        trial.trial_timestep = 0.005
        trial.name = f"trial{len(header.trial_header) - 1}"
        trial.trial_tag.extend(["walking"])
        for name in passes:
            ph = trial.processing_pass_header.add()
            ph.type = pb.ProcessingPassType.Value(name)

    blob = header.SerializeToString()
    body = bytearray()
    body += struct.pack("<q", len(blob))
    body += blob
    for trial_index, (length, passes) in enumerate(TRIALS):
        for frame_index in range(length):
            sensor = _sensor_frame(trial_index, frame_index)
            assert len(sensor) == header.raw_sensor_frame_size
            body += sensor
            for pass_index in range(len(passes)):
                record = _pass_frame(trial_index, frame_index, pass_index)
                assert len(record) == header.processing_pass_frame_size
                body += record
    body += trailing
    path.write_bytes(bytes(body))


@pytest.fixture(scope="module")
def synthetic_b3d(tmp_path_factory):
    path = tmp_path_factory.mktemp("b3d") / "synthetic.b3d"
    _build(path)
    return path


_MINI_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="mini">
    <gravity>0 -9.80665 0</gravity>
    <JointSet>
      <objects>
        <CustomJoint name="ground_pelvis">
          <socket_parent_frame>/ground</socket_parent_frame>
          <socket_child_frame>/bodyset/pelvis</socket_child_frame>
          <coordinates><Coordinate name="pelvis_tilt"/><Coordinate name="pelvis_tx"/></coordinates>
        </CustomJoint>
        <PinJoint name="hip_r">
          <socket_parent_frame>/bodyset/pelvis</socket_parent_frame>
          <socket_child_frame>/bodyset/femur_r</socket_child_frame>
          <coordinates><Coordinate name="hip_flexion_r"/></coordinates>
        </PinJoint>
      </objects>
    </JointSet>
  </Model>
</OpenSimDocument>
"""


def test_layout_accounts_for_every_byte(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    assert [t.length for t in file.trials] == [4, 3, 2]
    assert [t.pass_count for t in file.trials] == [3, 2, 3]
    assert file.payload_offset + sum(
        t.length * t.frame_size for t in file.trials
    ) == synthetic_b3d.stat().st_size
    assert file.trailing_bytes == 0


def test_a_pass_is_found_per_trial_by_name_not_per_file(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    # trial 1 has no lowPassFilter, so dynamics sits at a different index there
    assert [t.pass_index("dynamics") for t in file.trials] == [2, 1, 2]
    with pytest.raises(b3d.B3dLayoutError, match="lowPassFilter"):
        file.trials[1].pass_index("lowPassFilter")


def test_every_frame_of_every_trial_addresses_its_own_record(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    for trial_index, trial in enumerate(file.trials):
        for pass_index in range(trial.pass_count):
            frames = b3d.read_pass_frames(
                file, trial_index, range(trial.length), pass_index=pass_index
            )
            assert frames.pass_name == trial.pass_names[pass_index]
            for frame_index in range(trial.length):
                tag = 100 * trial_index + 10 * frame_index + pass_index
                assert frames.pos[frame_index].tolist() == pytest.approx(
                    [tag + 0.1, tag + 0.2, tag + 0.3]
                )
                np.testing.assert_allclose(
                    frames.world_frame_joint_centers[frame_index],
                    [[tag + 1.0, tag + 2.0, tag + 3.0], [tag + 4.0, tag + 5.0, tag + 6.0]],
                )


def test_joint_centres_come_back_shaped_as_points(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    frames = b3d.read_pass_frames(file, 0, [0, 2], pass_index=0)
    assert frames.world_frame_joint_centers.shape == (2, NUM_CENTRES, 3)
    assert frames.pos.shape == (2, NUM_DOFS)
    assert frames.ground_contact_force.shape == (2, 1, 3)
    assert frames.ground_contact_center_of_pressure.shape == (2, 1, 3)
    # fields the file did not store have zero width rather than a made-up value
    assert frames.vel.shape == (2, 0)
    assert frames.ground_contact_wrench.shape == (2, 0, 6)


def test_the_frames_are_the_files_own_numbers_in_the_files_own_frame(synthetic_b3d):
    """No rotation into another world: the format is Y-up and the reader says so, and stops."""
    file = b3d.read(synthetic_b3d)
    assert file.up_axis == "y"
    frames = b3d.read_pass_frames(file, 1, [0], pass_index=file.trials[1].pass_index("dynamics"))
    tag = 100 * 1 + 10 * 0 + 1
    np.testing.assert_array_equal(
        frames.world_frame_joint_centers[0],
        [[tag + 1.0, tag + 2.0, tag + 3.0], [tag + 4.0, tag + 5.0, tag + 6.0]],
    )
    assert file.model("dynamics").gravity == pytest.approx((0.0, -9.80665, 0.0))


def test_frames_carry_their_own_timestamps(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    frames = b3d.read_pass_frames(file, 0, [0, 1, 3], pass_index=0)
    assert frames.timestamps_s.tolist() == pytest.approx([0.0, 0.005, 0.015])
    assert frames.source_rate_hz == pytest.approx(200.0)


def test_reading_past_the_end_of_a_trial_is_refused(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    with pytest.raises(b3d.B3dLayoutError):
        b3d.read_pass_frames(file, 1, [3], pass_index=0)     # trial 1 holds 3 frames: 0..2
    with pytest.raises(b3d.B3dLayoutError):
        b3d.read_pass_frames(file, 0, [0], pass_index=3)
    with pytest.raises(b3d.B3dLayoutError):
        b3d.read_pass_frames(file, 9, [0], pass_index=0)


def test_a_truncated_file_is_refused_rather_than_read_short(synthetic_b3d, tmp_path):
    truncated = tmp_path / "short.b3d"
    truncated.write_bytes(synthetic_b3d.read_bytes()[:-40])
    with pytest.raises(b3d.B3dLayoutError):
        b3d.read(truncated)


def test_bytes_past_the_layout_are_refused_unless_told_otherwise(tmp_path):
    padded = tmp_path / "padded.b3d"
    _build(padded, trailing=b"\0" * 7)
    with pytest.raises(b3d.B3dLayoutError, match="7 bytes past"):
        b3d.read(padded)
    assert b3d.read(padded, strict_size=False).trailing_bytes == 7


def test_the_layout_error_is_a_format_error():
    assert issubclass(b3d.B3dLayoutError, FormatError)


def test_the_model_is_parsed_from_the_pass_the_caller_names(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    assert [p.name for p in file.passes] == ["kinematics", "lowPassFilter", "dynamics"]
    model = file.model("dynamics")
    assert model.independent_coordinate_names == ("pelvis_tilt", "pelvis_tx", "hip_flexion_r")
    assert model.joint_centre_names == ("ground_pelvis", "hip_r")
    assert file.model_text("kinematics") == _MINI_OSIM
    with pytest.raises(b3d.B3dLayoutError, match="lowPassFilter"):
        file.model_text("lowPassFilter")


def test_subject_and_header_fields_are_handed_over_as_stored(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    assert file.subject.biological_sex == "female"
    assert file.subject.height_m == pytest.approx(1.7)
    assert file.subject.mass_kg == pytest.approx(62.0)
    assert file.subject.href == "https://example.invalid/subject"
    assert file.subject.subject_tags == ("healthy",)
    assert file.subject.data_quality == "pilotData"
    assert file.version == 4
    assert file.ground_contact_bodies == ("calcn_r", "calcn_l")
    assert file.marker_names == ("M1",)
    assert [t.name for t in file.trials] == ["trial0", "trial1", "trial2"]
    assert file.trials[0].trial_tags == ("walking",)
    assert file.trials[0].trial_type == "treadmill"


def test_an_unfamiliar_sex_string_passes_through_unmapped(tmp_path):
    """Mapping ``unknown`` to anything is a profile's decision, not the reader's."""
    path = tmp_path / "unknown.b3d"
    _build(path, biological_sex="unknown")
    assert b3d.read(path).subject.biological_sex == "unknown"


def test_declared_dof_count_must_match_the_model(synthetic_b3d):
    """A header whose num_dofs disagrees with its own model means one of them is not ours."""
    file = b3d.read(synthetic_b3d)
    assert file.num_dofs == len(file.model("dynamics").independent_coordinate_names)


def test_pos_and_centres_are_read_as_float64(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    frames = b3d.read_pass_frames(file, 2, [0], pass_index=0)
    assert frames.pos.dtype == np.float64
    assert frames.world_frame_joint_centers.dtype == np.float64


def test_sensor_frames_address_their_own_record(synthetic_b3d):
    file = b3d.read(synthetic_b3d)
    for trial_index, trial in enumerate(file.trials):
        frames = b3d.read_sensor_frames(file, trial_index, range(trial.length))
        assert frames.marker_obs.shape == (trial.length, 1, 3)
        for frame_index in range(trial.length):
            tag = 100 * trial_index + 10 * frame_index
            assert frames.marker_obs[frame_index, 0].tolist() == pytest.approx(
                [tag + 0.5, 0.0, 0.0]
            )
