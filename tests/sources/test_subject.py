import pytest
import yaml

from smpl18.sources.subject import SubjectError, SubjectInfo


def write(path, **data):
    path.write_text(yaml.safe_dump({"schema": "smpl18_subject_v1", **data}), encoding="utf-8")
    return path


def test_a_subject_file_names_the_body_and_the_measurements(tmp_path) -> None:
    path = write(tmp_path / "s.yaml", id="S07", gender="female",
                 measurements={"marker_radius": 0.007, "knee_width_left": "0.1"})
    subject = SubjectInfo.load(path)
    assert (subject.id, subject.gender, subject.source) == ("S07", "female", "file")
    assert subject.measurements == {"marker_radius": 0.007, "knee_width_left": 0.1}
    assert len(subject.sha256) == 64
    assert subject.record()["file"] == "s.yaml"


def test_command_line_values_take_precedence(tmp_path) -> None:
    subject = SubjectInfo.load(write(tmp_path / "s.yaml", id="S07", gender="female",
                                     measurements={"marker_radius": 0.007}))
    changed = subject.with_overrides(gender="neutral", measurements={"marker_radius": 0.008})
    assert changed.gender == "neutral" and changed.measurements["marker_radius"] == 0.008
    assert changed.source == "file+argument"
    assert changed.gender_from == "argument" and subject.gender_from == "file"
    assert subject.with_overrides(measurements={"w": 1.0}).gender_from == "file"
    assert subject.with_overrides().source == "file"


@pytest.mark.parametrize("data, message", [
    ({"id": "S1", "gender": "other"}, "gender"),
    ({"id": "S1", "gender": "male", "height": 1.8}, "unknown keys"),
    ({"gender": "male"}, "missing key"),
    ({"id": "a/b", "gender": "male"}, "directory"),
    ({"id": "S1", "gender": "male", "measurements": {"w": "wide"}}, "not a number"),
])
def test_bad_files_are_refused(tmp_path, data, message) -> None:
    with pytest.raises(SubjectError, match=message):
        SubjectInfo.load(write(tmp_path / "s.yaml", **data))


def test_the_schema_must_be_named(tmp_path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"id": "S1", "gender": "male"}), encoding="utf-8")
    with pytest.raises(SubjectError, match="schema"):
        SubjectInfo.load(path)
