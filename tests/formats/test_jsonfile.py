"""Reading a JSON file: UTF-8, the object as encoded."""

import json

from smpl18.formats import jsonfile


def test_the_object_comes_back_as_encoded(tmp_path):
    path = tmp_path / "metadata.json"
    payload = {"inverse_kinematics": {"pelvis": {}, "tibia_r": {}}, "rate_hz": 100, "ok": True}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert jsonfile.read(path) == payload


def test_non_ascii_text_is_read_as_utf8(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps({"subject": "이름", "unit": "µm"}, ensure_ascii=False),
                    encoding="utf-8")
    assert jsonfile.read(str(path)) == {"subject": "이름", "unit": "µm"}
