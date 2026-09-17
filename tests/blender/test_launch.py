import json
import subprocess
from pathlib import Path

import pytest
import yaml

from smpl18.blender import launch

SHIPPED = launch.shipped_render_settings()


def test_the_shipped_settings_hold_every_value_a_scene_needs() -> None:
    settings, record = launch.load_render_settings([SHIPPED])
    launch.validate_render_settings(settings)
    assert record[0]["path"] == "default.yaml" and len(record[0]["sha256"]) == 64
    for section, key in launch.REQUIRED_SETTINGS:
        assert launch.render_setting(settings, section, key) is not None


def test_a_second_file_updates_sections_of_the_first(tmp_path) -> None:
    mine = tmp_path / "mine.yaml"
    mine.write_text(yaml.safe_dump({"schema": launch.RENDER_SCHEMA, "id": "mine",
                                    "render": {"samples": 4}}), encoding="utf-8")
    settings, record = launch.load_render_settings([SHIPPED, mine])
    assert launch.render_setting(settings, "render", "samples") == 4
    assert launch.render_setting(settings, "render", "engine") == "cycles"   # kept from the first
    assert [entry["path"] for entry in record] == ["default.yaml", "mine.yaml"]


def test_settings_that_are_not_render_settings_are_refused(tmp_path) -> None:
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump({"schema": "smpl18_settings_v1"}), encoding="utf-8")
    with pytest.raises(launch.BlenderError, match=launch.RENDER_SCHEMA):
        launch.load_render_settings([other])
    with pytest.raises(launch.BlenderError, match="no render settings file"):
        launch.load_render_settings([])
    with pytest.raises(launch.BlenderError, match="no render settings named"):
        launch.shipped_render_settings("nothing-like-this")


@pytest.mark.parametrize("broken, message", [
    ({"render": {"engine": "povray"}}, "render.engine"),
    ({"render": {"resolution": [0, 100]}}, "render.resolution"),
    ({"render": {"frame_step": 0}}, "render.frame_step"),
    ({"material": {"colour": [0.5, 0.5]}}, "material.colour"),
])
def test_a_value_that_cannot_work_is_refused_before_blender_starts(broken, message) -> None:
    settings, _ = launch.load_render_settings([SHIPPED])
    for section, values in broken.items():
        settings[section] = {**settings[section], **values}
    with pytest.raises(launch.BlenderError, match=message.replace(".", r"\.")):
        launch.validate_render_settings(settings)


def test_a_missing_section_is_named() -> None:
    settings, _ = launch.load_render_settings([SHIPPED])
    del settings["camera"]
    with pytest.raises(launch.BlenderError, match=r"camera\.distance_m"):
        launch.validate_render_settings(settings)


def test_the_command_runs_the_script_with_the_plan_and_settings(tmp_path) -> None:
    command = launch.blender_command("/usr/bin/blender", plan=tmp_path / "p.npz",
                                    settings=tmp_path / "r.json", blend=tmp_path / "s.blend",
                                    render=tmp_path / "frames", quiet=True)
    assert command[0] == "/usr/bin/blender"
    assert "--background" in command and "--factory-startup" in command
    assert command[command.index("--python") + 1] == str(launch.scene_script())
    after = command[command.index("--") + 1:]
    assert after[:2] == ["--plan", str(tmp_path / "p.npz")]
    assert "--blend" in after and "--render" in after and "--quiet" in after
    assert launch.scene_script().is_file()


def test_asking_for_nothing_is_refused(tmp_path) -> None:
    with pytest.raises(launch.BlenderError, match="nothing to do"):
        launch.blender_command("blender", plan=tmp_path / "p.npz", settings=tmp_path / "r.json")


def test_settings_are_written_where_the_script_can_read_them(tmp_path) -> None:
    settings, _ = launch.load_render_settings([SHIPPED])
    path = launch.write_render_settings(tmp_path / "deep" / "render.json", settings)
    assert json.loads(path.read_text(encoding="utf-8"))["render"]["engine"] == "cycles"


def test_a_blender_that_is_not_there_says_how_to_name_one(tmp_path) -> None:
    missing = tmp_path / "no-such-blender"
    with pytest.raises(launch.BlenderError, match="Install Blender"):
        launch.run_blender(missing, plan=tmp_path / "p.npz", settings=tmp_path / "r.json",
                           blend=tmp_path / "s.blend")


def test_a_blender_that_fails_hands_back_its_own_words(tmp_path, monkeypatch) -> None:
    def refuse(command, **_):
        assert Path(command[command.index("--python") + 1]).name == "scene.py"
        return subprocess.CompletedProcess(command, 2, stdout="building\nERROR: 3.2 mm off\n",
                                           stderr="")

    monkeypatch.setattr(subprocess, "run", refuse)
    with pytest.raises(launch.BlenderError, match="3.2 mm off"):
        launch.run_blender("blender", plan=tmp_path / "p.npz", settings=tmp_path / "r.json",
                           render=tmp_path / "frames")
