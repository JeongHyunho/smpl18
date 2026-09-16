import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from smpl18 import __version__, cli, synthetic
from smpl18.cli import main
from smpl18.corpus import read_corpus
from smpl18.formats import trc
from smpl18.model.demo import demo_model, write_models
from smpl18.skeleton.kinematics import rest_joints

PACKAGE = Path(__file__).resolve().parents[1]
CONFIGS = PACKAGE / "configs"


def test_version_flag_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"smpl18 {__version__}"


def test_no_arguments_prints_help_and_exits_zero(capsys) -> None:
    assert main([]) == 0
    assert "convert" in capsys.readouterr().out


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Stand-in models, a quick settings file, a short marker trial and its subject file."""
    root = tmp_path_factory.mktemp("cli")
    write_models(root / "models")
    settings = yaml.safe_load((CONFIGS / "settings" / "default.yaml").read_text("utf-8"))
    settings = copy.deepcopy(settings)
    settings["shape"]["sample_frames"] = 40
    settings["reduce"]["sample_frames"] = 100
    (root / "quick.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    rest = rest_joints(demo_model(), np.zeros(10))
    local, trans = synthetic.walking_motion(30, 100.0, seed=4)
    labels, markers = synthetic.conventional_markers(rest, local, trans)
    trc.write(root / "walk.trc", labels, markers * 1000.0, rate_hz=100.0, units="mm")
    (root / "subject.yaml").write_text(yaml.safe_dump({
        "schema": "smpl18_subject_v1", "id": "P01", "gender": "female",
        "measurements": {k: v for k, v in synthetic.MEASUREMENTS.items() if k != "marker_radius"},
    }), encoding="utf-8")
    return root


def markers_command(root, out, *extra):
    return ["convert", "markers", "--input", str(root / "walk.trc"),
            "--markerset", str(CONFIGS / "markersets" / "conventional_full_body.yaml"),
            "--up-axis", "y", "--settings", str(root / "quick.yaml"),
            "--models", str(root / "models"), "--out", str(out), *extra]


def test_convert_markers_writes_a_corpus_and_reports(workspace, tmp_path, capsys) -> None:
    out = tmp_path / "corpus"
    code = main(markers_command(workspace, out, "--subject", str(workspace / "subject.yaml"),
                                "--measurement", "marker_radius=0.007", "--quiet"))
    assert code == 0
    printed = capsys.readouterr()
    assert "wrote subject P01: 1 trial(s)" in printed.out
    assert "stand-in" in printed.err
    subject = read_corpus(out).subject("P01")
    assert subject.gender == "female"
    assert subject.record["subject_file"]["measurements"]["marker_radius"] == 0.007
    manifest = subject.trial("walk").manifest
    assert manifest["profile"]["id"] == "adhoc"
    assert manifest["profile"]["command"] == "convert markers"
    assert manifest["profile"]["arguments"]["markerset"] == "conventional_full_body.yaml"

    assert main(["info", str(out)]) == 0
    info = capsys.readouterr().out
    assert "P01: female" in info and "walk: 30 frames at 100 Hz" in info
    assert main(["info", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["trials"] == 1


def test_a_missing_measurement_is_an_error_not_a_default(workspace, tmp_path, capsys) -> None:
    code = main(markers_command(workspace, tmp_path / "corpus",
                                "--subject", str(workspace / "subject.yaml"), "--quiet"))
    assert code == 1
    assert "marker_radius" in capsys.readouterr().err


def test_a_subject_needs_an_id_and_a_gender(workspace, tmp_path, capsys) -> None:
    code = main(markers_command(workspace, tmp_path / "corpus", "--subject-id", "X"))
    assert code == 2
    assert "--subject-id and --gender" in capsys.readouterr().err
    code = main(markers_command(workspace, tmp_path / "corpus", "--subject-id", "X",
                                "--gender", "male", "--measurement", "radius"))
    assert code == 2
    assert "NAME=METRES" in capsys.readouterr().err


def test_the_model_directory_must_be_named(workspace, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SMPL18_MODELS", raising=False)
    command = markers_command(workspace, tmp_path / "corpus", "--subject-id", "X",
                              "--gender", "male")
    at = command.index("--models")
    del command[at:at + 2]
    assert main(command) == 1
    assert "SMPL18_MODELS" in capsys.readouterr().err


def test_required_arguments_are_required(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["convert", "markers", "--input", "a.trc"])
    assert "--markerset" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["convert", "bvh", "--input", "a.bvh", "--correspondence", "c.yaml",
              "--up-axis", "y", "--out", "o", "--settings", "s.yaml"])
    assert "--length-unit" in capsys.readouterr().err


def test_demo_models_are_written_and_labelled(tmp_path, capsys) -> None:
    assert main(["demo-models", "--out", str(tmp_path)]) == 0
    assert "SMPL_NEUTRAL_clean.npz" in capsys.readouterr().out
    assert (tmp_path / "SMPL_FEMALE_clean.npz").exists()


def test_fbx2bvh_drives_blender_in_the_background(tmp_path, monkeypatch, capsys) -> None:
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        Path(command[-1]).write_text("HIERARCHY\n", encoding="utf-8")

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    source, target = tmp_path / "clip.fbx", tmp_path / "clip.bvh"
    assert main(["fbx2bvh", "--input", str(source), "--out", str(target),
                 "--blender", "blender"]) == 0
    command = seen["command"]
    assert command[:3] == ["blender", "--background", "--factory-startup"]
    assert command[-3:] == ["--", str(source.resolve()), str(target.resolve())]
    assert "export_anim.bvh" in command[4] and "import_scene.fbx" in command[4]
    assert "--up-axis z" in capsys.readouterr().out


def test_fbx2bvh_reports_what_blender_said(tmp_path, monkeypatch, capsys) -> None:
    class Failed:
        returncode, stdout, stderr = 1, "", "no armature in clip.fbx"

    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: Failed())
    assert main(["fbx2bvh", "--input", str(tmp_path / "clip.fbx"),
                 "--out", str(tmp_path / "clip.bvh"), "--blender", "blender"]) == 2
    assert "no armature" in capsys.readouterr().err


def test_extract_model_needs_the_width(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["extract-model", "--pkl", "m.pkl", "--gender", "male", "--out", "o"])
    assert "--num-betas" in capsys.readouterr().err


def test_a_second_run_for_a_subject_needs_replace(workspace, tmp_path, capsys) -> None:
    out = tmp_path / "corpus"
    arguments = ["--subject", str(workspace / "subject.yaml"), "--measurement",
                 "marker_radius=0.007", "--quiet"]
    assert main(markers_command(workspace, out, *arguments)) == 0
    capsys.readouterr()
    assert main(markers_command(workspace, out, *arguments)) == 1
    assert "--replace" in capsys.readouterr().err
    assert main(markers_command(workspace, out, *arguments, "--replace")) == 0
    assert read_corpus(out).subject("P01").trial_ids() == ["walk"]


def test_warnings_are_printed(workspace, tmp_path, capsys) -> None:
    out = tmp_path / "corpus"
    command = markers_command(workspace, out, "--subject", str(workspace / "subject.yaml"),
                              "--measurement", "marker_radius=0.007", "--quiet")
    command[command.index("--up-axis") + 1] = "x"
    assert main(command) == 0
    assert "warning:" in capsys.readouterr().err
    assert main(["info", str(out)]) == 0
    assert "warning:" in capsys.readouterr().out
