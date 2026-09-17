import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from smpl18 import __version__, cli, synthetic
from smpl18.blender import launch
from smpl18.blender.plan import read_plan
from smpl18.cli import main
from smpl18.corpus import read_corpus
from smpl18.formats import trc
from smpl18.model import load
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

# --- out of the corpus again ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered(workspace, tmp_path_factory):
    """A corpus with a surfaced stand-in body, converted from SMPL parameters (which is quick)."""
    root = tmp_path_factory.mktemp("scene")
    write_models(root / "models", with_mesh=True)
    model = demo_model(with_mesh=True)
    local, trans = synthetic.walking_motion(12, 50.0, seed=6)
    np.savez(root / "walk.npz", poses=synthetic.rotations_to_axis_angle(local), trans=trans,
             betas=np.zeros(10), mocap_framerate=50.0)
    np.savez(root / "stand.npz", poses=synthetic.rotations_to_axis_angle(local[:6]),
             trans=trans[:6], betas=np.zeros(10), mocap_framerate=50.0)
    assert main(["convert", "smpl", "--input", str(root / "walk.npz"), str(root / "stand.npz"),
                 "--up-axis", "y", "--settings", str(workspace / "quick.yaml"),
                 "--models", str(root / "models"), "--out", str(root / "corpus"),
                 "--subject-id", "S01", "--gender", "neutral", "--quiet"]) == 0
    del model
    return root


def test_demo_models_can_carry_a_surface(tmp_path, capsys) -> None:
    assert main(["demo-models", "--out", str(tmp_path), "--with-mesh"]) == 0
    assert "blocky surface" in capsys.readouterr().out
    body = load(tmp_path / "SMPL_NEUTRAL_clean.npz")
    assert body.has_mesh and body.stand_in and body.faces.shape[1] == 3
    assert load(tmp_path / "SMPL_MALE_clean.npz").weights is not None


def test_export_smpl_writes_one_file_per_trial(rendered, tmp_path, capsys) -> None:
    out = tmp_path / "smpl"
    assert main(["export-smpl", "--corpus", str(rendered / "corpus"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "S01_walk.npz" in printed and "S01_stand.npz" in printed
    assert "18/24 joints measured" in printed
    with np.load(out / "S01_walk.npz", allow_pickle=False) as data:
        assert data["poses"].shape == (12, 72)
        assert float(data["mocap_framerate"]) == 50.0


def test_export_smpl_writes_one_named_file(rendered, tmp_path) -> None:
    target = tmp_path / "one.npz"
    assert main(["export-smpl", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--trial", "walk", "--out", str(target), "--poses", "grouped"]) == 0
    with np.load(target, allow_pickle=False) as data:
        assert data["poses"].shape == (12, 24, 3)


def test_export_smpl_will_not_put_several_trials_in_one_file(rendered, tmp_path, capsys) -> None:
    code = main(["export-smpl", "--corpus", str(rendered / "corpus"), "--out",
                 str(tmp_path / "one.npz")])
    assert code == 2
    assert "give a directory" in capsys.readouterr().err


def test_blender_writes_a_plan_and_prints_how_to_run_it(rendered, tmp_path, capsys) -> None:
    out = tmp_path / "scene"
    assert main(["blender", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--trial", "walk", "--models", str(rendered / "models"), "--out", str(out),
                 "--blend"]) == 0
    printed = capsys.readouterr()
    assert "12 frames" in printed.out and "no --blender given" in printed.out
    assert "scene.py" in printed.out and "--plan" in printed.out
    assert "mannequin of blocks" in printed.err
    plan = read_plan(out / "S01_walk.plan.npz")
    assert plan.frames == 12 and plan.about["subject"] == "S01"
    assert plan.about["render_settings"][0]["path"] == "default.yaml"
    settings = json.loads((out / "S01_walk.plan.render.json").read_text(encoding="utf-8"))
    assert settings["render"]["engine"] == "cycles"


def test_blender_takes_a_slice_and_the_correctives(rendered, tmp_path, capsys) -> None:
    out = tmp_path / "scene"
    assert main(["blender", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--trial", "walk", "--models", str(rendered / "models"), "--out", str(out),
                 "--frames", "2:10:2", "--correctives", "--render"]) == 0
    assert "207 pose shape keys" in capsys.readouterr().out
    plan = read_plan(out / "S01_walk.plan.npz")
    assert plan.frames == 4 and plan.has_correctives


def test_blender_says_which_trial_it_chose(rendered, tmp_path, capsys) -> None:
    assert main(["blender", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--models", str(rendered / "models"), "--out", str(tmp_path / "scene"),
                 "--blend"]) == 0
    assert "has 2 trials; building stand" in capsys.readouterr().err


def test_blender_needs_a_model_with_a_surface(workspace, rendered, tmp_path, capsys) -> None:
    code = main(["blender", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--trial", "walk", "--models", str(workspace / "models"),
                 "--out", str(tmp_path / "scene"), "--blend"])
    assert code == 2
    assert "extract-model --with-mesh" in capsys.readouterr().err


def test_a_frame_range_must_be_a_range(rendered, tmp_path, capsys) -> None:
    code = main(["blender", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--trial", "walk", "--models", str(rendered / "models"),
                 "--out", str(tmp_path / "scene"), "--frames", "10", "--blend"])
    assert code == 2
    assert "START:STOP" in capsys.readouterr().err


def test_blender_is_run_on_the_plan_when_one_is_named(rendered, tmp_path, monkeypatch,
                                                     capsys) -> None:
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command

        class Done:
            returncode = 0
            stdout = ("Blender 4.2\nbuilt S01/walk: 12 frames, 224 vertices, 24 bones\n"
                      "self-check: 3 frames, worst vertex 0.0001 mm from this package's own "
                      "skinning (tolerance 0.2000 mm)\nsaved scene.blend\n")
            stderr = ""
        return Done()

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    out = tmp_path / "scene"
    assert main(["blender", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--trial", "walk", "--models", str(rendered / "models"), "--out", str(out),
                 "--blender", "blender", "--blend"]) == 0
    printed = capsys.readouterr().out
    assert "self-check: 3 frames" in printed and "saved scene.blend" in printed
    assert "Blender 4.2" not in printed                      # only what the script reported
    command = seen["command"]
    assert command[:3] == ["blender", "--background", "--factory-startup"]
    after = command[command.index("--") + 1:]
    assert after[after.index("--plan") + 1] == str(out / "S01_walk.plan.npz")
    assert "--render" not in after


def test_blender_asks_what_to_build(rendered, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(launch.subprocess, "run",
                        lambda command, **kwargs: pytest.fail("Blender should not have run"))
    code = main(["blender", "--corpus", str(rendered / "corpus"), "--subject", "S01",
                 "--trial", "walk", "--models", str(rendered / "models"),
                 "--out", str(tmp_path / "scene"), "--blender", "blender"])
    assert code == 2
    assert "--blend, --render, or both" in capsys.readouterr().err
