import yaml

from smpl18 import __version__
from smpl18.cli import main

from .conftest import write_yaml


def test_validate_reports_the_error_path_and_exits_one(profile_dir, no_shared,
                                                       parameters_profile, capsys) -> None:
    parameters_profile["conventions"]["up_axis"] = "x"
    path = write_yaml(profile_dir / "bad.yaml", parameters_profile)
    assert main(["profile", "validate", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: conventions.up_axis:")


def test_validate_reports_a_missing_profile(no_shared, tmp_path, capsys) -> None:
    assert main(["profile", "validate", str(tmp_path / "none.yaml")]) == 1
    assert "no profile" in capsys.readouterr().err


def test_validate_prints_ok_with_the_referenced_files(profile_dir, no_shared, capsys) -> None:
    assert main(["profile", "validate", str(profile_dir / "skeleton.yaml")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("OK skeleton_fixture (skeleton_motion/b3d)")
    assert "correspondence:" in out and "sha256=" in out


def test_show_prints_the_resolved_profile_as_yaml(profile_dir, no_shared, capsys) -> None:
    assert main(["profile", "show", str(profile_dir / "skeleton.yaml")]) == 0
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["id"] == "skeleton_fixture"
    assert shown["profile"]["path"].endswith("skeleton.yaml")
    roles = {entry["role"]: entry for entry in shown["referenced_files"]}
    assert roles["correspondence"]["path"].endswith("correspondence.yaml")
    assert len(roles["settings[0]"]["sha256"]) == 64
    assert shown["settings_resolved"]["shape_fit"]["n_betas"] == 10


def test_version_still_works(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exit_info:
        assert exit_info.code == 0
    assert capsys.readouterr().out.strip() == f"smpl18 {__version__}"
