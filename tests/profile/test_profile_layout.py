import pathlib

import pytest

from smpl24.profile.layout import Layout, LayoutError, compile_pattern


def touch(path: pathlib.Path, content: bytes = b"x") -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_compile_pattern_placeholders_and_globs() -> None:
    regex = compile_pattern("{study}/{subject}/**/{subject}_{trial}.mat")
    match = regex.fullmatch("S/P01/deep/er/P01_walk.mat")
    assert match is not None
    assert match.groupdict() == {"study": "S", "subject": "P01", "trial": "walk"}
    assert regex.fullmatch("S/P01/P02_walk.mat") is None      # repeated placeholder must agree
    assert compile_pattern("{a}/*.npz").fullmatch("x/y.npz")
    assert compile_pattern("{a}/*.npz").fullmatch("x/y/z.npz") is None


def test_empty_pattern_is_refused() -> None:
    with pytest.raises(LayoutError):
        compile_pattern("")


def test_one_file_per_trial_with_groups_exclusions_and_sorted_order(tmp_path) -> None:
    root = tmp_path / "in"
    touch(root / "CMU" / "01" / "01_02_poses.npz")
    touch(root / "CMU" / "01" / "01_01_poses.npz")
    touch(root / "CMU" / "01" / "shape.npz")
    touch(root / "KIT" / "3" / "walk_poses.npz")
    layout = Layout(subject="{study}/{subject}", trial="{study}/{subject}/{trial}_poses.npz",
                    exclude=("shape.npz",))

    subjects = layout.subjects(root)
    assert [s.id for s in subjects] == ["01", "3"]
    assert subjects[0].groups == {"study": "CMU", "subject": "01"}
    assert subjects[0].path == root / "CMU" / "01"

    trials = layout.trials(root, subjects[0])
    assert [t.id for t in trials] == ["01_01", "01_02"]
    assert trials[0].groups == {"study": "CMU", "subject": "01", "trial": "01_01"}
    assert trials[0].path == root / "CMU" / "01" / "01_01_poses.npz"
    assert not trials[0].within_container


def test_within_container_yields_one_trial_per_subject_file(tmp_path) -> None:
    root = tmp_path / "in"
    touch(root / "train" / "arms" / "StudyA" / "s01" / "s01.b3d")
    touch(root / "train" / "arms" / "StudyA" / "s02" / "s02.b3d", b"")       # zero bytes
    touch(root / "test" / "noarms" / "StudyB" / "x" / "deeper" / "s03" / "s03.b3d")
    layout = Layout(subject="{split}/{variant}/{study}/**/{subject}/*.b3d",
                    trial="within_container", skip_empty_files=True)
    subjects = layout.subjects(root)
    assert [s.id for s in subjects] == ["s03", "s01"]          # sorted by relative path
    assert subjects[1].groups == {"split": "train", "variant": "arms", "study": "StudyA",
                                  "subject": "s01"}
    trials = layout.trials(root, subjects[1])
    assert len(trials) == 1
    assert trials[0].within_container
    assert trials[0].path == subjects[1].path
    assert trials[0].id == "s01"


def test_companion_files_and_directory_trials(tmp_path) -> None:
    root = tmp_path / "in"
    (root / "_meta").mkdir(parents=True)
    (root / ".hidden" / "x").mkdir(parents=True)
    (root / "alice" / "walk" / "ik" / "models").mkdir(parents=True)
    touch(root / "alice" / "walk" / "ik" / "models" / "scaled_alice_walk.osim")
    touch(root / "alice" / "walk" / "metadata_alice_walk.json")
    (root / "alice" / "run").mkdir()
    layout = Layout(
        subject="{subject}", trial="{subject}/{trial}",
        files={"osim": "{subject}/{trial}/ik/models/scaled_{subject}_{trial}.osim",
               "metadata": "{subject}/{trial}/metadata_{subject}_{trial}.json"},
        skip_dirs_starting_with=("_", "."),
    )
    subjects = layout.subjects(root)
    assert [s.id for s in subjects] == ["alice"]
    trials = layout.trials(root, subjects[0])
    assert [t.id for t in trials] == ["run", "walk"]
    walk = trials[1]
    assert walk.files["osim"] == root / "alice" / "walk" / "ik" / "models" / "scaled_alice_walk.osim"
    assert walk.missing_files() == ()
    assert trials[0].missing_files() == ("osim", "metadata")


def test_a_subject_matched_twice_is_an_error(tmp_path) -> None:
    root = tmp_path / "in"
    touch(root / "a" / "s1" / "s1.b3d")
    touch(root / "b" / "s1" / "s1.b3d")
    layout = Layout(subject="*/{subject}/*.b3d", trial="within_container")
    with pytest.raises(LayoutError, match="matched twice"):
        layout.subjects(root)


def test_patterns_must_carry_their_identity_placeholder(tmp_path) -> None:
    root = tmp_path / "in"
    touch(root / "a" / "t.npz")
    with pytest.raises(LayoutError, match="{subject}"):
        Layout(subject="{study}", trial="{trial}").subjects(root)
    subjects = Layout(subject="{subject}", trial="{subject}/*.npz").subjects(root)
    with pytest.raises(LayoutError, match="{trial}"):
        Layout(subject="{subject}", trial="{subject}/*.npz").trials(root, subjects[0])


def test_missing_root_is_an_error(tmp_path) -> None:
    with pytest.raises(LayoutError):
        Layout(subject="{subject}", trial="within_container").subjects(tmp_path / "absent")


def test_from_mapping_round_trip() -> None:
    layout = Layout.from_mapping({"subject": "{subject}", "trial": "within_container",
                                  "exclude": ["*.bak"], "skip_empty_files": True,
                                  "subject_table": {"file": "t.csv", "format": "csv",
                                                    "id": "ID", "columns": {"gender": "G"}}})
    assert layout.trials_within_container
    assert layout.exclude == ("*.bak",)
    assert layout.subject_table["columns"] == {"gender": "G"}
