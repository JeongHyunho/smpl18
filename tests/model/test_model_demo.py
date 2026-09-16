import numpy as np

from smpl18.model import load
from smpl18.model.demo import demo_model, demo_rest_joints, write_models
from smpl18.model.select import MODEL_FILENAMES
from smpl18.skeleton.definition import SEGMENTS
from smpl18.skeleton.kinematics import rest_joints, segment_lengths


def test_the_stand_in_has_human_proportions_and_a_pelvis_off_the_origin() -> None:
    rest = demo_rest_joints()
    lengths = dict(zip((s[0] for s in SEGMENTS), segment_lengths(rest)))
    assert 0.35 < lengths["thigh_l"] < 0.45 and 0.35 < lengths["shank_l"] < 0.45
    assert 0.2 < lengths["upperarm_l"] < 0.3
    stature = rest[:, 1].max() - rest[:, 1].min()
    assert 1.2 < stature < 1.8                       # joint centres span less than the body
    assert np.linalg.norm(rest[0]) > 0.05
    assert rest[1, 0] > 0 > rest[2, 0]               # the subject's left is +x


def test_betas_change_bone_lengths_linearly() -> None:
    model = demo_model()
    base = segment_lengths(rest_joints(model, np.zeros(10)))
    longer_legs = segment_lengths(rest_joints(model, np.eye(10)[1]))
    legs = [i for i, s in enumerate(SEGMENTS) if s[0].startswith(("thigh", "shank"))]
    others = [i for i in range(len(SEGMENTS)) if i not in legs]
    assert np.all(longer_legs[legs] > base[legs])
    np.testing.assert_allclose(longer_legs[others], base[others])


def test_written_models_load_and_say_they_are_stand_ins(tmp_path) -> None:
    written = write_models(tmp_path)
    assert sorted(p.name for p in written) == sorted(MODEL_FILENAMES.values())
    loaded = load(written[0])
    assert loaded.stand_in and loaded.num_betas == 10 and loaded.sha256
    np.testing.assert_allclose(rest_joints(loaded, np.zeros(10)), demo_rest_joints())
    assert demo_model().stand_in


def test_a_real_model_file_is_not_a_stand_in(tmp_path, tiny) -> None:
    path = tmp_path / "SMPL_MALE_clean.npz"
    np.savez(path, **tiny)
    assert not load(path).stand_in
