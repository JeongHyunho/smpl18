import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smpl18.mesh import (
    NUM_POSE_FEATURES,
    MeshError,
    lbs_transforms,
    pose_features,
    pose_offsets,
    posed_vertices,
    skin,
)
from smpl18.model.demo import demo_model
from smpl18.model.load import Model
from smpl18.skeleton.definition import CHILDREN, NUM_JOINTS
from smpl18.skeleton.kinematics import fk_batch, rest_joints, shaped_vertices


@pytest.fixture(scope="module")
def body():
    return demo_model(with_mesh=True)


def still(frames: int) -> np.ndarray:
    return np.tile(np.eye(3), (frames, NUM_JOINTS, 1, 1))


def turned(joint: int, angles) -> np.ndarray:
    """Local rotations that turn one joint through ``angles`` about x."""
    angles = np.atleast_1d(angles)
    local = still(angles.size)
    local[:, joint] = Rotation.from_rotvec(np.stack([angles, 0 * angles, 0 * angles], axis=1)).as_matrix()
    return local


def descendants(joint: int) -> set[int]:
    below = set()
    stack = [joint]
    while stack:
        node = stack.pop()
        for child in CHILDREN[node]:
            below.add(child)
            stack.append(child)
    return below


def test_a_model_without_the_mesh_says_what_to_do(body) -> None:
    with pytest.raises(MeshError, match="extract-model --with-mesh"):
        posed_vertices(demo_model(), np.zeros(10), still(1), np.zeros((1, 3)))


def test_the_rest_pose_is_the_shaped_template(body) -> None:
    betas = np.array([0.5, -0.3, 0.2, 0, 0, 0, 0, 0, 0, 0])
    posed = posed_vertices(body, betas, still(1), np.zeros((1, 3)))
    np.testing.assert_allclose(posed.vertices[0], shaped_vertices(body, betas), atol=1e-12)
    np.testing.assert_allclose(posed.joints[0], rest_joints(body, betas), atol=1e-12)


def test_a_pose_moves_what_is_below_the_joint_and_nothing_else(body) -> None:
    local = turned(4, [0.0, 0.9])                    # the left knee
    posed = posed_vertices(body, np.zeros(10), local, np.zeros((2, 3)))
    moved = np.linalg.norm(posed.vertices[1] - posed.vertices[0], axis=1) > 1e-9
    below = descendants(4) | {4}
    bound_below = body.weights[:, sorted(below)].sum(axis=1) > 0
    assert np.array_equal(moved, bound_below)


def test_the_root_carries_the_whole_body_rigidly(body) -> None:
    betas = np.zeros(10)
    turn = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    local = still(1)
    local[:, 0] = turn
    shift = np.array([[0.4, -0.2, 0.1]])
    posed = posed_vertices(body, betas, local, shift)
    rest = rest_joints(body, betas)
    expected = (shaped_vertices(body, betas) - rest[0]) @ turn.T + rest[0] + shift
    np.testing.assert_allclose(posed.vertices[0], expected, atol=1e-12)


def test_pose_features_are_the_rotations_less_the_identity(body) -> None:
    local = turned(18, [0.7])
    features = pose_features(local)
    assert features.shape == (1, NUM_POSE_FEATURES) == (1, 207)
    expected = (local[0, 18] - np.eye(3)).ravel()
    np.testing.assert_allclose(features[0, (18 - 1) * 9:(18 - 1) * 9 + 9], expected)
    np.testing.assert_allclose(pose_features(still(3)), 0.0)
    with pytest.raises(MeshError, match=r"\(T, 24, 3, 3\)"):
        pose_features(np.zeros((2, 24, 3)))


def test_pose_offsets_are_a_linear_function_of_the_features() -> None:
    rng = np.random.default_rng(3)
    vertices = rng.normal(size=(9, 3))
    model = Model(
        v_template=vertices,
        shapedirs=rng.normal(size=(9, 3, 2)) * 0.01,
        J_regressor=np.abs(rng.normal(size=(NUM_JOINTS, 9))),
        kintree_parents=np.array(demo_model().kintree_parents),
        weights=np.abs(rng.normal(size=(9, NUM_JOINTS))),
        posedirs=rng.normal(size=(9, 3, NUM_POSE_FEATURES)) * 0.01,
        faces=np.array([[0, 1, 2]]),
    )
    local = turned(5, [0.4])
    offsets = pose_offsets(model, local)
    expected = np.einsum("vij,j->vi", model.posedirs, pose_features(local)[0])
    np.testing.assert_allclose(offsets[0], expected, atol=1e-14)
    np.testing.assert_allclose(pose_offsets(model, still(1)), 0.0, atol=1e-14)


def test_leaving_the_pose_offsets_out_changes_only_them(body) -> None:
    rng = np.random.default_rng(7)
    posedirs = rng.normal(size=(body.num_vertices, 3, NUM_POSE_FEATURES)) * 0.001
    with_offsets = Model(
        v_template=body.v_template, shapedirs=body.shapedirs, J_regressor=body.J_regressor,
        kintree_parents=body.kintree_parents, weights=body.weights, posedirs=posedirs,
        faces=body.faces,
    )
    local, trans = turned(4, [0.8]), np.zeros((1, 3))
    on = posed_vertices(with_offsets, np.zeros(10), local, trans)
    off = posed_vertices(with_offsets, np.zeros(10), local, trans, with_pose_offsets=False)
    # Skinning is affine in the vertex, so the two differ by the offsets, turned by each vertex's
    # own blended rotation and nothing more.
    blended = np.einsum("vj,tjab->tvab", body.weights, on.transforms)
    expected = np.einsum("tvij,tvj->tvi", blended[:, :, :3, :3],
                         pose_offsets(with_offsets, local))
    assert np.abs(expected).max() > 1e-6
    np.testing.assert_allclose(on.vertices - off.vertices, expected, atol=1e-12)


def test_the_transforms_carry_the_rest_joints_onto_the_posed_ones(body) -> None:
    betas = np.array([0.2, 0.1, -0.4, 0, 0, 0, 0, 0, 0, 0])
    local = turned(5, np.linspace(0.0, 1.1, 5))
    trans = np.linspace(0.0, 0.5, 5)[:, None] * np.array([1.0, 0.0, 0.3])
    rest = rest_joints(body, betas)
    positions, world = fk_batch(rest, local, trans)
    transforms = lbs_transforms(rest, positions, world)
    homogeneous = np.concatenate([rest, np.ones((NUM_JOINTS, 1))], axis=1)
    carried = np.einsum("tjab,jb->tja", transforms, homogeneous)[:, :, :3]
    np.testing.assert_allclose(carried, positions, atol=1e-12)
    assert transforms.shape == (5, NUM_JOINTS, 4, 4)
    bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), transforms.shape[:2] + (4,))
    np.testing.assert_allclose(transforms[:, :, 3], bottom)


def test_skinning_by_the_transforms_is_what_posing_does(body) -> None:
    betas = np.zeros(10)
    local = turned(18, [0.0, 0.6, 1.2])
    trans = np.zeros((3, 3))
    posed = posed_vertices(body, betas, local, trans, with_pose_offsets=False)
    by_hand = skin(shaped_vertices(body, betas), body.weights, posed.transforms)
    np.testing.assert_allclose(posed.vertices, by_hand, atol=1e-12)


def test_shapes_that_do_not_fit_are_refused(body) -> None:
    posed = posed_vertices(body, np.zeros(10), still(2), np.zeros((2, 3)))
    with pytest.raises(MeshError, match="weights"):
        skin(body.v_template, np.zeros((3, NUM_JOINTS)), posed.transforms)
    with pytest.raises(MeshError, match="vertices"):
        skin(np.zeros((5, 224, 3)), body.weights, posed.transforms)
    with pytest.raises(MeshError, match="rest joints"):
        lbs_transforms(np.zeros((3, 3)), posed.joints, np.zeros((2, NUM_JOINTS, 3, 3)))
    with pytest.raises(MeshError, match="world rotations"):
        lbs_transforms(posed.rest, posed.joints, np.zeros((2, NUM_JOINTS, 3)))
