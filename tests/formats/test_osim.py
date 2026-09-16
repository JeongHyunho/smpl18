"""The two index maps that every downstream number depends on, plus what else the model states.

A container that stores coordinate values and joint centres as bare arrays gives them no names.
Getting either order wrong produces a plausible-looking skeleton that is silently wrong, so
the orders are derived from the model rather than hardcoded, and pinned here:
  * coordinate values = model coordinate order minus constraint-dependent coordinates
  * joint centres     = model joint order minus joints whose coordinates are all
                        constraint-dependent
"""

import pytest

from smpl18.formats import FormatError, osim

# A miniature lower limb: a floating pelvis, a hip, the coupled walker knee + patellofemoral
# pair, and a pin ankle. The function is a direct child carrying its concrete type as the tag,
# not a <function> wrapper -- the serialisation embedded models use.
MINI_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="mini">
    <BodySet>
      <objects>
        <Body name="pelvis"/>
        <Body name="femur_r"/>
        <Body name="tibia_r"/>
        <Body name="patella_r"/>
        <Body name="talus_r"/>
      </objects>
    </BodySet>
    <JointSet>
      <objects>
        <CustomJoint name="ground_pelvis">
          <socket_parent_frame>/ground</socket_parent_frame>
          <socket_child_frame>/bodyset/pelvis</socket_child_frame>
          <coordinates>
            <Coordinate name="pelvis_tilt"/>
            <Coordinate name="pelvis_tx"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>pelvis_tilt</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
            <TransformAxis name="translation1">
              <coordinates>pelvis_tx</coordinates>
              <axis>1 0 0</axis>
              <LinearFunction/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <CustomJoint name="hip_r">
          <socket_parent_frame>/bodyset/pelvis</socket_parent_frame>
          <socket_child_frame>/bodyset/femur_r</socket_child_frame>
          <coordinates>
            <Coordinate name="hip_flexion_r"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>hip_flexion_r</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
            <TransformAxis name="translation1">
              <coordinates></coordinates>
              <axis>1 0 0</axis>
              <Constant/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <CustomJoint name="walker_knee_r">
          <socket_parent_frame>/bodyset/femur_r</socket_parent_frame>
          <socket_child_frame>/bodyset/tibia_r</socket_child_frame>
          <coordinates>
            <Coordinate name="knee_angle_r"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>knee_angle_r</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
            <TransformAxis name="translation1">
              <coordinates>knee_angle_r</coordinates>
              <axis>1 0 0</axis>
              <MultiplierFunction/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <CustomJoint name="patellofemoral_r">
          <socket_parent_frame>/bodyset/femur_r</socket_parent_frame>
          <socket_child_frame>/bodyset/patella_r</socket_child_frame>
          <coordinates>
            <Coordinate name="knee_angle_r_beta"/>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1">
              <coordinates>knee_angle_r_beta</coordinates>
              <axis>0 0 1</axis>
              <LinearFunction/>
            </TransformAxis>
          </SpatialTransform>
        </CustomJoint>
        <PinJoint name="ankle_r">
          <socket_parent_frame>/bodyset/tibia_r</socket_parent_frame>
          <socket_child_frame>/bodyset/talus_r</socket_child_frame>
          <coordinates>
            <Coordinate name="ankle_angle_r"/>
          </coordinates>
        </PinJoint>
      </objects>
    </JointSet>
    <ConstraintSet>
      <objects>
        <CoordinateCouplerConstraint name="patellofemoral_knee_angle_r_con">
          <dependent_coordinate_name>knee_angle_r_beta</dependent_coordinate_name>
          <independent_coordinate_names>knee_angle_r</independent_coordinate_names>
        </CoordinateCouplerConstraint>
      </objects>
    </ConstraintSet>
    <MarkerSet>
      <objects>
        <Marker name="R_ASIS">
          <socket_parent_frame>/bodyset/pelvis</socket_parent_frame>
          <location>0.035 0.02 0.128</location>
        </Marker>
        <Marker name="R_KNEE">
          <socket_parent_frame>/bodyset/femur_r</socket_parent_frame>
          <location>0.0 -0.4 0.05</location>
        </Marker>
      </objects>
    </MarkerSet>
  </Model>
</OpenSimDocument>
"""


@pytest.fixture(scope="module")
def model():
    return osim.parse(MINI_OSIM)


def test_joint_order_follows_the_model(model):
    assert [j.name for j in model.joints] == [
        "ground_pelvis", "hip_r", "walker_knee_r", "patellofemoral_r", "ankle_r",
    ]


def test_bodies_are_read_from_the_body_set_in_file_order(model):
    assert model.name == "mini"
    assert model.bodies == ("pelvis", "femur_r", "tibia_r", "patella_r", "talus_r")


def test_bodies_resolve_through_the_socket_paths(model):
    by_name = {j.name: j for j in model.joints}
    assert by_name["ground_pelvis"].parent_body == "ground"
    assert by_name["ground_pelvis"].child_body == "pelvis"
    assert by_name["walker_knee_r"].parent_body == "femur_r"
    assert by_name["walker_knee_r"].child_body == "tibia_r"


def test_body_parents_follow_the_joints(model):
    assert model.body_parents == {
        "pelvis": "ground", "femur_r": "pelvis", "tibia_r": "femur_r",
        "patella_r": "femur_r", "talus_r": "tibia_r",
    }


def test_dependent_coordinates_come_from_the_constraint_set(model):
    assert model.dependent_coordinates == ("knee_angle_r_beta",)


def test_coordinate_value_order_drops_the_dependent_coordinate(model):
    assert model.coordinate_names == (
        "pelvis_tilt", "pelvis_tx", "hip_flexion_r",
        "knee_angle_r", "knee_angle_r_beta", "ankle_angle_r",
    )
    assert model.independent_coordinate_names == (
        "pelvis_tilt", "pelvis_tx", "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
    )


def test_joint_centre_order_drops_the_fully_dependent_joint(model):
    assert model.joint_centre_names == (
        "ground_pelvis", "hip_r", "walker_knee_r", "ankle_r",
    )


def test_a_driven_nonconstant_translation_axis_marks_the_joint_as_translating(model):
    by_name = {j.name: j for j in model.joints}
    # the walker knee slides along the femur; hip and ankle do not
    assert by_name["walker_knee_r"].translates is True
    assert by_name["hip_r"].translates is False
    assert by_name["ankle_r"].translates is False
    assert by_name["ground_pelvis"].translates is True


def test_rigidity_prediction_matches_the_topology(model):
    """Two centres hold a constant distance iff they are fixed in a common body."""
    rigid = model.rigid_centre_pairs()
    idx = {n: i for i, n in enumerate(model.joint_centre_names)}
    # hip and knee share the femur, but the knee slides along it -> not rigid
    assert not rigid[idx["hip_r"]][idx["walker_knee_r"]]
    # knee and ankle are both fixed in the tibia -> rigid
    assert rigid[idx["walker_knee_r"]][idx["ankle_r"]]
    # pelvis root and hip are both fixed in the pelvis -> rigid
    assert rigid[idx["ground_pelvis"]][idx["hip_r"]]


def test_markers_are_read_with_their_body_and_location(model):
    assert [m.name for m in model.markers] == ["R_ASIS", "R_KNEE"]
    assert model.markers[0].body == "pelvis"
    assert model.markers[0].location == pytest.approx((0.035, 0.02, 0.128))
    assert model.markers[1].body == "femur_r"


def test_a_marker_location_that_is_not_three_numbers_is_refused():
    broken = MINI_OSIM.replace(
        "<location>0.0 -0.4 0.05</location>", "<location>0.0 -0.4</location>"
    )
    with pytest.raises(FormatError, match="R_KNEE"):
        osim.parse(broken)


def test_gravity_is_read_from_the_model_not_assumed(model):
    """Which way is up is a statement the file makes; absent, it is not guessed."""
    assert model.gravity is None
    with_gravity = MINI_OSIM.replace(
        '<Model name="mini">', '<Model name="mini">\n    <gravity>0 -9.80665 0</gravity>'
    )
    assert osim.parse(with_gravity).gravity == pytest.approx((0.0, -9.80665, 0.0))


def test_malformed_gravity_is_reported_as_absent_rather_than_half_read():
    for bad in ("0 -9.80665", "", "up"):
        broken = MINI_OSIM.replace(
            '<Model name="mini">', f'<Model name="mini">\n    <gravity>{bad}</gravity>'
        )
        assert osim.parse(broken).gravity is None


def test_a_model_without_a_constraint_set_keeps_every_coordinate():
    renamed = MINI_OSIM.replace("knee_angle_r_beta", "knee_angle_r_beta_x")
    model2 = osim.parse(renamed)
    no_constraints = (
        MINI_OSIM[: MINI_OSIM.index("<ConstraintSet>")]
        + MINI_OSIM[MINI_OSIM.index("<MarkerSet>"):]
    )
    model3 = osim.parse(no_constraints)
    assert model3.dependent_coordinates == ()
    assert model3.independent_coordinate_names == model3.coordinate_names
    assert len(model3.joint_centre_names) == 5
    assert model2.dependent_coordinates == ("knee_angle_r_beta_x",)


def test_a_wrapped_function_element_is_read_the_same_as_a_bare_one():
    wrapped = MINI_OSIM.replace(
        "<coordinates>knee_angle_r</coordinates>\n              <axis>1 0 0</axis>\n"
        "              <MultiplierFunction/>",
        "<coordinates>knee_angle_r</coordinates>\n              <axis>1 0 0</axis>\n"
        "              <function><MultiplierFunction/></function>",
    )
    assert wrapped != MINI_OSIM
    by_name = {j.name: j for j in osim.parse(wrapped).joints}
    assert by_name["walker_knee_r"].translates is True


def test_read_accepts_a_path_or_the_xml_text_itself(tmp_path):
    path = tmp_path / "mini.osim"
    path.write_text(MINI_OSIM, encoding="utf-8")
    assert osim.read(path) == osim.read(str(path)) == osim.read(MINI_OSIM)


def test_text_that_is_not_xml_is_a_format_error():
    with pytest.raises(FormatError, match="well-formed"):
        osim.parse("<OpenSimDocument><Model></OpenSimDocument>")
