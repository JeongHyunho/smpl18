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


# --- the kinematic description ------------------------------------------------------------------

# A floating pelvis, a knee-like joint whose slides are splines of its angle (one wrapped in a
# MultiplierFunction, one bare), a pin with oriented offset frames, a weld, and a follower whose
# coordinate a coupler constraint drives. Offset frames carry non-zero translation and
# orientation, so a reader that drops either is caught.
KINEMATIC_OSIM = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="kinematic">
    <gravity>0 -9.80665 0</gravity>
    <Ground name="ground"/>
    <BodySet>
      <objects>
        <Body name="pelvis"/>
        <Body name="thigh">
          <components>
            <PhysicalOffsetFrame name="thigh_marker_frame">
              <socket_parent>/bodyset/thigh</socket_parent>
              <translation>0 -0.1 0</translation>
              <orientation>0 0 0</orientation>
            </PhysicalOffsetFrame>
          </components>
        </Body>
        <Body name="shank"/>
        <Body name="foot"/>
        <Body name="patella"/>
      </objects>
    </BodySet>
    <JointSet>
      <objects>
        <CustomJoint name="ground_pelvis">
          <socket_parent_frame>ground_offset</socket_parent_frame>
          <socket_child_frame>pelvis_offset</socket_child_frame>
          <coordinates>
            <Coordinate name="tilt"><default_value>0.1</default_value><locked>false</locked></Coordinate>
            <Coordinate name="list"/>
            <Coordinate name="rotation"/>
            <Coordinate name="tx"><locked>true</locked></Coordinate>
            <Coordinate name="ty"><default_value>0.9</default_value></Coordinate>
            <Coordinate name="tz"><motion_type>translational</motion_type></Coordinate>
          </coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1"><coordinates>tilt</coordinates><axis>0 0 1</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
            <TransformAxis name="rotation2"><coordinates>list</coordinates><axis>1 0 0</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
            <TransformAxis name="rotation3"><coordinates>rotation</coordinates><axis>0 1 0</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
            <TransformAxis name="translation1"><coordinates>tx</coordinates><axis>1 0 0</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
            <TransformAxis name="translation2"><coordinates>ty</coordinates><axis>0 1 0</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
            <TransformAxis name="translation3"><coordinates>tz</coordinates><axis>0 0 1</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
          </SpatialTransform>
          <frames>
            <PhysicalOffsetFrame name="ground_offset">
              <socket_parent>/ground</socket_parent>
              <translation>0 0 0</translation><orientation>0 0 0</orientation>
            </PhysicalOffsetFrame>
            <PhysicalOffsetFrame name="pelvis_offset">
              <socket_parent>/bodyset/pelvis</socket_parent>
              <translation>0 0 0</translation><orientation>0 0 0</orientation>
            </PhysicalOffsetFrame>
          </frames>
        </CustomJoint>
        <CustomJoint name="knee_like">
          <socket_parent_frame>pelvis_offset</socket_parent_frame>
          <socket_child_frame>thigh_offset</socket_child_frame>
          <coordinates><Coordinate name="flex"><default_value>-0.2</default_value></Coordinate></coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1"><coordinates>flex</coordinates><axis>0 0 1</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
            <TransformAxis name="rotation2"><coordinates></coordinates><axis>1 0 0</axis>
              <Constant><value>0</value></Constant></TransformAxis>
            <TransformAxis name="rotation3"><coordinates></coordinates><axis>0 1 0</axis>
              <Constant><value>0</value></Constant></TransformAxis>
            <TransformAxis name="translation1"><coordinates>flex</coordinates><axis>1 0 0</axis>
              <MultiplierFunction>
                <function><SimmSpline><x>-2 -1 0 1</x><y>0.01 0.02 0.0 -0.01</y></SimmSpline></function>
                <scale>1.05</scale>
              </MultiplierFunction>
            </TransformAxis>
            <TransformAxis name="translation2"><coordinates>flex</coordinates><axis>0 1 0</axis>
              <function><SimmSpline><x>-2 -1 0 1</x><y>-0.4 -0.39 -0.4 -0.41</y></SimmSpline></function>
            </TransformAxis>
            <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis>
              <MultiplierFunction><function><Constant><value>0.02</value></Constant></function><scale>1.1</scale></MultiplierFunction>
            </TransformAxis>
          </SpatialTransform>
          <frames>
            <PhysicalOffsetFrame name="pelvis_offset">
              <socket_parent>/bodyset/pelvis</socket_parent>
              <translation>-0.07 -0.09 0.08</translation><orientation>0.1 -0.2 0.3</orientation>
            </PhysicalOffsetFrame>
            <PhysicalOffsetFrame name="thigh_offset">
              <socket_parent>/bodyset/thigh</socket_parent>
              <translation>0 0.02 0</translation><orientation>0 0 -0.05</orientation>
            </PhysicalOffsetFrame>
          </frames>
        </CustomJoint>
        <PinJoint name="pin">
          <socket_parent_frame>/jointset/pin/thigh_offset</socket_parent_frame>
          <socket_child_frame>/bodyset/shank</socket_child_frame>
          <coordinates><Coordinate name="pin_angle"/></coordinates>
          <frames>
            <PhysicalOffsetFrame name="thigh_offset">
              <socket_parent>../../../bodyset/thigh</socket_parent>
              <translation>0 -0.4 0</translation><orientation>0.5 0 0</orientation>
            </PhysicalOffsetFrame>
          </frames>
        </PinJoint>
        <WeldJoint name="weld">
          <socket_parent_frame>/bodyset/shank</socket_parent_frame>
          <socket_child_frame>/bodyset/foot</socket_child_frame>
        </WeldJoint>
        <CustomJoint name="follower">
          <socket_parent_frame>/bodyset/thigh/thigh_marker_frame</socket_parent_frame>
          <socket_child_frame>/bodyset/patella</socket_child_frame>
          <coordinates><Coordinate name="flex_beta"/></coordinates>
          <SpatialTransform>
            <TransformAxis name="rotation1"><coordinates>flex_beta</coordinates><axis>0 0 1</axis>
              <LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
          </SpatialTransform>
        </CustomJoint>
      </objects>
    </JointSet>
    <ConstraintSet>
      <objects>
        <CoordinateCouplerConstraint name="follow">
          <coupled_coordinates_function>
            <PolynomialFunction><coefficients>0.5 0 0.1</coefficients></PolynomialFunction>
          </coupled_coordinates_function>
          <independent_coordinate_names>flex</independent_coordinate_names>
          <dependent_coordinate_name>flex_beta</dependent_coordinate_name>
          <scale_factor>2</scale_factor>
        </CoordinateCouplerConstraint>
      </objects>
    </ConstraintSet>
  </Model>
</OpenSimDocument>
"""


@pytest.fixture(scope="module")
def kinematic():
    return osim.parse(KINEMATIC_OSIM)


def _joint(model, name):
    return next(j for j in model.joints if j.name == name)


def test_each_joint_keeps_its_type_tag(kinematic):
    assert [j.kind for j in kinematic.joints] == [
        "CustomJoint", "CustomJoint", "PinJoint", "WeldJoint", "CustomJoint",
    ]


def test_offset_frames_are_read_with_their_body_translation_and_orientation(kinematic):
    knee = _joint(kinematic, "knee_like")
    assert knee.parent_frame == osim.OsimFrame(
        "pelvis", (-0.07, -0.09, 0.08), (0.1, -0.2, 0.3), name="pelvis_offset"
    )
    assert knee.child_frame == osim.OsimFrame(
        "thigh", (0.0, 0.02, 0.0), (0.0, 0.0, -0.05), name="thigh_offset"
    )
    assert knee.parent_socket == "pelvis_offset"


def test_a_socket_naming_a_body_is_that_body_with_no_offset(kinematic):
    weld = _joint(kinematic, "weld")
    assert weld.parent_frame == osim.OsimFrame("shank", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert weld.child_frame == osim.OsimFrame("foot", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    ground = _joint(kinematic, "ground_pelvis").parent_frame
    assert (ground.body, ground.translation, ground.orientation) == (
        "ground", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_absolute_and_relative_component_paths_resolve(kinematic):
    """``/jointset/<joint>/<frame>``, ``../`` steps and a frame a body owns all resolve."""
    pin = _joint(kinematic, "pin")
    assert pin.parent_frame == osim.OsimFrame(
        "thigh", (0.0, -0.4, 0.0), (0.5, 0.0, 0.0), name="thigh_offset"
    )
    follower = _joint(kinematic, "follower")
    assert follower.parent_frame == osim.OsimFrame(
        "thigh", (0.0, -0.1, 0.0), (0.0, 0.0, 0.0), name="thigh_marker_frame"
    )


def test_a_frame_that_does_not_resolve_is_absent_and_its_socket_kept():
    unresolved = KINEMATIC_OSIM.replace(
        "<socket_parent_frame>/bodyset/shank</socket_parent_frame>",
        "<socket_parent_frame>no_such_frame</socket_parent_frame>",
    )
    weld = _joint(osim.parse(unresolved), "weld")
    assert weld.parent_frame is None
    assert weld.parent_socket == "no_such_frame"
    # the body-level answer the reader always gave is unchanged
    assert weld.parent_body is None


def test_an_offset_frame_on_another_offset_frame_is_not_flattened():
    chained = KINEMATIC_OSIM.replace(
        "<socket_parent>../../../bodyset/thigh</socket_parent>",
        "<socket_parent>/bodyset/thigh/thigh_marker_frame</socket_parent>",
    )
    assert _joint(osim.parse(chained), "pin").parent_frame is None


def test_transform_axes_are_read_in_file_order_with_their_functions(kinematic):
    axes = _joint(kinematic, "ground_pelvis").transform_axes
    assert [a.name for a in axes] == [
        "rotation1", "rotation2", "rotation3", "translation1", "translation2", "translation3",
    ]
    assert axes[1] == osim.OsimTransformAxis(
        "rotation2", ("list",), (1.0, 0.0, 0.0),
        osim.OsimFunction("LinearFunction", {"coefficients": (1.0, 0.0)}),
    )


def test_bare_wrapped_and_nested_functions_are_all_read(kinematic):
    axes = {a.name: a for a in _joint(kinematic, "knee_like").transform_axes}
    multiplier = axes["translation1"].function
    assert multiplier.kind == "MultiplierFunction"
    assert multiplier.values["scale"] == pytest.approx(1.05)
    assert multiplier.values["function"] == osim.OsimFunction(
        "SimmSpline", {"x": (-2.0, -1.0, 0.0, 1.0), "y": (0.01, 0.02, 0.0, -0.01)}
    )
    wrapped = axes["translation2"].function
    assert wrapped.kind == "SimmSpline"
    assert wrapped.values["y"] == (-0.4, -0.39, -0.4, -0.41)
    assert axes["rotation2"].coordinates == ()
    assert axes["rotation2"].function == osim.OsimFunction("Constant", {"value": 0.0})
    constant = axes["translation3"].function.values["function"]
    assert constant == osim.OsimFunction("Constant", {"value": 0.02})


def test_an_omitted_property_stays_omitted():
    """Defaults are OpenSim's to give, and the skeleton model gives them; the reader does not."""
    model = osim.parse(MINI_OSIM)
    axes = {a.name: a for a in model.joints[0].transform_axes}
    assert axes["rotation1"].function == osim.OsimFunction("LinearFunction", {})
    detail = model.joints[0].coordinate_details[0]
    assert detail == osim.OsimCoordinate("pelvis_tilt", None, False, None)
    assert model.couplers[0].function is None
    assert model.couplers[0].scale_factor is None
    no_axis = KINEMATIC_OSIM.replace(
        "<coordinates>list</coordinates><axis>1 0 0</axis>", "<coordinates>list</coordinates>"
    )
    assert _joint(osim.parse(no_axis), "ground_pelvis").transform_axes[1].axis is None


def test_function_values_are_read_only_and_joints_stay_hashable(kinematic):
    function = _joint(kinematic, "ground_pelvis").transform_axes[0].function
    with pytest.raises(TypeError):
        function.values["coefficients"] = (2.0, 0.0)
    assert hash(kinematic) == hash(osim.parse(KINEMATIC_OSIM))


def test_coordinate_details_follow_the_coordinate_listing(kinematic):
    pelvis = _joint(kinematic, "ground_pelvis")
    assert tuple(d.name for d in pelvis.coordinate_details) == pelvis.coordinates
    by_name = {d.name: d for d in pelvis.coordinate_details}
    assert by_name["tilt"] == osim.OsimCoordinate("tilt", 0.1, False, None)
    assert by_name["tx"].locked is True
    assert by_name["ty"].default_value == pytest.approx(0.9)
    assert by_name["tz"].motion_type == "translational"
    assert _joint(kinematic, "weld").coordinate_details == ()


def test_coupler_constraints_are_read_whole(kinematic):
    assert kinematic.couplers == (
        osim.OsimCouplerConstraint(
            name="follow",
            independent=("flex",),
            dependent="flex_beta",
            function=osim.OsimFunction("PolynomialFunction", {"coefficients": (0.5, 0.0, 0.1)}),
            scale_factor=2.0,
        ),
    )
    assert kinematic.dependent_coordinates == ("flex_beta",)


def test_knot_functions_and_gcv_splines_keep_their_own_properties():
    gcv = KINEMATIC_OSIM.replace(
        "<function><SimmSpline><x>-2 -1 0 1</x><y>-0.4 -0.39 -0.4 -0.41</y></SimmSpline></function>",
        "<GCVSpline><half_order>2</half_order><error_variance>0</error_variance>"
        "<x>-2 -1 0 1</x><y>1 2 3 4</y></GCVSpline>",
    ).replace(
        "<PolynomialFunction><coefficients>0.5 0 0.1</coefficients></PolynomialFunction>",
        "<PiecewiseLinearFunction><x>0 1</x><y>0 2</y></PiecewiseLinearFunction>",
    )
    model = osim.parse(gcv)
    axes = {a.name: a for a in _joint(model, "knee_like").transform_axes}
    assert axes["translation2"].function == osim.OsimFunction(
        "GCVSpline",
        {"x": (-2.0, -1.0, 0.0, 1.0), "y": (1.0, 2.0, 3.0, 4.0), "half_order": 2,
         "error_variance": 0.0},
    )
    assert model.couplers[0].function == osim.OsimFunction(
        "PiecewiseLinearFunction", {"x": (0.0, 1.0), "y": (0.0, 2.0)}
    )


def test_an_unknown_function_type_keeps_its_tag():
    odd = KINEMATIC_OSIM.replace(
        "<PolynomialFunction><coefficients>0.5 0 0.1</coefficients></PolynomialFunction>",
        "<Sine><amplitude>1</amplitude></Sine>",
    )
    assert osim.parse(odd).couplers[0].function == osim.OsimFunction("Sine", {})


@pytest.mark.parametrize(
    ("original", "broken", "message"),
    [
        ("<y>-0.4 -0.39 -0.4 -0.41</y>", "<y>-0.4 -0.39 x -0.41</y>",
         "knee_like.*translation2.*y"),
        ("<translation>-0.07 -0.09 0.08</translation>", "<translation>-0.07 -0.09</translation>",
         "pelvis_offset.*three numbers"),
        ("<default_value>0.9</default_value>", "<default_value>high</default_value>",
         "coordinate 'ty'"),
        ("<scale_factor>2</scale_factor>", "<scale_factor>2 3</scale_factor>",
         "follow.*one number"),
        ("<axis>0 0 1</axis>", "<axis>0 0 one</axis>", "rotation1.*axis"),
        ("<scale>1.05</scale>", "<scale>big</scale>", "MultiplierFunction scale"),
        ("<half_order>2</half_order>", "<half_order>2.5</half_order>", "half_order"),
    ],
)
def test_malformed_numbers_are_a_format_error_that_names_them(original, broken, message):
    text = KINEMATIC_OSIM.replace(
        "<Constant><value>0.02</value></Constant>",
        "<GCVSpline><half_order>2</half_order><x>0 1</x><y>0 1</y></GCVSpline>",
    )
    assert original in text
    with pytest.raises(FormatError, match=message):
        osim.parse(text.replace(original, broken, 1))


def test_the_new_fields_leave_the_derived_orders_alone(kinematic):
    assert kinematic.independent_coordinate_names == (
        "tilt", "list", "rotation", "tx", "ty", "tz", "flex", "pin_angle",
    )
    assert kinematic.gravity == pytest.approx((0.0, -9.80665, 0.0))
    assert kinematic.body_parents["patella"] == "thigh"
