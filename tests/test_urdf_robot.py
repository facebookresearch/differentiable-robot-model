from unittest import mock

import pytest
import torch

from differentiable_robot_model.robot_model import DifferentiableRobotModel

# this file contains no dummy link
urdf_clean_file = r"""
<robot name="test_robot">
    <link name="link_base"/>
    <joint name="joint_1" type="continuous">
        <parent link="link_base"/>
        <child link="link_1"/>
        <axis xyz="0 0 1"/>
        <limit effort="40" lower="-6.28" upper="6.28" velocity="0.628"/>
        <origin rpy="0 3.14 0" xyz="0 0 0.15675"/>
    </joint>
    <link name="link_1">
        <inertial>
            <mass value="1.0"/>
            <origin xyz="0 0 -0.06"/>
            <inertia ixx="0.00034" ixy="0" ixz="0" iyy="0.00034" iyz="0" izz="0.00058"/>
        </inertial>
    </link>
</robot>
"""

# this file contains one dummy link at the beginning
urdf_with_dummy_link = r"""
<robot name="test_robot">
    <link name="link_base"/>
    <link name="dummy_link"/>
    <joint name="joint_1" type="continuous">
        <parent link="link_base"/>
        <child link="link_1"/>
        <axis xyz="0 0 1"/>
        <limit effort="40" lower="-6.28" upper="6.28" velocity="0.628"/>
        <origin rpy="0 3.14 0" xyz="0 0 0.15675"/>
    </joint>
    <link name="link_1">
        <inertial>
            <mass value="1.0"/>
            <origin xyz="0 0 -0.06"/>
            <inertia ixx="0.00034" ixy="0" ixz="0" iyy="0.00034" iyz="0" izz="0.00058"/>
        </inertial>
    </link>
</robot>
"""

# this file contains one dummy link (the end_effector frame) between arm and finger
urdf_link1_eff_finger1 = r"""
<robot name="test_robot">
    <link name="link_base">
        <inertial>
            <mass value="0.5"/>
            <origin rpy="0 0 0" xyz="0 0 0.1255"/>
            <inertia ixx="0.0009" ixy="0" ixz="0" iyy="0.0009" iyz="0" izz="0.0003"/>
        </inertial>
    </link>
    <joint name="joint_1" type="continuous">
        <parent link="link_base"/>
        <child link="link_1"/>
        <axis xyz="0 0 1"/>
        <limit effort="40" lower="-6.28" upper="6.28" velocity="0.628"/>
        <origin rpy="0 3.14 0" xyz="0 0 0.15675"/>
    </joint>
    <link name="link_1">
        <inertial>
            <mass value="1.0"/>
            <origin xyz="0 0 -0.06"/>
            <inertia ixx="0.00034" ixy="0" ixz="0" iyy="0.00034" iyz="0" izz="0.00058"/>
        </inertial>
    </link>
    <link name="end_effector"/>
    <joint name="joint_end_effector" type="fixed">
        <parent link="link_1"/>
        <child link="end_effector"/>
        <axis xyz="0 0 0"/>
        <limit effort="2000" lower="0" upper="0" velocity="1"/>
        <origin rpy="3.14 0 1.57" xyz="0 0 -0.1600"/>
    </joint>
    <link name="link_finger_1">
        <inertial>
            <mass value="0.01"/>
            <origin xyz="0.022 0 0"/>
            <inertia ixx="7.89e-07" ixy="0" ixz="0" iyy="7.89e-07" iyz="0" izz="8e-08"/>
        </inertial>
    </link>
    <joint name="joint_finger_1" type="revolute">
        <parent link="link_1"/>
        <child link="link_finger_1"/>
        <axis xyz="0 0 1"/>
        <origin rpy="-1.57 .64 1.35" xyz="0.0027 0.031 -0.114"/>
        <limit effort="2" lower="0" upper="1.51" velocity="1"/>
    </joint>
</robot>
"""


def build_robot_model(links_blacklist, file_content):
    with mock.patch("builtins.open", mock.mock_open(read_data=file_content)):
        robot_model = DifferentiableRobotModel(
            "mocked_file.urdf", "test", links_blacklist=links_blacklist,
        )
    return robot_model


@pytest.fixture
def robot_model(request):
    return build_robot_model(*request.param)


@pytest.mark.filterwarnings("ignore:Intermediate link")
@pytest.mark.parametrize(
    "robot_model",
    [(set(), urdf_link1_eff_finger1), ({"foobar"}, urdf_link1_eff_finger1)],
    indirect=["robot_model"],
)
def test_exception(robot_model):
    with pytest.raises(TypeError):
        rand_qs = torch.rand(1, 2)
        # TypeError: unsupported operand type(s) for *: 'NoneType' and 'NoneType
        robot_model.compute_forward_dynamics(q=rand_qs, qd=rand_qs, f=rand_qs)


@pytest.mark.parametrize(
    "links_blacklist, file_content",
    [
        (set(), urdf_link1_eff_finger1),
        ({"foobar"}, urdf_link1_eff_finger1),
        (set(), urdf_with_dummy_link),
        ({"foobar"}, urdf_with_dummy_link),
    ],
)
def test_emit_warnings(links_blacklist, file_content):
    # assert a warning is emitted
    with pytest.warns(UserWarning):
        build_robot_model(links_blacklist, file_content)


@pytest.mark.parametrize(
    "robot_model",
    [
        ({"end_effector"}, urdf_link1_eff_finger1),
        ({"dummy_link"}, urdf_with_dummy_link),
        (set(), urdf_clean_file),
    ],
    indirect=["robot_model"],
)
def test_no_exception_with_correct_blacklist(robot_model):
    rand_qs = torch.rand(1, len(robot_model.get_link_names()) - 1)
    robot_model.compute_forward_dynamics(q=rand_qs, qd=rand_qs, f=rand_qs)


@pytest.mark.parametrize(
    "links_blacklist, file_content",
    [
        ({"end_effector"}, urdf_link1_eff_finger1),
        ({"dummy_link"}, urdf_with_dummy_link),
        (set(), urdf_clean_file),
    ],
)
def test_no_emit_warnings(links_blacklist, file_content):
    with pytest.warns(None) as record:
        # assert no warning after using correct links blacklist
        build_robot_model(links_blacklist, file_content)
        assert (
            len(record) == 0
        ), f"Warnings should not be emitted. {[r.message for r in record]}"
