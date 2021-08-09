# Copyright (c) Facebook, Inc. and its affiliates.
"""
URDF Utils
====================================
TODO
"""
import os
import torch
from urdf_parser_py.urdf import URDF


class URDFRobotModel(object):
    def __init__(self, urdf_path, device="cpu"):
        self.robot = URDF.from_xml_file(urdf_path)
        self._device = torch.device(device)

    def find_joint_of_body(self, body_name):
        for (i, joint) in enumerate(self.robot.joints):
            if joint.child == body_name:
                return i
        return -1

    def get_name_of_parent_body(self, link_name):
        jid = self.find_joint_of_body(link_name)
        joint = self.robot.joints[jid]
        return joint.parent

    def get_body_parameters_from_urdf(self, i, link):
        body_params = {}
        body_params["joint_id"] = i
        body_params["link_name"] = link.name

        if i == 0:
            rot_angles = torch.zeros(3, device=self._device)
            trans = torch.zeros(3, device=self._device)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_damping = None
            joint_axis = torch.zeros((1, 3), device=self._device)
        else:
            link_name = link.name
            jid = self.find_joint_of_body(link_name)
            joint = self.robot.joints[jid]
            joint_name = joint.name
            # find joint that is the "child" of this body according to urdf

            rot_angles = torch.tensor(
                joint.origin.rotation, dtype=torch.float32, device=self._device
            )
            trans = torch.tensor(
                joint.origin.position, dtype=torch.float32, device=self._device
            )
            joint_type = joint.type
            joint_limits = None
            joint_damping = torch.zeros(1, device=self._device)
            joint_axis = torch.zeros((1, 3), device=self._device)
            if joint_type != "fixed":
                joint_limits = {
                    "effort": joint.limit.effort,
                    "lower": joint.limit.lower,
                    "upper": joint.limit.upper,
                    "velocity": joint.limit.velocity,
                }
                try:
                    joint_damping = torch.tensor(
                        [joint.dynamics.damping],
                        dtype=torch.float32,
                        device=self._device,
                    )
                except AttributeError:
                    joint_damping = torch.zeros(1, device=self._device)
                joint_axis = torch.tensor(
                    joint.axis, dtype=torch.float32, device=self._device
                ).reshape(1, 3)

        body_params["rot_angles"] = rot_angles
        body_params["trans"] = trans
        body_params["joint_name"] = joint_name
        body_params["joint_type"] = joint_type
        body_params["joint_limits"] = joint_limits
        body_params["joint_damping"] = joint_damping
        body_params["joint_axis"] = joint_axis

        if link.inertial is not None:
            mass = torch.tensor(
                [link.inertial.mass], dtype=torch.float32, device=self._device
            )
            com = (
                torch.tensor(
                    link.inertial.origin.position,
                    dtype=torch.float32,
                    device=self._device,
                )
                .reshape((1, 3))
                .to(self._device)
            )

            inert_mat = torch.zeros((3, 3), device=self._device)
            inert_mat[0, 0] = link.inertial.inertia.ixx
            inert_mat[0, 1] = link.inertial.inertia.ixy
            inert_mat[0, 2] = link.inertial.inertia.ixz
            inert_mat[1, 0] = link.inertial.inertia.ixy
            inert_mat[1, 1] = link.inertial.inertia.iyy
            inert_mat[1, 2] = link.inertial.inertia.iyz
            inert_mat[2, 0] = link.inertial.inertia.ixz
            inert_mat[2, 1] = link.inertial.inertia.iyz
            inert_mat[2, 2] = link.inertial.inertia.izz

            inert_mat = inert_mat.unsqueeze(0)
            body_params["mass"] = mass
            body_params["com"] = com
            body_params["inertia_mat"] = inert_mat
        else:
            body_params["mass"] = torch.ones((1,), device=self._device)
            body_params["com"] = torch.zeros((1, 3), device=self._device)
            body_params["inertia_mat"] = torch.eye(3, 3, device=self._device).unsqueeze(
                0
            )
            print(
                "Warning: No dynamics information for link: {}, setting all inertial properties to 1.".format(
                    link.name
                )
            )

        return body_params
