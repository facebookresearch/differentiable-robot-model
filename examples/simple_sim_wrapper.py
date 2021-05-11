import pybullet as p
import os

import diff_robot_data


class SimplePybulletWrapperPositionControl:
    def __init__(self, rel_urdf_path, controlled_joints, GUI=False):
        urdf_path = os.path.join(diff_robot_data.__path__[0], rel_urdf_path)
        self._controlled_joints = controlled_joints
        self._n_dofs = len(controlled_joints)
        if GUI:
            self._pc_id = p.connect(p.GUI)
        else:
            self._pc_id = p.connect(p.DIRECT)

        self._robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self._pc_id,
        )

        p.setGravity(0, 0, -9.81, physicsClientId=self._pc_id)

        for i in range(self._n_dofs):
            p.resetJointState(
                bodyUniqueId=self._robot_id,
                jointIndex=i,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=self._pc_id,
            )

    def reset_joint_state(self, joint_pos):
        for i in range(self._n_dofs):
            p.resetJointState(
                bodyUniqueId=self._robot_id,
                jointIndex=i,
                targetValue=joint_pos[i],
                targetVelocity=0.0,
                physicsClientId=self._pc_id,
            )

    def get_link_state(self, link_id):
        ee_state = p.getLinkState(
            bodyUniqueId=self._robot_id,
            linkIndex=link_id,
            computeForwardKinematics=1,
            physicsClientId=self._pc_id,
        )
        return ee_state[0], ee_state[1]

    def step(self, des_joint_pos):
        p.setJointMotorControlArray(
            bodyUniqueId=self._robot_id,
            jointIndices=self._controlled_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=des_joint_pos.tolist(),
            physicsClientId=self._pc_id,
        )
        for i in range(10):
            p.stepSimulation(physicsClientId=self._pc_id)
