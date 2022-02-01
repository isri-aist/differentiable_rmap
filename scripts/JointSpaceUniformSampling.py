#! /usr/bin/env python

import numpy as np

import rospy
import rospkg
import xacro

import eigen as e
import sva
import rbdyn


# Sample joint-space uniformly and dump task-space position
class JointSpaceUniformSampling(object):
    def __init__(self):
        # Setup robot from xacro
        urdf_path = rospkg.RosPack().get_path("differentiable_rmap") + "/urdf/Simple2DoFManipulator.urdf.xacro"
        urdf_doc = xacro.process_file(urdf_path)
        parser_result = rbdyn.parsers.from_urdf(urdf_doc.toprettyxml())
        self.mb = parser_result.mb
        self.mbc = parser_result.mbc
        self.limits = parser_result.limits
        rbdyn.forwardKinematics(self.mb, self.mbc)
        rbdyn.forwardVelocity(self.mb, self.mbc)

        # Setup joint limits
        self.upper_pos = []
        self.lower_pos = []
        for joint in self.mb.joints():
            if joint.name() in self.limits.upper:
                self.upper_pos += self.limits.upper[joint.name()]
            if joint.name() in self.limits.lower:
                self.lower_pos += self.limits.lower[joint.name()]
        self.upper_pos = np.array(self.upper_pos)
        self.lower_pos = np.array(self.lower_pos)

        body_name = rospy.get_param("body_name", "EEF")
        joint_pos_step = np.deg2rad(rospy.get_param("joint_pos_step", 8))

        # Sample joint positions on grids
        divide_num_list = list(((self.upper_pos - self.lower_pos) / joint_pos_step).astype(int))
        meshgrid_ratio_list = np.meshgrid(*[np.linspace(0., 1., divide_num) for divide_num in divide_num_list])
        grid_ratio_list = [meshgrid_ratio.flatten() for meshgrid_ratio in meshgrid_ratio_list]
        joint_pos_list = []
        body_pos_list = []
        for i in range(len(grid_ratio_list[0])):
            joint_pos_ratio = np.array([grid_ratio[i] for grid_ratio in grid_ratio_list])
            joint_pos = np.multiply(joint_pos_ratio, self.upper_pos) + np.multiply(1 - joint_pos_ratio, self.lower_pos)
            rbdyn.vectorToParam(joint_pos, self.mbc.q)
            rbdyn.forwardKinematics(self.mb, self.mbc)
            body_pos = self.mbc.bodyPosW[self.mb.bodyIndexByName(body_name)].translation()
            joint_pos_list.append(joint_pos)
            body_pos_list.append(body_pos)
        joint_pos_list = np.array(joint_pos_list)
        body_pos_list = np.array(body_pos_list)

        # Dump sampling result
        sample_path = "/tmp/uniform_joint_sampling.npz"
        rospy.loginfo("Dump sampling result to {}".format(sample_path))
        np.savez(sample_path, body_pos_list=body_pos_list, joint_pos_list=joint_pos_list)


if __name__ == "__main__":
    rospy.init_node("uniform_sampling", anonymous=False)
    sampling = JointSpaceUniformSampling()
