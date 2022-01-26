#! /usr/bin/env python

import numpy as np

import rospy
import rospkg
from tf import transformations
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3, Pose2D
from hwm_msgs.msg import LocomanipFootstepSequence2DStamped, LocomanipFootstep2D, Path2D

import mc_rtc


# Convert Pose message to Pose2D message
def pose2DFromPose(pose_msg):
    pose_2d_msg = Pose2D()
    pose_2d_msg.x = pose_msg.position.x
    pose_2d_msg.y = pose_msg.position.y
    ori = pose_msg.orientation
    pose_2d_msg.theta = transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2]
    return pose_2d_msg

# Dump the planning result of RmapPlanningLocomanip in a format that can be handled by hwm_planner
class DumpHwmPlannerLocomanipCommands(object):
    def __init__(self):
        config_path = rospkg.RosPack().get_path("differentiable_rmap") + "/config/RmapPlanningLocomanip.yaml"
        self.config = mc_rtc.Configuration(config_path)
        self.hand_traj_center = np.array(self.config("hand_traj_center", [float]))
        self.hand_traj_radius = self.config("hand_traj_radius", float)
        # rospy.loginfo("hand_traj_center: {}, hand_traj_radius: {}".format(self.hand_traj_center, self.hand_traj_radius))

        self.sub = rospy.Subscriber("current_pose_arr", PoseArray, self.callback, queue_size=1)
        self.pub = rospy.Publisher("footstep_sequence", LocomanipFootstepSequence2DStamped, queue_size=1, latch=True)

    def callback(self, msg):
        motion_len = (len(msg.poses) - 3) / 2
        # rospy.loginfo("motion_len: {}".format(motion_len))

        locomanip_footstep_seq_msg = LocomanipFootstepSequence2DStamped()
        locomanip_footstep_seq_msg.header = msg.header

        path_segment_divide_num = 10
        end_theta = 0
        for i in range(motion_len + 1):
            # Set footstep
            locomanip_footstep_msg = LocomanipFootstep2D()
            if i % 2 == 0:
                locomanip_footstep_msg.stance_foot_label = LocomanipFootstep2D.LEFT
                locomanip_footstep_msg.swing_foot_label = LocomanipFootstep2D.RIGHT
            else:
                locomanip_footstep_msg.stance_foot_label = LocomanipFootstep2D.RIGHT
                locomanip_footstep_msg.swing_foot_label = LocomanipFootstep2D.LEFT
            locomanip_footstep_msg.hand_label = LocomanipFootstep2D.LEFT
            locomanip_footstep_msg.obj_pose_idx = i * path_segment_divide_num
            locomanip_footstep_msg.regrasp_obj_pose_idx = 0
            locomanip_footstep_msg.stance_foot_pose = pose2DFromPose(msg.poses[i])
            locomanip_footstep_msg.swing_foot_pose = pose2DFromPose(msg.poses[i + 1])
            locomanip_footstep_msg.obj_pose = pose2DFromPose(msg.poses[motion_len + 2 + i])

            # Set obj_path_segment
            path_segment_msg = Path2D()
            start_theta = end_theta
            end_theta = locomanip_footstep_msg.obj_pose.theta
            if i == 0:
                path_segment_msg.poses = []
                path_segment_msg.length = 0
            else:
                for j in range(path_segment_divide_num):
                    ratio = float(j) / (path_segment_divide_num - 1)
                    theta = (1 - ratio) * start_theta + ratio * end_theta
                    obj_pose_msg = Pose2D()
                    obj_pose_msg.x = self.hand_traj_center[0] + self.hand_traj_radius * np.sin(theta)
                    obj_pose_msg.y = self.hand_traj_center[1] - self.hand_traj_radius * np.cos(theta)
                    obj_pose_msg.theta = theta
                    path_segment_msg.poses.append(obj_pose_msg)
                path_segment_msg.length = self.hand_traj_radius * np.abs(end_theta - start_theta)

            if i <= 1:
                locomanip_footstep_seq_msg.sequence.obj_path.poses += path_segment_msg.poses
            else:
                locomanip_footstep_seq_msg.sequence.obj_path.poses += path_segment_msg.poses[1:]
            locomanip_footstep_seq_msg.sequence.obj_path.length += path_segment_msg.length

            # Append
            locomanip_footstep_seq_msg.sequence.footsteps.append(locomanip_footstep_msg)
            locomanip_footstep_seq_msg.sequence.obj_path_segments.append(path_segment_msg)


        self.pub.publish(locomanip_footstep_seq_msg)

        self.sub.unregister()
        print("Run the following command to save ROS bag:")
        print("  rosbag record /footstep_sequence /marker_arr /current_left_poly_arr /current_right_poly_arr /current_cloud -l 1 -O `rospack find differentiable_rmap`/data/bridge_rmap_planning_locomanip.bag")

    def spin(self):
        # Cannot shut down this node until footstep_sequence topic is saved in rosbag
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("dump_locomanip", anonymous=True)
    dump = DumpHwmPlannerLocomanipCommands()
    dump.spin()
