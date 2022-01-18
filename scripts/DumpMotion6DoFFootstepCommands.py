#! /usr/bin/env python

import numpy as np
import rospy
from tf import transformations

from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3


# Dump the planning result of RmapPlanningFootstep in a format that can be handled by Motion6DoF controller
class DumpMotion6DoFFootstepCommands(object):
    LEFT = 1
    RIGHT = -1

    limbEnd = {
        LEFT: "LeftFoot",
        RIGHT: "RightFoot"
    }

    constraintName = {
        LEFT: "LeftFootContact",
        RIGHT: "RightFootContact"
    }

    verticesName = {
        LEFT: "LeftFoot",
        RIGHT: "RightFoot"
    }

    def __init__(self):
        self.exit_flag = False
        self.sub = rospy.Subscriber("current_pose_arr", PoseArray, self.callback, queue_size=1)

    def opposite(self, footSide):
        return -1 * footSide

    def dumpOneCommand(self, fileObj, footSide, footPose, startTime, endTime):
        dumpStr = """        - limbEnd: {0}
          type: Add
          startTime: {1}
          endTime: {2}
          surfacePose:
            translation: [{3[0]}, {3[1]}, 0]
            rotation: [0, 0, {3[2]}]
          constraint:
            type: Surface
            name: {4}
            fricCoeff: 0.4
            verticesName: {5}""".format(
                self.limbEnd[footSide],
                startTime, endTime,
                footPose,
                self.constraintName[footSide], self.verticesName[footSide])
        if fileObj is None:
            print(dumpStr)
        else:
            dumpStr += "\n"
            fileObj.write(dumpStr)

    def callback(self, msg):
        fileObj = None
        footSide = self.LEFT
        startTime = 3.0
        singleSupportDuration = 1.0
        doubleSupportDuration = 0.5
        footPoseOffset = np.array([0.036, -0.1, 0.0])

        for pose_msg in msg.poses[1:]:
            ori = pose_msg.orientation
            theta = transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2]
            footPose = np.array([pose_msg.position.x, pose_msg.position.y, theta]) + footPoseOffset

            self.dumpOneCommand(fileObj, footSide, footPose, startTime, startTime+singleSupportDuration)

            footSide = self.opposite(footSide)
            startTime += (singleSupportDuration + doubleSupportDuration)

        if fileObj is not None:
            fileObj.close()

        self.exit_flag = True

    def spin(self):
        rate = rospy.Rate(100)
        while not self.exit_flag:
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("dump_footstep", anonymous=True)
    dump = DumpMotion6DoFFootstepCommands()
    dump.spin()
    print("Run the following command to save ROS bag:")
    print("  rosbag record /marker_arr /current_left_poly_arr /current_right_poly_arr -l 1 -O `rospack find differentiable_rmap`/data/bridge_rmap_planning_footstep.bag")
