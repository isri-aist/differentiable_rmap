#! /usr/bin/env python

import numpy as np

import rospy
from tf import transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion


def makeTransformMsgFromDict(d):
    pos_arr = d.get("pos", [0, 0, 0])
    axis_angle_arr = d.get("rot", [1, 0, 0, 0])
    return Transform(Vector3(*pos_arr),
                     Quaternion(*transformations.quaternion_about_axis(axis_angle_arr[3], axis_angle_arr[0:3])))

# Broadcast a rotating TF to be used as a camera TF in Rviz
class BroadcastRotateTF(object):
    def __init__(self):
        self.dt = rospy.get_param("~dt", 0.05) # [sec]
        self.rotate_vel = rospy.get_param("~rotate_vel", np.deg2rad(10)) # [rad]

        self.br = tf2_ros.TransformBroadcaster()
        self.static_br = tf2_ros.StaticTransformBroadcaster()

        self.theta = 0.0 # [rad]
        rospy.Timer(rospy.Duration(self.dt), self.timerCallback)

        self.broadcastStaticTF()

    def broadcastStaticTF(self):
        stamp = rospy.Time.now()
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.child_frame_id = "rotate_parent"
        msg.transform = makeTransformMsgFromDict(rospy.get_param("~origin", {"pos": [0.5, 1.0, -0.2]}))
        self.static_br.sendTransform(msg)

    def timerCallback(self, event):
        self.theta += self.dt * self.rotate_vel
        stamp = rospy.Time.now()
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "rotate_parent"
        msg.child_frame_id = "rotate_child"
        msg.transform.translation.x = 0
        msg.transform.translation.y = 0
        msg.transform.translation.z = 0
        quat = transformations.quaternion_from_euler(0, 0, self.theta)
        msg.transform.rotation.x = quat[0]
        msg.transform.rotation.y = quat[1]
        msg.transform.rotation.z = quat[2]
        msg.transform.rotation.w = quat[3]
        self.br.sendTransform(msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("broadcast_rotate_tf", anonymous=True)
    broadcast = BroadcastRotateTF()
    broadcast.spin()
