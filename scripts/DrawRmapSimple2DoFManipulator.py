#! /usr/bin/env python

import rospy
import numpy as np

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import MarkerArray, Marker


class DrawRmapSimple2DoFManipulator(object):
    def __init__(self):
        self.pub = rospy.Publisher("rmap_boundary_gt", MarkerArray, queue_size=1, latch=True)

    def run(self):
        marker_arr_msg = MarkerArray()

        del_marker_msg = Marker()
        del_marker_msg.header.frame_id = "world"
        del_marker_msg.id = len(marker_arr_msg.markers)
        del_marker_msg.action = Marker.DELETEALL
        marker_arr_msg.markers.append(del_marker_msg)

        line_marker_msg = Marker()
        line_marker_msg.header.frame_id = "world"
        line_marker_msg.type = Marker.LINE_STRIP
        line_marker_msg.scale.x = 0.008
        line_marker_msg.color = ColorRGBA(1, 0, 0, 1)
        line_marker_msg.pose.orientation.w = 1
        line_marker_msg.points += self.makeArcPoints(
            center=np.array([0.0, 0.0]),
            radius=1.5,
            theta_range=[0.0, np.pi/2])
        line_marker_msg.points += self.makeArcPoints(
            center=np.array([0.0, 1.0]),
            radius=0.5,
            theta_range=[np.pi/2, np.pi*3/2])
        line_marker_msg.points += self.makeArcPoints(
            center=np.array([0.0, 0.0]),
            radius=0.5,
            theta_range=[np.pi/2, 0.0])
        line_marker_msg.points += self.makeArcPoints(
            center=np.array([1.0, 0.0]),
            radius=0.5,
            theta_range=[np.pi, 0])
        marker_arr_msg.markers.append(line_marker_msg)

        self.pub.publish(marker_arr_msg)

    def makeArcPoints(
            self,
            center,
            radius,
            theta_range,
            theta_step=np.deg2rad(1)):
        point_msg_list = []
        theta_step = np.sign(theta_range[1] - theta_range[0]) * np.abs(theta_step)
        for theta in np.arange(theta_range[0], theta_range[1], theta_step):
            pos = radius * np.array([np.cos(theta), np.sin(theta)]) + center
            point_msg_list.append(Point(pos[0], pos[1], 10.0))
        return point_msg_list

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("draw_rmap", anonymous=True)
    draw = DrawRmapSimple2DoFManipulator()
    draw.run()
    draw.spin()
