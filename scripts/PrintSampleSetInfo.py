#! /usr/bin/env python

import sys
import numpy as np

import rospy
import rosbag


# Print information of sample set
class PrintSampleSetInfo(object):
    sampleTypeToStr = {
        21: "R2",
        22: "SO2",
        23: "SE2",
        31: "R3",
        32: "SO3",
        33: "SE3"}

    def __init__(self, bag_path):
        # Load bag
        rospy.loginfo("Load sample set from {}".format(bag_path))
        cnt = 0
        with rosbag.Bag(bag_path) as bag:
            for topic, msg, t in bag.read_messages():
                if cnt >= 1:
                    rospy.logwarn("Multiple messages are stored in bag file. load only last one.")
                self.msg = msg
                cnt += 1
            if cnt == 0:
                rospy.logerr("No message found.")

        # Print info
        sample_num_list = [0, 0]
        for sample_msg in self.msg.samples:
            sample_num_list[0 if sample_msg.is_reachable else 1] += 1
        print("==== Sample set information ====")
        print("- Sampling space: {}".format(self.sampleTypeToStr[self.msg.type]))
        print("- Sample num: {} (reachable: {} ({} %) / unreachable: {} ({} %))".format(
            len(self.msg.samples),
            sample_num_list[0], 100 * float(sample_num_list[0]) / len(self.msg.samples),
            sample_num_list[1], 100 * float(sample_num_list[1]) / len(self.msg.samples)))
        print("- Min position: {}".format(self.msg.min))
        print("- Max position: {}".format(self.msg.max))


if __name__ == "__main__":
    rospy.init_node("print_sample_set_info", anonymous=False)
    if len(sys.argv) < 2:
        print("usage: print_sample_set_info <bag_path>")
        sys.exit(1)
    print_info = PrintSampleSetInfo(sys.argv[1])
