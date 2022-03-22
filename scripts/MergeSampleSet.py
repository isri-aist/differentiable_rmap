#! /usr/bin/env python

import numpy as np

import rospy
import rosbag
from differentiable_rmap.msg import RmapSampleSet, RmapSample


# Merge sample sets
class MergeSampleSet(object):
    sampleTypeToStr = {
        21: "R2",
        22: "SO2",
        23: "SE2",
        31: "R3",
        32: "SO3",
        33: "SE3"}

    def __init__(self, bag_path_list):
        # Load bag
        self.msg_list = [None] * len(bag_path_list)
        for i, bag_path in enumerate(bag_path_list):
            rospy.loginfo("Load sample set from {}".format(bag_path))
            cnt = 0
            with rosbag.Bag(bag_path) as bag:
                for topic, msg, t in bag.read_messages():
                    if cnt >= 1:
                        rospy.logwarn("Multiple messages are stored in bag file. load only last one.")
                    self.msg_list[i] = msg
                    cnt += 1
                if cnt == 0:
                    rospy.logerr("No message found.")

        # Check bag
        for msg in self.msg_list[1:]:
            if msg.type != self.msg_list[0].type:
                rospy.logerr("Sample type is not consistent: {} != {}",
                             sampleTypeToStr(msg.type), sampleTypeToStr(self.msg_list[0].type))

        # Merge bag list
        merged_msg = RmapSampleSet()

        merged_msg.type = self.msg_list[0].type
        merged_msg.min = self.msg_list[0].min
        merged_msg.max = self.msg_list[0].max
        for msg in self.msg_list[1:]:
            merged_msg.min = np.minimum(merged_msg.min, msg.min)
            merged_msg.max = np.maximum(merged_msg.max, msg.max)

        sample_num_list = [None] * len(self.msg_list)
        for i, msg in enumerate(self.msg_list):
            sample_num_list[i] = len(msg.samples)
            if i > 0:
                sample_num_list[i] += sample_num_list[i - 1]

        idx_list = np.arange(sample_num_list[-1])
        np.random.shuffle(idx_list)
        for idx in idx_list:
            msg_idx = np.argmax(sample_num_list > idx)
            idx_offset = (0 if msg_idx == 0 else sample_num_list[msg_idx - 1])
            sample_msg = self.msg_list[msg_idx].samples[idx - idx_offset]
            merged_msg.samples.append(sample_msg)

        # Save as rosbag
        merged_bag_path = "/tmp/merged_sample_set.bag"
        rospy.loginfo("Save merged sample set to {}".format(merged_bag_path))
        bag = rosbag.Bag(merged_bag_path, 'w')
        try:
            bag.write("/rmap_sample_set", merged_msg)
        finally:
            bag.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: {} <bag_path1> <bag_path2> ...".format(sys.argv[0]))
        sys.exit(1)
    rospy.init_node("merge_sample_set", anonymous=False)
    merge = MergeSampleSet(sys.argv[1:])
