/* Author: Masaki Murooka */

#include <optmotiongen_msgs/RobotStateArray.h>

#include <differentiable_rmap/RmapSampling.h>
#include <differentiable_rmap/RmapSamplingIK.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_sampling");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::shared_ptr<rbd::parsers::ParserResult> parse_res =
      OmgCore::parseUrdfFromRosparam(
          nh,
          "robot_description",
          rbd::Joint::Type::Fixed,
          {});
  auto rb = std::make_shared<OmgCore::Robot>(
      parse_res->mb,
      parse_res->name,
      parse_res->limits,
      parse_res->visual,
      parse_res->collision);

  std::string sampling_space_str = "R2";
  pnh.param<std::string>("sampling_space", sampling_space_str, sampling_space_str);
  SamplingSpace sampling_space = strToSamplingSpace(sampling_space_str);

  std::string body_name = "tool0";
  pnh.param<std::string>("body_name", body_name, body_name);

  std::vector<std::string> joint_name_list;
  pnh.param<std::vector<std::string>>("joint_name_list", joint_name_list, joint_name_list);

  auto rmap_sampling = createRmapSampling(
      sampling_space,
      rb,
      body_name,
      joint_name_list);

  std::string bag_path = "/tmp/rmap_sample_set.bag";
  pnh.param<std::string>("bag_path", bag_path, bag_path);

  int sample_num = 10000;
  pnh.param<int>("sample_num", sample_num, sample_num);

  double sleep_rate = 0;
  pnh.param<double>("sleep_rate", sleep_rate, sleep_rate);

  rmap_sampling->run(bag_path, sample_num, sleep_rate);

  bool keep_alive = true;
  pnh.param<bool>("keep_alive", keep_alive, keep_alive);
  if (keep_alive) {
    ros::spin();
  }

  return 0;
}
