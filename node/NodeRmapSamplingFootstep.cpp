/* Author: Masaki Murooka */

#include <differentiable_rmap/RmapSamplingFootstep.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_sampling_footstep");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::shared_ptr<rbd::parsers::ParserResult> parse_res =
      OmgCore::parseUrdfFromRosparam(
          nh,
          "robot_description",
          rbd::Joint::Type::Free,
          {});
  auto rb = std::make_shared<OmgCore::Robot>(
      parse_res->mb,
      parse_res->name,
      parse_res->limits,
      parse_res->visual,
      parse_res->collision);
  // no velocity limit for the offline posture generator
  rb->jvel_max_scale_ = 1e10;

  std::string sampling_space_str = "SE2";
  pnh.param<std::string>("sampling_space", sampling_space_str, sampling_space_str);
  SamplingSpace sampling_space = strToSamplingSpace(sampling_space_str);

  std::string support_foot_body_name = "Rleg_Link5";
  pnh.param<std::string>("support_foot_body_name", support_foot_body_name, support_foot_body_name);
  std::string swing_foot_body_name = "Lleg_Link5";
  pnh.param<std::string>("swing_foot_body_name", swing_foot_body_name, swing_foot_body_name);
  std::string waist_body_name = "Body";
  pnh.param<std::string>("waist_body_name", waist_body_name, waist_body_name);

  std::shared_ptr<RmapSamplingBase> rmap_sampling = createRmapSamplingFootstep(
      sampling_space,
      rb,
      support_foot_body_name,
      swing_foot_body_name,
      waist_body_name);

  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    rmap_sampling->configure(mc_rtc::Configuration(config_path));
  }

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
