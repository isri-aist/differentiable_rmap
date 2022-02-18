/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <differentiable_rmap/RmapSampling.h>
#include <differentiable_rmap/RmapSamplingIK.h>

using namespace DiffRmap;

void testGenerateSample(bool use_ik)
{
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::shared_ptr<rbd::parsers::ParserResult> parse_res =
      OmgCore::parseUrdfFromRosparam(nh, "robot_description", rbd::Joint::Type::Fixed, {});
  auto rb = std::make_shared<OmgCore::Robot>(parse_res->mb, parse_res->name, parse_res->limits, parse_res->visual,
                                             parse_res->collision);
  // no velocity limit for the offline posture generator
  rb->jvel_max_scale_ = 1e10;

  std::string sampling_space_str = "R2";
  pnh.param<std::string>("sampling_space", sampling_space_str, sampling_space_str);
  SamplingSpace sampling_space = strToSamplingSpace(sampling_space_str);

  std::string body_name = "tool0";
  pnh.param<std::string>("body_name", body_name, body_name);

  std::vector<std::string> joint_name_list;
  pnh.param<std::vector<std::string>>("joint_name_list", joint_name_list, joint_name_list);

  std::shared_ptr<RmapSamplingBase> rmap_sampling;
  if(use_ik)
  {
    rmap_sampling = createRmapSamplingIK(sampling_space, rb, body_name, joint_name_list);
  }
  else
  {
    rmap_sampling = createRmapSampling(sampling_space, rb, body_name, joint_name_list);
  }

  if(pnh.hasParam("config_path"))
  {
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
}

TEST(TestRmapSampling, GenerateSampleR2FK)
{
  testGenerateSample(false);
}

TEST(TestRmapSampling, GenerateSampleR2IK)
{
  testGenerateSample(true);
}

int main(int argc, char ** argv)
{
  // Setup ROS
  ros::init(argc, argv, "test_rmap_sampling");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
