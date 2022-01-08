/* Author: Masaki Murooka */

#include <differentiable_rmap/RmapPlanning.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_planning");
  ros::NodeHandle pnh("~");

  std::string sampling_space_str = "R2";
  pnh.param<std::string>("sampling_space", sampling_space_str, sampling_space_str);
  SamplingSpace sampling_space = strToSamplingSpace(sampling_space_str);

  std::string svm_path = "/tmp/rmap_svm_model.libsvm";
  pnh.param<std::string>("svm_path", svm_path, svm_path);

  auto rmap_planning = createRmapPlanning(
      sampling_space,
      svm_path);

  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    rmap_planning->configure(mc_rtc::Configuration(config_path));
  }

  rmap_planning->runLoop();

  bool keep_alive = true;
  pnh.param<bool>("keep_alive", keep_alive, keep_alive);
  if (keep_alive) {
    ros::spin();
  }

  return 0;
}
