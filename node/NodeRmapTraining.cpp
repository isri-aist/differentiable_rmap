/* Author: Masaki Murooka */

#include <differentiable_rmap/RmapTraining.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_training");
  ros::NodeHandle pnh("~");

  std::string sampling_space_str = "R2";
  pnh.param<std::string>("sampling_space", sampling_space_str, sampling_space_str);
  SamplingSpace sampling_space = strToSamplingSpace(sampling_space_str);

  std::string bag_path = "/tmp/rmap_sample_set.bag";
  pnh.param<std::string>("bag_path", bag_path, bag_path);

  std::string svm_path = "/tmp/rmap_svm_model.libsvm";
  pnh.param<std::string>("svm_path", svm_path, svm_path);

  bool load_svm = false;
  pnh.param<bool>("load_svm", load_svm, load_svm);

  auto rmap_training = createRmapTraining(
      sampling_space,
      bag_path,
      svm_path,
      load_svm);

  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    rmap_training->configure(mc_rtc::Configuration(config_path));
  }

  rmap_training->runLoop();

  bool keep_alive = true;
  pnh.param<bool>("keep_alive", keep_alive, keep_alive);
  if (keep_alive) {
    ros::spin();
  }

  return 0;
}
