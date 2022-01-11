/* Author: Masaki Murooka */

#include <differentiable_rmap/RmapVisualization.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_visualization");
  ros::NodeHandle pnh("~");

  std::string sampling_space_str = "R2";
  pnh.param<std::string>("sampling_space", sampling_space_str, sampling_space_str);
  SamplingSpace sampling_space = strToSamplingSpace(sampling_space_str);

  std::string sample_bag_path = "/tmp/rmap_sample_set.bag";
  pnh.param<std::string>("sample_bag_path", sample_bag_path, sample_bag_path);

  std::string svm_path = "/tmp/rmap_svm_model.libsvm";
  pnh.param<std::string>("svm_path", svm_path, svm_path);

  auto rmap_visualization = createRmapVisualization(
      sampling_space,
      sample_bag_path,
      svm_path);

  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    rmap_visualization->configure(mc_rtc::Configuration(config_path));
  }

  std::string grid_bag_path = "/tmp/rmap_grid_set.bag";
  pnh.param<std::string>("grid_bag_path", grid_bag_path, grid_bag_path);

  rmap_visualization->runLoop(grid_bag_path);

  bool keep_alive = true;
  pnh.param<bool>("keep_alive", keep_alive, keep_alive);
  if (keep_alive) {
    ros::spin();
  }

  return 0;
}
