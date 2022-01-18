/* Author: Masaki Murooka */

#include <differentiable_rmap/RmapPlanningPlacement.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_planning_placement");
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
  // no velocity limit for the offline posture generator
  rb->jvel_max_scale_ = 1e10;

  std::string sampling_space_str = "R2";
  pnh.param<std::string>("sampling_space", sampling_space_str, sampling_space_str);
  SamplingSpace sampling_space = strToSamplingSpace(sampling_space_str);

  std::string svm_path = "/tmp/rmap_svm_model.libsvm";
  pnh.param<std::string>("svm_path", svm_path, svm_path);

  std::string bag_path = "/tmp/rmap_grid_set.bag";
  pnh.param<std::string>("bag_path", bag_path, bag_path);

  auto rmap_planning = createRmapPlanningPlacement(
      sampling_space,
      svm_path,
      bag_path);

  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    rmap_planning->configure(mc_rtc::Configuration(config_path));
  }

  std::dynamic_pointer_cast<RmapPlanningPlacementBase>(rmap_planning)->runLoop(rb);

  bool keep_alive = true;
  pnh.param<bool>("keep_alive", keep_alive, keep_alive);
  if (keep_alive) {
    ros::spin();
  }

  return 0;
}
