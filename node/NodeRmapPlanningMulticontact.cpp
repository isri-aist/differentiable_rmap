/* Author: Masaki Murooka */

#include <differentiable_rmap/RmapPlanningMulticontact.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_planning_multicontact");
  ros::NodeHandle pnh("~");

  std::vector<std::string> limb_name_list;
  pnh.param<std::vector<std::string>>("limb_name_list", limb_name_list, limb_name_list);
  std::vector<std::string> svm_path_list;
  pnh.param<std::vector<std::string>>("svm_path_list", svm_path_list, svm_path_list);
  std::vector<std::string> bag_path_list;
  pnh.param<std::vector<std::string>>("bag_path_list", bag_path_list, bag_path_list);

  std::unordered_map<Limb, std::string> svm_path_map;
  std::unordered_map<Limb, std::string> bag_path_map;
  for (size_t i = 0; i < limb_name_list.size(); i++) {
    Limb limb = strToLimb(limb_name_list[i]);
    svm_path_map[limb] = svm_path_list[i];
    bag_path_map[limb] = bag_path_list[i];
  }
  auto rmap_planning = std::make_shared<RmapPlanningMulticontact>(svm_path_map, bag_path_map);

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
