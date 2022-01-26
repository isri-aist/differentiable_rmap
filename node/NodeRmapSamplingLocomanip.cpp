/* Author: Masaki Murooka */

#include <sch/S_Polyhedron/S_Polyhedron.h>
#include <sch/S_Object/S_Box.h>

#include <differentiable_rmap/RmapSamplingIK.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_sampling_locomanip");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // Setup robot
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

  // Instantiate
  std::string body_name = "tool0";
  pnh.param<std::string>("body_name", body_name, body_name);

  std::vector<std::string> joint_name_list;
  pnh.param<std::vector<std::string>>("joint_name_list", joint_name_list, joint_name_list);

  auto rmap_sampling =
      std::dynamic_pointer_cast<RmapSamplingIK<SamplingSpace::SE2>>(
          createRmapSamplingIK(
              SamplingSpace::SE2,
              rb,
              body_name,
              joint_name_list));

  // Configure
  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    rmap_sampling->configure(mc_rtc::Configuration(config_path));
  }

  // Set additional task list
  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    mc_rtc::Configuration mc_rtc_config(config_path);

    std::vector<std::string> door_collision_body_names_list = mc_rtc_config("door_collision_body_names_list");
    Eigen::Vector3d door_collision_box_scale = mc_rtc_config("door_collision_box_scale");
    double collision_task_weight = mc_rtc_config("collision_task_weight", 1.0);
    std::string robot_convex_path;
    nh.getParam("robot_convex_path", robot_convex_path);

    std::vector<std::shared_ptr<OmgCore::TaskBase>> additional_task_list;
    for (const auto& body_name : door_collision_body_names_list) {
      OmgCore::Twin<std::shared_ptr<sch::S_Object>> sch_objs;
      sch_objs[0] = OmgCore::loadSchPolyhedron(robot_convex_path + body_name + "_mesh-ch.txt");
      sch_objs[1] = std::make_shared<sch::S_Box>(
          door_collision_box_scale.x(), door_collision_box_scale.y(), door_collision_box_scale.z());
      auto task = std::make_shared<OmgCore::CollisionTask>(
          std::make_shared<OmgCore::CollisionFunc>(
              rmap_sampling->rbArr(),
              OmgCore::Twin<int>{0, 0},
              OmgCore::Twin<std::string>{body_name, "door"},
              sch_objs),
          0.05);
      task->setWeight(collision_task_weight);
      additional_task_list.push_back(task);
    }
    rmap_sampling->setAdditionalTaskList(additional_task_list);
  }

  // Run
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
