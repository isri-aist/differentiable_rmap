/* Author: Masaki Murooka */

#include <sch/S_Polyhedron/S_Polyhedron.h>

#include <optmotiongen/Task/CollisionTask.h>

#include <differentiable_rmap/RmapSamplingFootstep.h>

using namespace DiffRmap;


int main(int argc, char **argv)
{
  // Setup ROS
  ros::init(argc, argv, "rmap_sampling_footstep");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // Setup robot
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

  // Instantiate
  std::string support_foot_body_name = "Rleg_Link5";
  pnh.param<std::string>("support_foot_body_name", support_foot_body_name, support_foot_body_name);
  std::string swing_foot_body_name = "Lleg_Link5";
  pnh.param<std::string>("swing_foot_body_name", swing_foot_body_name, swing_foot_body_name);
  std::string waist_body_name = "Body";
  pnh.param<std::string>("waist_body_name", waist_body_name, waist_body_name);

  auto rmap_sampling =
      std::dynamic_pointer_cast<RmapSamplingFootstep<SamplingSpace::SE2>>(
          createRmapSamplingFootstep(
              SamplingSpace::SE2,
              rb,
              support_foot_body_name,
              swing_foot_body_name,
              waist_body_name));

  // Configure
  if (pnh.hasParam("config_path")) {
    std::string config_path;
    pnh.getParam("config_path", config_path);
    rmap_sampling->configure(mc_rtc::Configuration(config_path));
  }

  // Add collision tasks
  std::string robot_convex_path;
  nh.getParam("robot/convex_path", robot_convex_path);
  std::vector<OmgCore::Twin<std::string>> collision_body_names_list = {
    {"Lleg_Link2", "Rleg_Link2"},
    {"Lleg_Link3", "Rleg_Link3"},
    {"Lleg_Link5", "Rleg_Link5"},
  };

  std::vector<std::shared_ptr<OmgCore::TaskBase>> additional_task_list;
  for (const auto& body_names : collision_body_names_list) {
    OmgCore::Twin<int> rb_idxs;
    OmgCore::Twin<std::shared_ptr<sch::S_Object>> sch_objs;
    for (auto i : {0, 1}) {
      rb_idxs[i] = 0;
      sch_objs[i] = OmgCore::loadSchPolyhedron(robot_convex_path + body_names[i] + "_mesh-ch.txt");
    }
    additional_task_list.push_back(
        std::make_shared<OmgCore::CollisionTask>(
            std::make_shared<OmgCore::CollisionFunc>(
                rmap_sampling->rbArr(),
                rb_idxs,
                body_names,
                sch_objs),
            0.05));
  }
  rmap_sampling->setAdditionalTaskList(additional_task_list);

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
