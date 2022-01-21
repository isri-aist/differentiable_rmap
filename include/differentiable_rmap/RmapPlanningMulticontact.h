/* Author: Masaki Murooka */

#pragma once

#include <unordered_map>

#include <differentiable_rmap/RmapPlanning.h>


namespace DiffRmap
{
/** \brief Limb. */
enum class Limb
{
  LeftFoot = 0,
  RightFoot,
  LeftHand,
  RightHand
};

/** \brief Convert string to limb. */
Limb strToLimb(const std::string& limb_str);
}

namespace std
{
using DiffRmap::Limb;

inline string to_string(Limb limb)
{
  if (limb == Limb::LeftFoot) {
    return std::string("LeftFoot");
  } else if (limb == Limb::RightFoot) {
    return std::string("RightFoot");
  } else if (limb == Limb::LeftHand) {
    return std::string("LeftHand");
  } else if (limb == Limb::RightHand) {
    return std::string("RightHand");
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[to_string] Unsupported Limb: {}", static_cast<int>(limb));
  }
}
}

namespace DiffRmap
{
/** \brief Class to plan multi-contact motion based on differentiable reachability map.

    This class does not inherit RmapPlanning because it has many differences from RmapPlanning (e.g., it holds multiple SVM models).
 */
class RmapPlanningMulticontact
{
 public:
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Rate in runLoop()
    int loop_rate = 2000;

    //! Step interval to publish in runLoop()
    int publish_interval = 20;

    //! Threshold of SVM predict value to be determined as reachable
    double svm_thre = 0.0;

    //! Limit of configuration update in one step [m], [rad]
    double delta_config_limit = 0.1;

    //! Initial sample pose list
    std::vector<sva::PTransformd> initial_sample_pose_list;

    //! Number of footsteps
    int motion_seq_len = 3;

    //! Regularization weight
    double reg_weight = 1e-6;

    //! Adjacent regularization weight
    double adjacent_reg_weight = 1e-3;

    //! QP objective weight for SVM inequality error
    double svm_ineq_weight = 1e6;

    //! Vertices of foot marker
    std::vector<Eigen::Vector3d> foot_vertices = {
      Eigen::Vector3d(-0.1, -0.05, 0.0),
      Eigen::Vector3d(0.1, -0.05, 0.0),
      Eigen::Vector3d(0.1, 0.05, 0.0),
      Eigen::Vector3d(-0.1, 0.05, 0.0)
    };

    /*! \brief Load mc_rtc configuration. */
    inline void load(const mc_rtc::Configuration& mc_rtc_config)
    {
      mc_rtc_config("loop_rate", loop_rate);
      mc_rtc_config("publish_interval", publish_interval);
      mc_rtc_config("svm_thre", svm_thre);
      mc_rtc_config("delta_config_limit", delta_config_limit);
      mc_rtc_config("initial_sample_pose_list", initial_sample_pose_list);
      mc_rtc_config("motion_seq_len", motion_seq_len);
      mc_rtc_config("reg_weight", reg_weight);
      mc_rtc_config("adjacent_reg_weight", adjacent_reg_weight);
      mc_rtc_config("svm_ineq_weight", svm_ineq_weight);
      mc_rtc_config("foot_vertices", foot_vertices);
    }
  };

 public:
  /*! \brief Number of limb. */
  static constexpr size_t limb_num = 3;

  /*! \brief Sampling space for foot. */
  static constexpr SamplingSpace FootSamplingSpaceType = SamplingSpace::SE2;

  /*! \brief Sampling space for hand. */
  static constexpr SamplingSpace HandSamplingSpaceType = SamplingSpace::R3;

  /*! \brief Get sampling space. */
  template <Limb limb>
  static inline constexpr SamplingSpace samplingSpaceType()
  {
    if constexpr (limb == Limb::LeftFoot || limb == Limb::RightFoot) {
        return FootSamplingSpaceType;
      } else if constexpr (limb == Limb::LeftHand || limb == Limb::RightHand) {
        return HandSamplingSpaceType;
      } else {
      static_assert(static_cast<bool>(static_cast<int>(limb)) && false,
                    "[samplingSpaceType] unsupported limb.");
    }
  }

 public:
  /*! \brief Type of sample vector. */
  template <Limb limb>
  using SampleType = Sample<samplingSpaceType<limb>()>;

  /*! \brief Type of input vector. */
  template <Limb limb>
  using InputType = Input<samplingSpaceType<limb>()>;

  /*! \brief Type of velocity vector. */
  template <Limb limb>
  using VelType = Vel<samplingSpaceType<limb>()>;

 public:
  /** \brief Constructor.
      \param svm_path_list path list of SVM model file
      \param bag_path_list path list of ROS bag file of grid set (empty for no grid set)
   */
  RmapPlanningMulticontact(const std::unordered_map<Limb, std::string>& svm_path_list,
                           const std::unordered_map<Limb, std::string>& bag_path_list);

  /** \brief Destructor. */
  ~RmapPlanningMulticontact();

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  void configure(const mc_rtc::Configuration& mc_rtc_config);

  /** \brief Setup planning. */
  void setup();

  /** \brief Run planning once.
      \param publish whether to publish message
   */
  void runOnce(bool publish);

  /** \brief Setup and run planning loop. */
  void runLoop();

 protected:
  /** \brief Publish marker array. */
  void publishMarkerArray() const;

  /** \brief Publish current state. */
  void publishCurrentState() const;

  /** \brief Transform topic callback. */
  void transCallback(const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg);

 protected:
  //! mc_rtc Configuration
  mc_rtc::Configuration mc_rtc_config_;

  //! Configuration
  Configuration config_;

  //! 
  std::unordered_map<Limb, std::shared_ptr<RmapPlanningBase>> rmap_planning_list_;

  //! QP coefficients
  OmgCore::QpCoeff qp_coeff_;

  //! QP solver
  std::shared_ptr<OmgCore::QpSolver> qp_solver_;

  //! Current sample
  // SampleType current_sample_ = poseToSample<SamplingSpaceType>(sva::PTransformd::Identity());

  //! Target sample
  Sample<FootSamplingSpaceType> target_sample_ = poseToSample<FootSamplingSpaceType>(sva::PTransformd::Identity());

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Subscriber trans_sub_;
  ros::Publisher marker_arr_pub_;
  ros::Publisher current_pose_arr_pub_;
  ros::Publisher current_poly_arr_pub_;
};
}
