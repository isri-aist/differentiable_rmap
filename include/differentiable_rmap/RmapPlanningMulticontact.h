/* Author: Masaki Murooka */

#pragma once

#include <map>
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
    std::map<Limb, sva::PTransformd> initial_sample_pose_list = {
      {Limb::LeftFoot, sva::PTransformd::Identity()},
      {Limb::RightFoot, sva::PTransformd::Identity()},
      {Limb::LeftHand, sva::PTransformd::Identity()},
      {Limb::RightHand, sva::PTransformd::Identity()}
    };

    //! Number of footsteps
    int motion_len = 3;

    //! Regularization weight
    double reg_weight = 1e-6;

    //! Adjacent regularization weight
    double adjacent_reg_weight = 1e-3;

    //! Start foot weight
    double start_foot_weight = 1e3;

    //! QP objective weight for SVM inequality error
    double svm_ineq_weight = 1e6;

    //! Waist height [m]
    double waist_height = 0.8;

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

      std::map<std::string, sva::PTransformd> tmp_initial_sample_pose_list;
      mc_rtc_config("initial_sample_pose_list", tmp_initial_sample_pose_list);
      for (const auto& tmp_initial_sample_pose_kv : tmp_initial_sample_pose_list) {
        initial_sample_pose_list[strToLimb(tmp_initial_sample_pose_kv.first)] = tmp_initial_sample_pose_kv.second;
      }

      mc_rtc_config("motion_len", motion_len);
      mc_rtc_config("reg_weight", reg_weight);
      mc_rtc_config("adjacent_reg_weight", adjacent_reg_weight);
      mc_rtc_config("start_foot_weight", start_foot_weight);
      mc_rtc_config("svm_ineq_weight", svm_ineq_weight);
      mc_rtc_config("waist_height", waist_height);
      mc_rtc_config("foot_vertices", foot_vertices);
    }
  };

 public:
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

  /*! \brief Dimension of sample for foot. */
  static constexpr int foot_sample_dim_ = sampleDim<FootSamplingSpaceType>();

  /*! \brief Dimension of SVM input for foot. */
  static constexpr int foot_input_dim_ = inputDim<FootSamplingSpaceType>();

  /*! \brief Dimension of velocity for foot. */
  static constexpr int foot_vel_dim_ = velDim<FootSamplingSpaceType>();

  /*! \brief Dimension of sample for hand. */
  static constexpr int hand_sample_dim_ = sampleDim<HandSamplingSpaceType>();

  /*! \brief Dimension of SVM input for hand. */
  static constexpr int hand_input_dim_ = inputDim<HandSamplingSpaceType>();

  /*! \brief Dimension of velocity for hand. */
  static constexpr int hand_vel_dim_ = velDim<HandSamplingSpaceType>();

 public:
  /*! \brief Type of sample vector for foot. */
  using FootSampleType = Sample<FootSamplingSpaceType>;

  /*! \brief Type of input vector for foot. */
  using FootInputType = Input<FootSamplingSpaceType>;

  /*! \brief Type of velocity vector for foot. */
  using FootVelType = Vel<FootSamplingSpaceType>;

  /*! \brief Type of sample vector for hand. */
  using HandSampleType = Sample<HandSamplingSpaceType>;

  /*! \brief Type of input vector for hand. */
  using HandInputType = Input<HandSamplingSpaceType>;

  /*! \brief Type of velocity vector for hand. */
  using HandVelType = Vel<HandSamplingSpaceType>;

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
  /** \brief Get list of rmap planning for specified limb. */
  template <Limb limb>
  inline std::shared_ptr<RmapPlanning<samplingSpaceType<limb>()>> rmapPlanning()
  {
    return std::dynamic_pointer_cast<RmapPlanning<samplingSpaceType<limb>()>>(rmap_planning_list_.at(limb));
  }

  /** \brief Publish marker array. */
  void publishMarkerArray() const;

  /** \brief Publish current state. */
  void publishCurrentState() const;

  /** \brief Transform topic callback. */
  void transCallback(const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg);

 protected:
  //! Sample corresponding to identity pose for foot
  static inline const Sample<FootSamplingSpaceType> identity_foot_sample_ =
      poseToSample<FootSamplingSpaceType>(sva::PTransformd::Identity());

  //! Sample corresponding to identity pose for hand
  static inline const Sample<HandSamplingSpaceType> identity_hand_sample_ =
      poseToSample<HandSamplingSpaceType>(sva::PTransformd::Identity());

 protected:
  //! mc_rtc Configuration
  mc_rtc::Configuration mc_rtc_config_;

  //! Configuration
  Configuration config_;

  //! List of rmap planning for each limb
  std::unordered_map<Limb, std::shared_ptr<RmapPlanningBase>> rmap_planning_list_;

  //! QP coefficients
  OmgCore::QpCoeff qp_coeff_;

  //! QP solver
  std::shared_ptr<OmgCore::QpSolver> qp_solver_;

  //! Current sample sequence for foot
  std::vector<Sample<FootSamplingSpaceType>> current_foot_sample_seq_;

  //! Current sample sequence for hand
  std::vector<Sample<HandSamplingSpaceType>> current_hand_sample_seq_;

  //! Target sample
  Sample<FootSamplingSpaceType> target_foot_sample_ = poseToSample<FootSamplingSpaceType>(sva::PTransformd::Identity());

  //! Adjacent regularization matrix
  Eigen::MatrixXd adjacent_reg_mat_;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Subscriber trans_sub_;
  ros::Publisher marker_arr_pub_;
  ros::Publisher current_pose_arr_pub_;
  ros::Publisher current_poly_arr_pub_;
};
}
