/* Author: Masaki Murooka */

#pragma once

#include <optmotiongen/Problem/IterativeQpProblem.h>
#include <optmotiongen/Task/BodyTask.h>
#include <optmotiongen/Task/CollisionTask.h>

#include <differentiable_rmap/RmapSampling.h>


namespace DiffRmap
{
/** \brief Class to generate samples for reachability map based on inverse kinematics.
    \tparam SamplingSpaceType sampling space
*/
template <SamplingSpace SamplingSpaceType>
class RmapSamplingIK: public RmapSampling<SamplingSpaceType>
{
 public:
  /*! \brief Configuration. */
  struct Configuration: public RmapSampling<SamplingSpaceType>::Configuration
  {
    //! Number of samples to make bounding box
    int bbox_sample_num = 1000;

    //! Padding rate of bounding box
    double bbox_padding_rate = 1.2;

    //! Number of IK trial
    int ik_trial_num = 10;

    //! Number of IK loop
    int ik_loop_num = 50;

    //! Threshold of IK [m], [rad]
    double ik_error_thre = 1e-2;

    //! Constraint space of IK (default is same as template parameter SamplingSpaceType)
    std::string ik_constraint_space = "";

    //! Body name pair list for collision avoidance
    std::vector<OmgCore::Twin<std::string>> collision_body_names_list;

    /*! \brief Load mc_rtc configuration. */
    inline virtual void load(const mc_rtc::Configuration& mc_rtc_config) override
    {
      RmapSampling<SamplingSpaceType>::Configuration::load(mc_rtc_config);

      mc_rtc_config("bbox_sample_num", bbox_sample_num);
      mc_rtc_config("bbox_padding_rate", bbox_padding_rate);
      mc_rtc_config("ik_trial_num", ik_trial_num);
      mc_rtc_config("ik_loop_num", ik_loop_num);
      mc_rtc_config("ik_error_thre", ik_error_thre);
      mc_rtc_config("ik_constraint_space", ik_constraint_space);
      if (mc_rtc_config.has("collision_body_names_list")) {
        std::vector<std::string> collision_body_names_list_flat = mc_rtc_config("collision_body_names_list");
        if (collision_body_names_list_flat.size() % 2 != 0) {
          mc_rtc::log::error_and_throw<std::runtime_error>(
              "collision_body_names_list size must be a multiple of 2, but is {}",
              collision_body_names_list_flat.size());
        }
        collision_body_names_list.clear();
        for (size_t i = 0; i < collision_body_names_list_flat.size() / 2; i++) {
          collision_body_names_list.push_back(OmgCore::Twin<std::string>(
              collision_body_names_list_flat[2 * i],
              collision_body_names_list_flat[2 * i + 1]));
        }
      }
    }
  };

 public:
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleType = Sample<SamplingSpaceType>;

 public:
  /** \brief Constructor.
      \param rb robot
      \param body_name name of body whose pose is sampled
      \param joint_name_list name list of joints whose position is changed
  */
  RmapSamplingIK(const std::shared_ptr<OmgCore::Robot>& rb,
                 const std::string& body_name,
                 const std::vector<std::string>& joint_name_list);

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) override;

 protected:
  /** \brief Constructor.
      \param rb robot
  */
  RmapSamplingIK(const std::shared_ptr<OmgCore::Robot>& rb);

  /** \brief Setup sampling. */
  virtual void setupSampling() override;

  /** \brief Setup collision tasks. */
  void setupCollisionTask();

  /** \brief Generate one sample. */
  virtual void sampleOnce(int sample_idx) override;

  /** \brief Publish ROS message. */
  virtual void publish() override;

 protected:
  //! Configuration
  Configuration config_;

  //! Taskset for IK
  OmgCore::Taskset taskset_;

  //! Collision task list in IK
  std::vector<std::shared_ptr<OmgCore::CollisionTask>> collision_task_list_;

  //! Auxiliary robot array (always empty)
  OmgCore::AuxRobotArray aux_rb_arr_;

  //! Body task for IK
  std::shared_ptr<OmgCore::BodyPoseTask> body_task_;

  //! IK problem
  std::shared_ptr<OmgCore::IterativeQpProblem> problem_;

  //! Body position coefficient to make sample from [-1:1] random value
  Eigen::Vector3d body_pos_coeff_;
  //! Body position offset to make sample from [-1:1] random value
  Eigen::Vector3d body_pos_offset_;

  //! ROS related members
  ros::Publisher collision_marker_pub_;

 protected:
  // See https://stackoverflow.com/a/6592617
  using RmapSampling<SamplingSpaceType>::rb_arr_;
  using RmapSampling<SamplingSpaceType>::rbc_arr_;

  using RmapSampling<SamplingSpaceType>::body_name_;
  using RmapSampling<SamplingSpaceType>::body_idx_;

  using RmapSampling<SamplingSpaceType>::joint_name_list_;
  using RmapSampling<SamplingSpaceType>::joint_idx_list_;
  using RmapSampling<SamplingSpaceType>::joint_pos_coeff_;
  using RmapSampling<SamplingSpaceType>::joint_pos_offset_;

  using RmapSampling<SamplingSpaceType>::sample_list_;

  using RmapSampling<SamplingSpaceType>::reachability_list_;

  using RmapSampling<SamplingSpaceType>::nh_;

  using RmapSampling<SamplingSpaceType>::reachable_cloud_msg_;
  using RmapSampling<SamplingSpaceType>::unreachable_cloud_msg_;
};

/** \brief Create RmapSamplingIK instance.
    \param sampling_space sampling space
    \param rb robot
    \param body_name name of body whose pose is sampled
    \param joint_name_list name list of joints whose position is changed
*/
std::shared_ptr<RmapSamplingBase> createRmapSamplingIK(
    SamplingSpace sampling_space,
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& body_name,
    const std::vector<std::string>& joint_name_list);
}
