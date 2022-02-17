/* Author: Masaki Murooka */

#pragma once

#include <optmotiongen/Problem/IterativeQpProblem.h>
#include <optmotiongen/Task/BodyTask.h>

#include <differentiable_rmap/RmapSampling.h>

namespace DiffRmap
{
/** \brief Class to generate samples for reachability map based on inverse kinematics.
    \tparam SamplingSpaceType sampling space
*/
template<SamplingSpace SamplingSpaceType>
class RmapSamplingIK : public RmapSampling<SamplingSpaceType>
{
public:
  /*! \brief Configuration. */
  struct Configuration : public RmapSampling<SamplingSpaceType>::Configuration
  {
    //! Number of samples to make bounding box
    int bbox_sample_num = 1000;

    //! Padding rate of bounding box
    double bbox_padding_rate = 1.2;

    //! Lower and upper limits of body Yaw angle [rad] ([deg] in YAML file)
    std::pair<double, double> body_yaw_limits = {-M_PI, M_PI};

    //! Number of IK trial
    int ik_trial_num = 10;

    //! Number of IK loop
    int ik_loop_num = 50;

    //! Threshold of IK [m], [rad]
    double ik_error_thre = 1e-2;

    //! Constraint space of IK (default is same as template parameter SamplingSpaceType)
    std::string ik_constraint_space = "";

    //! Lower and upper limits of ratio of reachable samples
    std::pair<double, double> reachable_sample_ratio_limits = {0.0, 1.0};

    /*! \brief Load mc_rtc configuration. */
    inline virtual void load(const mc_rtc::Configuration & mc_rtc_config) override
    {
      RmapSampling<SamplingSpaceType>::Configuration::load(mc_rtc_config);

      mc_rtc_config("bbox_sample_num", bbox_sample_num);
      mc_rtc_config("bbox_padding_rate", bbox_padding_rate);
      if(mc_rtc_config.has("body_yaw_limits"))
      {
        mc_rtc_config("body_yaw_limits", body_yaw_limits);
        body_yaw_limits.first = mc_rtc::constants::toRad(body_yaw_limits.first);
        body_yaw_limits.second = mc_rtc::constants::toRad(body_yaw_limits.second);
      }
      mc_rtc_config("ik_trial_num", ik_trial_num);
      mc_rtc_config("ik_loop_num", ik_loop_num);
      mc_rtc_config("ik_error_thre", ik_error_thre);
      mc_rtc_config("ik_constraint_space", ik_constraint_space);
      mc_rtc_config("reachable_sample_ratio_limits", reachable_sample_ratio_limits);
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
  RmapSamplingIK(const std::shared_ptr<OmgCore::Robot> & rb,
                 const std::string & body_name,
                 const std::vector<std::string> & joint_name_list);

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration & mc_rtc_config) override;

  /** \brief Set additional task list in IK
      \param additional_task_list additional task list in IK
  */
  inline void setAdditionalTaskList(const std::vector<std::shared_ptr<OmgCore::TaskBase>> & additional_task_list)
  {
    additional_task_list_ = additional_task_list;
  }

protected:
  /** \brief Constructor.
      \param rb robot
  */
  RmapSamplingIK(const std::shared_ptr<OmgCore::Robot> & rb);

  /** \brief Setup. */
  virtual void setup() override;

  /** \brief Setup sampling. */
  virtual void setupSampling() override;

  /** \brief Generate one sample.
      \return true if succeeded to generate sample
  */
  virtual bool sampleOnce(int sample_idx) override;

  /** \brief Publish ROS message. */
  virtual void publish() override;

protected:
  //! Configuration
  Configuration config_;

  //! Taskset for IK
  OmgCore::Taskset taskset_;

  //! Body task for IK
  std::shared_ptr<OmgCore::BodyPoseTask> body_task_;

  //! Additional task list in IK
  std::vector<std::shared_ptr<OmgCore::TaskBase>> additional_task_list_;

  //! IK problem
  std::shared_ptr<OmgCore::IterativeQpProblem> problem_;

  //! Body position coefficient to make sample from [-1:1] random value
  Eigen::Vector3d body_pos_coeff_;
  //! Body position offset to make sample from [-1:1] random value
  Eigen::Vector3d body_pos_offset_;
  //! Body Yaw angle coefficient to make sample from [-1:1] random value
  double body_yaw_coeff_;
  //! Body Yaw angle offset to make sample from [-1:1] random value
  double body_yaw_offset_;

  //! Number of reachabe samples
  size_t reachable_sample_num_ = 0;

protected:
  // See https://stackoverflow.com/a/6592617
  using RmapSampling<SamplingSpaceType>::rb_arr_;
  using RmapSampling<SamplingSpaceType>::rbc_arr_;
  using RmapSampling<SamplingSpaceType>::aux_rb_arr_;

  using RmapSampling<SamplingSpaceType>::body_name_;
  using RmapSampling<SamplingSpaceType>::body_idx_;

  using RmapSampling<SamplingSpaceType>::joint_name_list_;
  using RmapSampling<SamplingSpaceType>::joint_idx_list_;
  using RmapSampling<SamplingSpaceType>::joint_pos_coeff_;
  using RmapSampling<SamplingSpaceType>::joint_pos_offset_;

  using RmapSampling<SamplingSpaceType>::sample_list_;

  using RmapSampling<SamplingSpaceType>::reachability_list_;

  using RmapSampling<SamplingSpaceType>::collision_task_list_;

  using RmapSampling<SamplingSpaceType>::nh_;

  using RmapSampling<SamplingSpaceType>::rs_arr_pub_;
  using RmapSampling<SamplingSpaceType>::reachable_cloud_pub_;
  using RmapSampling<SamplingSpaceType>::unreachable_cloud_pub_;
  using RmapSampling<SamplingSpaceType>::collision_marker_pub_;

  using RmapSampling<SamplingSpaceType>::reachable_cloud_msg_;
  using RmapSampling<SamplingSpaceType>::unreachable_cloud_msg_;
};

/** \brief Create RmapSamplingIK instance.
    \param sampling_space sampling space
    \param rb robot
    \param body_name name of body whose pose is sampled
    \param joint_name_list name list of joints whose position is changed
*/
std::shared_ptr<RmapSamplingBase> createRmapSamplingIK(SamplingSpace sampling_space,
                                                       const std::shared_ptr<OmgCore::Robot> & rb,
                                                       const std::string & body_name,
                                                       const std::vector<std::string> & joint_name_list);
} // namespace DiffRmap
