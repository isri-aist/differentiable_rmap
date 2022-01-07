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
template <SamplingSpace SamplingSpaceType>
class RmapSamplingIK: public RmapSampling<SamplingSpaceType>
{
 public:
  /*! \brief Configuration. */
  struct Configuration: public RmapSampling<SamplingSpaceType>::Configuration
  {
    //! Number of samples to make bounding box
    int bbox_sample_num = 1000;

    //! Number of IK trial
    int ik_trial_num = 10;

    //! Number of IK loop
    int ik_loop_num = 50;

    //! Threshold of IK [m], [rad]
    double ik_error_thre = 1e-2;

    /*! \brief Load mc_rtc configuration. */
    inline virtual void load(const mc_rtc::Configuration& mc_rtc_config) override
    {
      mc_rtc_config("bbox_sample_num", bbox_sample_num);
      mc_rtc_config("ik_trial_num", ik_trial_num);
      mc_rtc_config("ik_loop_num", ik_loop_num);
      mc_rtc_config("ik_error_thre", ik_error_thre);
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

  /** \brief Generate one sample. */
  virtual void sampleOnce(int sample_idx) override;

 protected:
  //! Configuration
  Configuration config_;

  //! Taskset for IK
  OmgCore::Taskset taskset_;

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
