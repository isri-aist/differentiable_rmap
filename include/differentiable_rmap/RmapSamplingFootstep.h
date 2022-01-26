/* Author: Masaki Murooka */

#pragma once

#include <mc_rtc/constants.h>

#include <differentiable_rmap/RmapSamplingIK.h>


namespace DiffRmap
{
/** \brief Class to generate samples for reachability map for footstep planning.
    \tparam SamplingSpaceType sampling space
*/
template <SamplingSpace SamplingSpaceType>
class RmapSamplingFootstep: public RmapSamplingIK<SamplingSpaceType>
{
 public:
  /*! \brief Type of footstep position. */
  using FootstepPos = Sample<SamplingSpace::SE2>;

 public:
  /*! \brief Configuration. */
  struct Configuration: public RmapSamplingIK<SamplingSpaceType>::Configuration
  {
    //! Upper limit of footstep sampling position [m], [rad]
    FootstepPos upper_footstep_pos = FootstepPos(0.2, 0.3, mc_rtc::constants::toRad(20));

    //! Lower limit of footstep sampling position [m], [rad]
    FootstepPos lower_footstep_pos = FootstepPos(-0.2, 0.1, mc_rtc::constants::toRad(-20));

    //! Waist height [m]
    double waist_height = 0.8;

    //! Initial posture (list of joint name and angle [deg])
    std::map<std::string, double> initial_posture;

    /*! \brief Load mc_rtc configuration. */
    inline virtual void load(const mc_rtc::Configuration& mc_rtc_config) override
    {
      RmapSamplingIK<SamplingSpaceType>::Configuration::load(mc_rtc_config);

      if (mc_rtc_config.has("upper_footstep_pos")) {
        mc_rtc_config("upper_footstep_pos", upper_footstep_pos);
        upper_footstep_pos.z() = mc_rtc::constants::toRad(upper_footstep_pos.z());
      }
      if (mc_rtc_config.has("lower_footstep_pos")) {
        mc_rtc_config("lower_footstep_pos", lower_footstep_pos);
        lower_footstep_pos.z() = mc_rtc::constants::toRad(lower_footstep_pos.z());
      }
      mc_rtc_config("waist_height", waist_height);
      mc_rtc_config("initial_posture", initial_posture);
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
      \param support_foot_body_name name of support foot body
      \param swing_foot_body_name name of swing foot body
      \param waist_body_name name of waist body
  */
  RmapSamplingFootstep(const std::shared_ptr<OmgCore::Robot>& rb,
                       const std::string& support_foot_body_name,
                       const std::string& swing_foot_body_name,
                       const std::string& waist_body_name);

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) override;

 protected:
  /** \brief Setup sampling. */
  virtual void setupSampling() override;

  /** \brief Generate one sample. */
  virtual void sampleOnce(int sample_idx) override;

 protected:
  //! Configuration
  Configuration config_;

  //! Task of bodies used in IK
  std::shared_ptr<OmgCore::BodyPoseTask> support_foot_body_task_;
  std::shared_ptr<OmgCore::BodyPoseTask> swing_foot_body_task_;
  std::shared_ptr<OmgCore::BodyPoseTask> waist_body_task_;

  //! Taskset list for IK
  std::vector<OmgCore::Taskset> taskset_list_;

  //! Name of bodies used in IK
  std::string support_foot_body_name_;
  std::string swing_foot_body_name_;
  std::string waist_body_name_;

  //! Footstep position coefficient to make sample from [-1:1] random value
  FootstepPos footstep_pos_coeff_;
  //! Footstep position offset to make sample from [-1:1] random value
  FootstepPos footstep_pos_offset_;

 protected:
  // See https://stackoverflow.com/a/6592617
  using RmapSamplingIK<SamplingSpaceType>::rb_arr_;
  using RmapSamplingIK<SamplingSpaceType>::rbc_arr_;

  using RmapSamplingIK<SamplingSpaceType>::body_name_;
  using RmapSamplingIK<SamplingSpaceType>::body_idx_;

  using RmapSamplingIK<SamplingSpaceType>::joint_name_list_;
  using RmapSamplingIK<SamplingSpaceType>::joint_idx_list_;
  using RmapSamplingIK<SamplingSpaceType>::joint_pos_coeff_;
  using RmapSamplingIK<SamplingSpaceType>::joint_pos_offset_;

  using RmapSamplingIK<SamplingSpaceType>::sample_list_;

  using RmapSamplingIK<SamplingSpaceType>::reachability_list_;

  using RmapSamplingIK<SamplingSpaceType>::nh_;

  using RmapSamplingIK<SamplingSpaceType>::reachable_cloud_msg_;
  using RmapSamplingIK<SamplingSpaceType>::unreachable_cloud_msg_;

  using RmapSamplingIK<SamplingSpaceType>::taskset_;

  using RmapSamplingIK<SamplingSpaceType>::collision_task_list_;

  using RmapSamplingIK<SamplingSpaceType>::aux_rb_arr_;

  using RmapSamplingIK<SamplingSpaceType>::body_task_;

  using RmapSamplingIK<SamplingSpaceType>::problem_;

  using RmapSamplingIK<SamplingSpaceType>::body_pos_coeff_;
  using RmapSamplingIK<SamplingSpaceType>::body_pos_offset_;
};

/** \brief Create RmapSamplingFootstep instance.
    \param sampling_space sampling space
    \param rb robot
    \param support_foot_body_name name of support foot body
    \param swing_foot_body_name name of swing foot body
    \param waist_body_name name of waist body
*/
std::shared_ptr<RmapSamplingBase> createRmapSamplingFootstep(
    SamplingSpace sampling_space,
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& support_foot_body_name,
    const std::string& swing_foot_body_name,
    const std::string& waist_body_name);
}
