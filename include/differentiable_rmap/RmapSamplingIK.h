/* Author: Masaki Murooka */

#pragma once

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
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleVector = Eigen::Matrix<double, sample_dim_, 1>;

 public:
  /** \brief Constructor.
      \param rb robot
      \param body_name name of body whose pose is sampled
      \param joint_name_list name list of joints whose position is changed
  */
  RmapSamplingIK(const std::shared_ptr<OmgCore::Robot>& rb,
                 const std::string& body_name,
                 const std::vector<std::string>& joint_name_list);

 protected:
  /** \brief Generate one sample. */
  virtual void sampleOnce(int sample_idx) override;

 private:
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
