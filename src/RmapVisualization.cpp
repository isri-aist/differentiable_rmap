/* Author: Masaki Murooka */

#include <chrono>

#include <mc_rtc/constants.h>

#include <visualization_msgs/MarkerArray.h>
#include <differentiable_rmap/RmapSampleSet.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapVisualization.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/GridUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapVisualization<SamplingSpaceType>::RmapVisualization(
    const std::string& sample_bag_path,
    const std::string& svm_path)
{
  // Setup ROS
  marker_arr_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_arr", 1, true);

  // Load ROS bag
  loadSampleSet(sample_bag_path);

  // Load SVM model
  loadSVM(svm_path);
}

template <SamplingSpace SamplingSpaceType>
RmapVisualization<SamplingSpaceType>::~RmapVisualization()
{
  if (svm_mo_) {
    delete svm_mo_;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::configure(const mc_rtc::Configuration& mc_rtc_config)
{
  mc_rtc_config_ = mc_rtc_config;
  config_.load(mc_rtc_config);
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::setup(const std::string& grid_bag_path)
{
  dumpGridSet(grid_bag_path);
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::runOnce()
{
  // Publish
  publishMarkerArray();
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::runLoop(const std::string& grid_bag_path)
{
  setup(grid_bag_path);

  ros::Rate rate(100);
  while (ros::ok()) {
    runOnce();

    rate.sleep();
    ros::spinOnce();
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::loadSampleSet(const std::string& sample_bag_path)
{
  ROS_INFO_STREAM("Load sample set from " << sample_bag_path);

  differentiable_rmap::RmapSampleSet::ConstPtr sample_set_msg =
      loadBag<differentiable_rmap::RmapSampleSet>(sample_bag_path);

  if (sample_set_msg->type != static_cast<size_t>(SamplingSpaceType)) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "SamplingSpace does not match with message: {} != {}",
        sample_set_msg->type, static_cast<size_t>(SamplingSpaceType));
  }

  for (int i = 0; i < sample_dim_; i++) {
    sample_min_[i] = sample_set_msg->min[i];
    sample_max_[i] = sample_set_msg->max[i];
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::loadSVM(const std::string& svm_path)
{
  ROS_INFO_STREAM("Load SVM model from " << svm_path);
  svm_mo_ = svm_load_model(svm_path.c_str());

  int num_sv = svm_mo_->l;
  svm_coeff_vec_.resize(num_sv);
  svm_sv_mat_.resize(input_dim_, num_sv);
  setSVMPredictionMat<SamplingSpaceType>(
      svm_coeff_vec_,
      svm_sv_mat_,
      svm_mo_);
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::dumpGridSet(
    const std::string& grid_bag_path)
{
  SampleType sample_range = sample_max_ - sample_min_;

  // Set number of division
  Eigen::Matrix<int, sample_dim_, 1> divide_nums;
  SampleType resolution;
  if constexpr (SamplingSpaceType == SamplingSpace::R2 ||
                SamplingSpaceType == SamplingSpace::R3) {
      resolution.setConstant(config_.pos_resolution);
    } else if constexpr (SamplingSpaceType == SamplingSpace::SO2 ||
                         SamplingSpaceType == SamplingSpace::SO3) {
      resolution.setConstant(config_.rot_resolution);
    } else if constexpr (SamplingSpaceType == SamplingSpace::SE2) {
      resolution <<
          config_.pos_resolution, config_.pos_resolution, config_.rot_resolution;
    } else if constexpr (SamplingSpaceType == SamplingSpace::SE3) {
      resolution <<
          config_.pos_resolution, config_.pos_resolution, config_.pos_resolution,
          config_.rot_resolution, config_.rot_resolution, config_.rot_resolution;
    }
  divide_nums = (sample_range.array() / resolution.array()).ceil().template cast<int>().max(1);

  // Set grid set message
  grid_set_msg_.type = static_cast<size_t>(SamplingSpaceType);
  grid_set_msg_.divide_nums.resize(sample_dim_);
  grid_set_msg_.min.resize(sample_dim_);
  grid_set_msg_.max.resize(sample_dim_);
  int total_grid_num = 1;
  for (int i = 0; i < sample_dim_; i++) {
    grid_set_msg_.divide_nums[i] = divide_nums[i];
    grid_set_msg_.min[i] = sample_min_[i];
    grid_set_msg_.max[i] = sample_max_[i];
    total_grid_num *= divide_nums[i];
  }
  grid_set_msg_.values.resize(total_grid_num);

  // Predict on whole grid
  SampleType sample;
  SampleType divide_ratios;
  Eigen::Matrix<int, sample_dim_, 1> divide_idxs = Eigen::Matrix<int, sample_dim_, 1>::Zero();
  int grid_idx = 0;
  bool break_flag = false;
  do {
    // Calculate ratio of division
    calcGridDivideRatios(divide_ratios, divide_idxs, divide_nums);

    // Predict
    sample = divide_ratios.cwiseProduct(sample_range) + sample_min_;
    double svm_value = calcSVMValue<SamplingSpaceType>(
        sample,
        svm_mo_->param,
        svm_mo_,
        svm_coeff_vec_,
        svm_sv_mat_);
    grid_set_msg_.values[grid_idx] = svm_value;

    // Update divide_idxs
    break_flag = updateGridDivideIdxs(divide_idxs, divide_nums);

    grid_idx++;
  } while (!break_flag);
  assert(grid_idx == grid_set_msg_.values.size());

  // Dump to ROS bag
  rosbag::Bag bag(grid_bag_path, rosbag::bagmode::Write);
  bag.write("/rmap_grid_set", ros::Time::now(), grid_set_msg_);
  ROS_INFO_STREAM("Dump grid set to " << grid_bag_path);
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::publishMarkerArray() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Instantiate marker array
  visualization_msgs::MarkerArray marker_arr_msg;

  // Delete marker
  visualization_msgs::Marker del_marker;
  del_marker.action = visualization_msgs::Marker::DELETEALL;
  del_marker.header = header_msg;
  del_marker.id = marker_arr_msg.markers.size();
  marker_arr_msg.markers.push_back(del_marker);

  // Reachable grids marker
  {
    SampleType sample_range = sample_max_ - sample_min_;

    visualization_msgs::Marker grids_marker;
    grids_marker.header = header_msg;
    grids_marker.ns = "reachable_grids";
    grids_marker.id = marker_arr_msg.markers.size();
    grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
    grids_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.0, 0.0, 1.0});
    grids_marker.scale.x = 1.1 * sample_range[0] / grid_set_msg_.divide_nums[0];
    if constexpr (sample_dim_ > 1) {
        grids_marker.scale.y = 1.1 * sample_range[1] / grid_set_msg_.divide_nums[1];
      } else {
      grids_marker.scale.y = 0.1;
    }
    if constexpr (sample_dim_ > 2) {
        grids_marker.scale.z = 1.1 * sample_range[2] / grid_set_msg_.divide_nums[2];
      } else {
      grids_marker.scale.z = 0.1;
    }
    grids_marker.pose = OmgCore::toPoseMsg(sva::PTransformd::Identity());

    // Predict on whole grid
    SampleType sample;
    SampleType divide_ratios;
    Eigen::Matrix<int, sample_dim_, 1> divide_idxs = Eigen::Matrix<int, sample_dim_, 1>::Zero();
    int grid_idx = 0;
    bool break_flag = false;
    do {
      // Calculate ratio of division
      calcGridDivideRatios(divide_ratios, divide_idxs, grid_set_msg_.divide_nums);

      // Append cube
      sample = divide_ratios.cwiseProduct(sample_range) + sample_min_;
      if (grid_set_msg_.values[grid_idx] > config_.svm_thre) {
        grids_marker.points.push_back(
            OmgCore::toPointMsg(sampleToCloudPos<SamplingSpaceType>(sample)));
      }

      // Update divide_idxs
      break_flag = updateGridDivideIdxs(divide_idxs, grid_set_msg_.divide_nums);

      grid_idx++;
    } while (!break_flag);
    marker_arr_msg.markers.push_back(grids_marker);
  }

  marker_arr_pub_.publish(marker_arr_msg);
}

std::shared_ptr<RmapVisualizationBase> DiffRmap::createRmapVisualization(
    SamplingSpace sampling_space,
    const std::string& sample_bag_path,
    const std::string& svm_path)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapVisualization<SamplingSpace::R2>>(sample_bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapVisualization<SamplingSpace::SO2>>(sample_bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapVisualization<SamplingSpace::SE2>>(sample_bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapVisualization<SamplingSpace::R3>>(sample_bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapVisualization<SamplingSpace::SO3>>(sample_bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapVisualization<SamplingSpace::SE3>>(sample_bag_path, svm_path);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapVisualization] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapVisualization<SamplingSpace::R2>;
template class RmapVisualization<SamplingSpace::SO2>;
template class RmapVisualization<SamplingSpace::SE2>;
template class RmapVisualization<SamplingSpace::R3>;
template class RmapVisualization<SamplingSpace::SO3>;
template class RmapVisualization<SamplingSpace::SE3>;
