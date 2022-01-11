/* Author: Masaki Murooka */

#include <chrono>

#include <mc_rtc/constants.h>

#include <visualization_msgs/MarkerArray.h>
#include <differentiable_rmap/RmapSampleSet.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapVisualization.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapVisualization<SamplingSpaceType>::RmapVisualization(
    const std::string& bag_path,
    const std::string& svm_path)
{
  // Setup ROS
  marker_arr_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_arr", 1, true);

  // Load ROS bag
  loadSampleSet(bag_path);

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
void RmapVisualization<SamplingSpaceType>::setup()
{
  // Predict on whole grid
  SampleType sample_range = sample_max_ - sample_min_;
  Eigen::Matrix<int, sample_dim_, 1> divide_nums;
  divide_nums.setConstant(5);
  Eigen::Matrix<int, sample_dim_, 1> divide_idxs = Eigen::Matrix<int, sample_dim_, 1>::Zero();
  bool break_flag = false;
  while (true) {
    // Predict
    SampleType sample = divide_idxs.template cast<double>().cwiseProduct(
        (divide_nums.array() - 1).matrix().template cast<double>().cwiseInverse()).cwiseProduct(
            sample_range) + sample_min_;
    double svm_value = calcSVMValue<SamplingSpaceType>(
        sample,
        svm_mo_->param,
        svm_mo_,
        svm_coeff_vec_,
        svm_sv_mat_);
    std::cout << "sample: " << sample.transpose() << " / " << svm_value << std::endl;

    // Update divide_idxs
    for (size_t i = 0; i < sample_dim_; i++) {
      divide_idxs[i]++;
      if (divide_idxs[i] == divide_nums[i]) {
        // If there is a carry, the current digit value is set to zero
        divide_idxs[i] = 0;
        // If there is a carry at the top, exit the outer loop
        if (i == sample_dim_ - 1) {
          break_flag = true;
        }
      } else {
        // If there is no carry, it will end
        break;
      }
    }
    if (break_flag) {
      break;
    }
  }
  std::cout << "sample_min: " << sample_min_.transpose() << std::endl;
  std::cout << "sample_max: " << sample_max_.transpose() << std::endl;
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::runOnce()
{
  // Publish
  publishMarkerArray();
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::runLoop()
{
  setup();

  ros::Rate rate(100);
  while (ros::ok()) {
    runOnce();

    rate.sleep();
    ros::spinOnce();
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::loadSampleSet(const std::string& bag_path)
{
  ROS_INFO_STREAM("Load sample set from " << bag_path);

  differentiable_rmap::RmapSampleSet::ConstPtr sample_set_msg =
      loadBag<differentiable_rmap::RmapSampleSet>(bag_path);

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

  // XY plane marker
  visualization_msgs::Marker xy_plane_marker;
  double plane_thickness = 0.01;
  xy_plane_marker.header = header_msg;
  xy_plane_marker.ns = "xy_plane";
  xy_plane_marker.id = marker_arr_msg.markers.size();
  xy_plane_marker.type = visualization_msgs::Marker::CUBE;
  xy_plane_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.8, 0.8, 1.0});
  xy_plane_marker.scale.x = 100.0;
  xy_plane_marker.scale.y = 100.0;
  xy_plane_marker.scale.z = plane_thickness;
  xy_plane_marker.pose = OmgCore::toPoseMsg(
      sva::PTransformd(Eigen::Vector3d(0, 0, config_.svm_thre - 0.5 * plane_thickness)));
  marker_arr_msg.markers.push_back(xy_plane_marker);

  marker_arr_pub_.publish(marker_arr_msg);
}

std::shared_ptr<RmapVisualizationBase> DiffRmap::createRmapVisualization(
    SamplingSpace sampling_space,
    const std::string& bag_path,
    const std::string& svm_path)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapVisualization<SamplingSpace::R2>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapVisualization<SamplingSpace::SO2>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapVisualization<SamplingSpace::SE2>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapVisualization<SamplingSpace::R3>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapVisualization<SamplingSpace::SO3>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapVisualization<SamplingSpace::SE3>>(bag_path, svm_path);
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
