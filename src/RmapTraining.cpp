/* Author: Masaki Murooka */

#include <map>
#include <unordered_set>
#include <chrono>

#include <mc_rtc/constants.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/MarkerArray.h>
#include <differentiable_rmap/RmapSampleSet.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapTraining.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


void RmapTrainingBase::configure(const mc_rtc::Configuration& mc_rtc_config)
{
  mc_rtc_config_ = mc_rtc_config;
  config_ = mc_rtc_config;
}

template <SamplingSpace SamplingSpaceType>
RmapTraining<SamplingSpaceType>::RmapTraining(const std::string& bag_path,
                                              const std::string& svm_path):
    svm_loaded_(bag_path.empty()),
    svm_path_(svm_path)
{
  // Setup SVM parameter
  setupSVMParam();

  // Setup ROS
  rmap_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("rmap_cloud", 1, true);
  sliced_rmap_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("rmap_cloud_sliced", 1, true);
  marker_arr_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_arr", 1, true);
  grid_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);

  xy_plane_height_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/xy_plane_height",
          0.0);
  svm_gamma_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/svm_gamma",
          svm_param_.gamma);
  svm_nu_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/svm_nu",
          svm_param_.nu);
  slice_z_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/slice_z",
          0.0);
  slice_roll_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/slice_roll",
          0.0);
  slice_pitch_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/slice_pitch",
          0.0);
  slice_yaw_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/slice_yaw",
          0.0);

  // Load
  if (svm_loaded_) {
    loadSVM();
  } else {
    loadBag(bag_path);
  }

  // Setup grid map
  setupGridMap();
}

template <SamplingSpace SamplingSpaceType>
RmapTraining<SamplingSpaceType>::~RmapTraining()
{
  // Free memory
  if (all_input_nodes_) {
    delete[] all_input_nodes_;
  }
  if (svm_prob_.x) {
    delete[] svm_prob_.x;
  }
  if (svm_prob_.y) {
    delete[] svm_prob_.y;
  }
  if (svm_mo_) {
    delete svm_mo_;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::configure(const mc_rtc::Configuration& mc_rtc_config)
{
  RmapTrainingBase::configure(mc_rtc_config);

  if (mc_rtc_config_.has("xy_plane_height")) {
    xy_plane_height_manager_->setValue(static_cast<double>(mc_rtc_config_("xy_plane_height")));
  }
  if (mc_rtc_config_.has("svm_gamma")) {
    svm_gamma_manager_->setValue(static_cast<double>(mc_rtc_config_("svm_gamma")));
  }
  if (mc_rtc_config_.has("svm_nu")) {
    svm_nu_manager_->setValue(static_cast<double>(mc_rtc_config_("svm_nu")));
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::run()
{
  ros::Rate rate(100);
  while (ros::ok()) {
    // Update
    updateSVMParam();
    updateSliceOrigin();

    // Train SVM
    if (!svm_loaded_ && train_required_) {
      train_required_ = false;
      trainSVM();
    }

    // Predict SVM
    if (train_updated_ || slice_updated_) {
      train_updated_ = false;
      slice_updated_ = false;
      predictOnGridMap();
      publishSlicedCloud();
    }

    // Publish
    publishMarkerArray();

    rate.sleep();
    ros::spinOnce();
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::setupSVMParam()
{
  svm_param_.svm_type = ONE_CLASS;
  svm_param_.kernel_type = RBF;
  svm_param_.degree = 3;
  svm_param_.gamma = 30; // smoothness (smaller is smoother)
  svm_param_.nu = 0.05; // outliers ratio
  svm_param_.coef0 = 0;
  svm_param_.cache_size = 16000; // 16GB // default 100
  svm_param_.C = 1;
  svm_param_.eps = 1e-1; // default 1e-3
  svm_param_.p = 0.1;
  svm_param_.shrinking = 0; // default 1
  svm_param_.probability = 0;
  svm_param_.nr_weight = 0;
  svm_param_.weight_label = NULL;
  svm_param_.weight = NULL;

  if constexpr (SamplingSpaceType == SamplingSpace::SE3) {
      svm_param_.gamma = 15; // 10 // 20
      svm_param_.eps = 0.1; // 0.5
    }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::updateSVMParam()
{
  if (svm_gamma_manager_->hasNewValue()) {
    svm_param_.gamma = svm_gamma_manager_->value();
    svm_gamma_manager_->update();
    train_required_ = true;
  }
  if (svm_nu_manager_->hasNewValue()) {
    svm_param_.nu = svm_nu_manager_->value();
    svm_nu_manager_->update();
    train_required_ = true;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::updateSliceOrigin()
{
  if (slice_z_manager_->hasNewValue()) {
    slice_origin_.translation().z() = slice_z_manager_->value();
    slice_z_manager_->update();
    slice_updated_ = true;
  }
  if (slice_roll_manager_->hasNewValue() ||
      slice_pitch_manager_->hasNewValue() ||
      slice_yaw_manager_->hasNewValue()) {
    Eigen::Vector3d vec(
        slice_roll_manager_->value(), slice_pitch_manager_->value(), slice_yaw_manager_->value());
    if (vec.norm() < 1e-20) {
      slice_origin_.rotation().setIdentity();
    } else {
      slice_origin_.rotation() = Eigen::AngleAxisd(vec.norm(), vec.normalized()).toRotationMatrix().transpose();
    }
    slice_roll_manager_->update();
    slice_pitch_manager_->update();
    slice_yaw_manager_->update();
    slice_updated_ = true;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::setupGridMap()
{
  // Calculate sample min/max
  SampleVector sample_min = SampleVector::Constant(1e10);
  SampleVector sample_max = SampleVector::Constant(-1e10);
  SampleVector sample_range;
  {
    for (const SampleVector& sample : sample_list_) {
      sample_min = sample_min.cwiseMin(sample);
      sample_max = sample_max.cwiseMax(sample);
    }
    sample_range = sample_max - sample_min;

    sample_min -= config_.grid_map_margin_ratio * sample_range;
    sample_max += config_.grid_map_margin_ratio * sample_range;
    sample_range = sample_max - sample_min;
  }

  // Create grid map
  {
    grid_map_ = std::make_shared<grid_map::GridMap>(std::vector<std::string>{"svm_prediction"});

    SampleVector sample_center = (sample_min + sample_max) / 2;
    grid_map_->setFrameId("world");
    grid_map_->setGeometry(grid_map::Length(sample_range[0], sample_range[1]),
                           config_.grid_map_resolution,
                           grid_map::Position(sample_center[0], sample_center[1]));
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::loadBag(const std::string& bag_path)
{
  // Load ROS bag
  ROS_INFO_STREAM("Load sample set from " << bag_path);
  rosbag::Bag bag(bag_path, rosbag::bagmode::Read);
  int cnt = 0;
  for (rosbag::MessageInstance const m :
           rosbag::View(bag, rosbag::TopicQuery(std::vector<std::string>{"/rmap_sample_set"}))) {
    if (cnt >= 1) {
      ROS_WARN("Multiple messages are stored in bag file. load only first one.");
      break;
    }

    differentiable_rmap::RmapSampleSet::ConstPtr sample_set_msg =
        m.instantiate<differentiable_rmap::RmapSampleSet>();
    if (sample_set_msg == nullptr) {
      mc_rtc::log::error_and_throw<std::runtime_error>("Failed to load sample set message from rosbag.");
    }
    if (sample_set_msg->type != static_cast<size_t>(SamplingSpaceType)) {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "SamplingSpace does not match with message: {} != {}",
          sample_set_msg->type, static_cast<size_t>(SamplingSpaceType));
    }
    sample_list_.resize(sample_set_msg->samples.size());
    for (size_t i = 0; i < sample_set_msg->samples.size(); i++) {
      for (int j = 0; j < sample_dim_; j++) {
        sample_list_[i][j] = sample_set_msg->samples[i].position[j];
      }
    }

    cnt++;
  }
  if (cnt == 0) {
    mc_rtc::log::error_and_throw<std::runtime_error>("Sample set message not found.");
  }

  // Publish cloud
  {
    std_msgs::Header header_msg;
    header_msg.frame_id = "world";
    header_msg.stamp = ros::Time::now();

    sensor_msgs::PointCloud cloud_msg;
    cloud_msg.header = header_msg;
    cloud_msg.points.resize(sample_list_.size());
    for (size_t i = 0; i < sample_list_.size(); i++) {
      cloud_msg.points[i] = OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample_list_[i]));
    }
    rmap_cloud_pub_.publish(cloud_msg);
  }

  // Setup SVM problem
  {
    svm_prob_.l = sample_list_.size();
    svm_prob_.y = new double[svm_prob_.l];
    svm_prob_.x = new svm_node*[svm_prob_.l];

    all_input_nodes_ = new svm_node[(input_dim_ + 1) * svm_prob_.l];
    for (size_t i = 0; i < sample_list_.size(); i++) {
      const SampleVector& sample = sample_list_[i];
      size_t idx = (input_dim_ + 1) * i;
      setInputNode<SamplingSpaceType>(&(all_input_nodes_[idx]), sampleToInput<SamplingSpaceType>(sample));
      svm_prob_.x[i] = &all_input_nodes_[idx];
      svm_prob_.y[i] = 1;
    }
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::loadSVM()
{
  ROS_INFO_STREAM("Load SVM model from " << svm_path_);
  svm_mo_ = svm_load_model(svm_path_.c_str());
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::trainSVM()
{
  // Check SVM problem and parameter
  {
    const char* check_ret = svm_check_parameter(&svm_prob_, &svm_param_);
    if (check_ret) {
      mc_rtc::log::error_and_throw<std::runtime_error>("[svm_check_parameter] {}", check_ret);
    }
  }

  // Train SVM
  {
    auto start_time = std::chrono::system_clock::now();

    svm_mo_ = svm_train(&svm_prob_, &svm_param_);

    double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now() - start_time).count();
    ROS_INFO_STREAM("SVM train duration: " << duration << " [ms]");

    int num_sv = svm_mo_->l;
    svm_coeff_vec_.resize(num_sv);
    svm_sv_mat_.resize(input_dim_, num_sv);
    for (int i = 0; i < num_sv; i++) {
      svm_coeff_vec_[i] = svm_mo_->sv_coef[0][i];
      svm_sv_mat_.col(i) = toEigenVector<SamplingSpaceType>(svm_mo_->SV[i]);
    }
  }

  // Save SVM
  {
    auto start_time = std::chrono::system_clock::now();

    ROS_INFO_STREAM("Save SVM model to " << svm_path_);
    // The original function causes SEGV, so use the hotfix version
    svm_save_model_hotfix(svm_path_.c_str(), svm_mo_);

    double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now() - start_time).count();
    ROS_INFO_STREAM("SVM save duration: " << duration << " [ms]");
  }

  train_updated_ = true;
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::predictOnGridMap()
{
  // Predict
  {
    auto start_time = std::chrono::system_clock::now();

    if constexpr (use_libsvm_prediction_) {
        setInputNode<SamplingSpaceType>(input_node_, InputVector::Zero());
      }

    size_t grid_idx = 0;
    SampleVector origin_sample = poseToSample<SamplingSpaceType>(slice_origin_);
    for (grid_map::GridMapIterator it(*grid_map_); !it.isPastEnd(); ++it) {
      grid_map::Position pos;
      grid_map_->getPosition(*it, pos);

      SampleVector sample = origin_sample;
      sample.x() = pos.x();
      if constexpr (sample_dim_ > 1) {
          sample.y() = pos.y();
        }

      double svm_value;
      if constexpr (use_libsvm_prediction_) {
          setInputNodeOnlyValue<SamplingSpaceType>(input_node_, sampleToInput<SamplingSpaceType>(sample));
          svm_predict_values(svm_mo_, input_node_, &svm_value);
        } else {
        svm_value = calcSVMValue<SamplingSpaceType>(
            sampleToInput<SamplingSpaceType>(sample),
            svm_param_,
            svm_mo_,
            svm_coeff_vec_,
            svm_sv_mat_);
      }
      grid_map_->at("svm_prediction", *it) = config_.grid_map_height_scale * svm_value;
      grid_idx++;
    }

    double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now() - start_time).count();
    ROS_INFO_STREAM("SVM predict duration: " << duration << " [ms] (predict-one: " <<
                    duration / grid_idx <<" [ms])");
  }

  // Publish
  {
    auto start_time = std::chrono::system_clock::now();

    grid_map_->setTimestamp(ros::Time::now().toNSec());
    grid_map_msgs::GridMap grid_map_msg;
    grid_map::GridMapRosConverter::toMessage(*grid_map_, grid_map_msg);
    grid_map_pub_.publish(grid_map_msg);

    double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now() - start_time).count();
    // Publish is fast compared with other process
    // ROS_INFO_STREAM("SVM publish duration: " << duration << " [ms]");
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::publishSlicedCloud() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  sensor_msgs::PointCloud cloud_msg;
  cloud_msg.header = header_msg;
  for (const SampleVector& sample : sample_list_) {
    if constexpr (SamplingSpaceType == SamplingSpace::SE2) {
        double origin_theta = calcYawAngle(slice_origin_.rotation().transpose());
        double theta_diff = sample.z() - origin_theta;
        theta_diff = std::fabs(std::atan2(std::sin(theta_diff), std::cos(theta_diff)));
        if (theta_diff > mc_rtc::constants::toRad(config_.slice_se2_theta_thre)) {
          continue;
        }
      } else if constexpr (SamplingSpaceType == SamplingSpace::R3) {
        double origin_z = slice_origin_.translation().z();
        if (std::fabs(sample.z() - origin_z) > config_.slice_r3_z_thre) {
          continue;
        }
      } else if constexpr (SamplingSpaceType == SamplingSpace::SE3) {
        Eigen::Quaterniond origin_quat(slice_origin_.rotation().transpose());
        Eigen::Matrix<double, sampleDim<SamplingSpace::SO3>(), 1> sample_so3 =
            sample.template tail<sampleDim<SamplingSpace::SO3>()>();
        Eigen::Quaterniond sample_quat(sample_so3.w(), sample_so3.x(), sample_so3.y(), sample_so3.z());
        double theta_diff = std::fabs(Eigen::AngleAxisd(origin_quat.inverse() * sample_quat).angle());
        if (std::fabs(sample.z() - slice_origin_.translation().z()) > config_.slice_se3_z_thre ||
            theta_diff > mc_rtc::constants::toRad(config_.slice_se3_theta_thre)) {
          continue;
        }
      }

    cloud_msg.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  }
  sliced_rmap_cloud_pub_.publish(cloud_msg);
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::publishMarkerArray() const
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
      sva::PTransformd(Eigen::Vector3d(0, 0, xy_plane_height_manager_->value() - 0.5 * plane_thickness)));
  marker_arr_msg.markers.push_back(xy_plane_marker);

  marker_arr_pub_.publish(marker_arr_msg);
}

std::shared_ptr<RmapTrainingBase> DiffRmap::createRmapTraining(
    SamplingSpace sampling_space,
    const std::string& bag_path,
    const std::string& svm_path)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapTraining<SamplingSpace::R2>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapTraining<SamplingSpace::SO2>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapTraining<SamplingSpace::SE2>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapTraining<SamplingSpace::R3>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapTraining<SamplingSpace::SO3>>(bag_path, svm_path);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapTraining<SamplingSpace::SE3>>(bag_path, svm_path);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapTraining] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}
