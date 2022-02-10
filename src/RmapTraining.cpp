/* Author: Masaki Murooka */

#include <chrono>
#include <functional>

#include <mc_rtc/constants.h>

#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/MarkerArray.h>
#include <differentiable_rmap/RmapSampleSet.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapTraining.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/EvalUtils.h>
#include <differentiable_rmap/BaselineUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapTraining<SamplingSpaceType>::RmapTraining(const std::string& bag_path,
                                              const std::string& svm_path,
                                              bool load_svm):
    load_svm_(load_svm),
    svm_path_(svm_path),
    train_required_(!load_svm)
{
  // Setup ROS
  reachable_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("reachable_cloud", 1, true);
  unreachable_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("unreachable_cloud", 1, true);
  sliced_reachable_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("reachable_cloud_sliced", 1, true);
  sliced_unreachable_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("unreachable_cloud_sliced", 1, true);
  marker_arr_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_arr", 1, true);
  grid_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
  eval_srv_ = nh_.advertiseService(
      "evaluate",
      &RmapTraining<SamplingSpaceType>::evaluateCallback,
      this);

  // Load ROS bag
  loadSampleSet(bag_path);

  // Setup SVM parameter
  // This must be called after loadSampleSet() because this depends on reachability list
  if (!load_svm_) {
    setupSVMParam();
  }

  // Setup SubscVariableManager
  // This must be called after setupSVMParam() because this depends on SVM parameter
  svm_thre_manager_ =
      std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
          "variable/svm_thre",
          0.0);
  if (!load_svm_) {
    svm_gamma_manager_ =
        std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
            "variable/svm_gamma",
            svm_param_.gamma);
    svm_nu_manager_ =
        std::make_shared<SubscVariableManager<std_msgs::Float64, double>>(
            "variable/svm_nu",
            svm_param_.nu);
  }
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

  // Load SVM model
  if (load_svm_) {
    loadSVM();
  }
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
  mc_rtc_config_ = mc_rtc_config;
  config_.load(mc_rtc_config);

  if (mc_rtc_config_.has("svm_thre")) {
    svm_thre_manager_->setValue(static_cast<double>(mc_rtc_config_("svm_thre")));
  }
  if (!load_svm_ && mc_rtc_config_.has("svm_gamma")) {
    svm_gamma_manager_->setValue(static_cast<double>(mc_rtc_config_("svm_gamma")));
  }
  if (!load_svm_ && mc_rtc_config_.has("svm_nu")) {
    svm_nu_manager_->setValue(static_cast<double>(mc_rtc_config_("svm_nu")));
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::setup()
{
  // Setup grid map
  setupGridMap();
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::runOnce()
{
  // Update
  if (!load_svm_) {
    updateSVMParam();
  }
  updateSliceOrigin();

  // Train SVM
  if (train_required_) {
    train_required_ = false;
    trainSVM();
  }

  // Predict SVM
  if (train_updated_ || slice_updated_) {
    train_updated_ = false;
    slice_updated_ = false;
    predictOnSlicePlane();
    publishSlicedCloud();
  }

  // Publish
  publishMarkerArray();
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::runLoop()
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
void RmapTraining<SamplingSpaceType>::evaluateAccuracy(
    const std::string& bag_path,
    const PredictOnceFuncType& predict_once_func,
    const PredictSetupFuncType& predict_setup_func)
{
  ROS_INFO_STREAM("Load evaluation sample set from " << bag_path);

  // Load ROS bag
  differentiable_rmap::RmapSampleSet::ConstPtr sample_set_msg =
      loadBag<differentiable_rmap::RmapSampleSet>(bag_path);

  if (sample_set_msg->type != static_cast<size_t>(SamplingSpaceType)) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[RmapTraining::evaluateAccuracy] SamplingSpace does not match with message: {} != {}",
        sample_set_msg->type, static_cast<size_t>(SamplingSpaceType));
  }

  std::unordered_map<PredictResult, size_t> predict_result_table;
  for (const auto& result : PredictResults::all) {
    predict_result_table.emplace(result, 0);
  }

  // Predict
  auto start_time = std::chrono::system_clock::now();

  if (predict_setup_func) {
    predict_setup_func();
  }

  size_t sample_num = sample_set_msg->samples.size();
  SampleType sample;
  for (size_t i = 0; i < sample_num; i++) {
    for (int j = 0; j < sample_dim_; j++) {
      sample[j] = sample_set_msg->samples[i].position[j];
    }

    bool reachability_gt = sample_set_msg->samples[i].is_reachable;
    bool reachability_pred = predict_once_func(sample);

    if (reachability_pred) {
      if (reachability_gt) {
        predict_result_table.at(PredictResult::TrueReachable)++;
      } else {
        predict_result_table.at(PredictResult::FalseReachable)++;
      }
    } else {
      if (reachability_gt) {
        predict_result_table.at(PredictResult::FalseUnreachable)++;
      } else {
        predict_result_table.at(PredictResult::TrueUnreachable)++;
      }
    }
  }

  double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
      std::chrono::system_clock::now() - start_time).count();

  double iou = static_cast<double>(predict_result_table.at(PredictResult::TrueReachable)) / (
      predict_result_table.at(PredictResult::TrueReachable) +
      predict_result_table.at(PredictResult::FalseReachable) +
      predict_result_table.at(PredictResult::FalseUnreachable));
  ROS_INFO_STREAM("IoU: " << iou);
  for (const auto& result : PredictResults::all) {
    ROS_INFO_STREAM("  - " << std::to_string(result) << ": " << predict_result_table.at(result));
  }
  ROS_INFO_STREAM("Predict duration: " << duration << " [ms] (predict-one: " << duration / sample_num <<" [ms])");
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::setupSVMParam()
{
  svm_param_.svm_type = contain_unreachable_sample_ ? NU_SVC : ONE_CLASS;
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
    slice_origin_.rotation() =
        (Eigen::AngleAxisd(slice_roll_manager_->value(), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(slice_pitch_manager_->value(),  Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(slice_yaw_manager_->value(), Eigen::Vector3d::UnitZ())).toRotationMatrix().transpose();

    slice_roll_manager_->update();
    slice_pitch_manager_->update();
    slice_yaw_manager_->update();
    slice_updated_ = true;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::setupGridMap()
{
  // Calculate min/max position with margin
  SampleType sample_min_with_margin = sample_min_;
  SampleType sample_max_with_margin = sample_max_;
  SampleType sample_range = sample_max_ - sample_min_;
  {
    sample_min_with_margin -= config_.grid_map_margin_ratio * sample_range;
    sample_max_with_margin += config_.grid_map_margin_ratio * sample_range;
    sample_range = sample_max_with_margin - sample_min_with_margin;
  }

  // Create grid map
  {
    grid_map_ = std::make_shared<grid_map::GridMap>(std::vector<std::string>{"svm_value"});

    SampleType sample_center = (sample_min_ + sample_max_) / 2;
    grid_map_->setFrameId("world");
    grid_map_->setGeometry(grid_map::Length(sample_range[0], sample_range[1]),
                           config_.grid_map_resolution,
                           grid_map::Position(sample_center[0], sample_center[1]));
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::loadSampleSet(const std::string& bag_path)
{
  // Load ROS bag
  {
    ROS_INFO_STREAM("Load sample set from " << bag_path);

    differentiable_rmap::RmapSampleSet::ConstPtr sample_set_msg =
        loadBag<differentiable_rmap::RmapSampleSet>(bag_path);

    if (sample_set_msg->type != static_cast<size_t>(SamplingSpaceType)) {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[RmapTraining::loadSampleSet] SamplingSpace does not match with message: {} != {}",
          sample_set_msg->type, static_cast<size_t>(SamplingSpaceType));
    }

    size_t sample_num = sample_set_msg->samples.size();
    sample_list_.resize(sample_num);
    reachability_list_.resize(sample_num);
    for (size_t i = 0; i < sample_num; i++) {
      for (int j = 0; j < sample_dim_; j++) {
        sample_list_[i][j] = sample_set_msg->samples[i].position[j];
      }
      reachability_list_[i] = sample_set_msg->samples[i].is_reachable;
    }
    for (int i = 0; i < sample_dim_; i++) {
      sample_min_[i] = sample_set_msg->min[i];
      sample_max_[i] = sample_set_msg->max[i];
    }
  }

  // Publish cloud
  {
    std_msgs::Header header_msg;
    header_msg.frame_id = "world";
    header_msg.stamp = ros::Time::now();

    sensor_msgs::PointCloud reachable_cloud_msg;
    sensor_msgs::PointCloud unreachable_cloud_msg;
    reachable_cloud_msg.header = header_msg;
    unreachable_cloud_msg.header = header_msg;
    for (size_t i = 0; i < sample_list_.size(); i++) {
      if (reachability_list_[i]) {
        reachable_cloud_msg.points.push_back(
            OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample_list_[i])));
      } else {
        unreachable_cloud_msg.points.push_back(
            OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample_list_[i])));
      }
    }
    reachable_cloud_pub_.publish(reachable_cloud_msg);
    unreachable_cloud_pub_.publish(unreachable_cloud_msg);
  }

  // Setup SVM problem
  {
    svm_prob_.l = sample_list_.size();
    svm_prob_.y = new double[svm_prob_.l];
    svm_prob_.x = new svm_node*[svm_prob_.l];

    all_input_nodes_ = new svm_node[(input_dim_ + 1) * svm_prob_.l];
    for (size_t i = 0; i < sample_list_.size(); i++) {
      const SampleType& sample = sample_list_[i];
      size_t idx = (input_dim_ + 1) * i;
      setInputNode<SamplingSpaceType>(&(all_input_nodes_[idx]), sampleToInput<SamplingSpaceType>(sample));
      svm_prob_.x[i] = &all_input_nodes_[idx];
      svm_prob_.y[i] = reachability_list_[i] ? 1 : -1;
    }
  }

  // Check unreachable samples are contained
  contain_unreachable_sample_ = false;
  for (bool reachability : reachability_list_) {
    if (!reachability) {
      contain_unreachable_sample_ = true;
      break;
    }
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::loadSVM()
{
  ROS_INFO_STREAM("Load SVM model from " << svm_path_);
  svm_mo_ = svm_load_model(svm_path_.c_str());

  if constexpr (!use_libsvm_prediction_) {
      int num_sv = svm_mo_->l;
      svm_coeff_vec_.resize(num_sv);
      svm_sv_mat_.resize(input_dim_, num_sv);
      setSVMPredictionMat<SamplingSpaceType>(svm_coeff_vec_, svm_sv_mat_, svm_mo_);
    }

  train_updated_ = true;
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

    if constexpr (!use_libsvm_prediction_) {
        int num_sv = svm_mo_->l;
        svm_coeff_vec_.resize(num_sv);
        svm_sv_mat_.resize(input_dim_, num_sv);
        setSVMPredictionMat<SamplingSpaceType>(svm_coeff_vec_, svm_sv_mat_, svm_mo_);
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
    // Save is fast compared with other process
    // ROS_INFO_STREAM("SVM save duration: " << duration << " [ms]");
  }

  train_updated_ = true;
}

template <SamplingSpace SamplingSpaceType>
double RmapTraining<SamplingSpaceType>::calcSVMValue(
    const SampleType& sample) const
{
  return DiffRmap::calcSVMValue<SamplingSpaceType>(
      sample,
      svm_mo_->param,
      svm_mo_,
      svm_coeff_vec_,
      svm_sv_mat_);
}

template <SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> RmapTraining<SamplingSpaceType>::calcSVMGrad(
    const SampleType& sample) const
{
  return DiffRmap::calcSVMGrad<SamplingSpaceType>(
          sample,
          svm_mo_->param,
          svm_mo_,
          svm_coeff_vec_,
          svm_sv_mat_);
}

template <SamplingSpace SamplingSpaceType>
Vel<SamplingSpaceType> RmapTraining<SamplingSpaceType>::calcSVMGradWithVel(
    const SampleType& sample) const
{
  return sampleToVelMat<SamplingSpaceType>(sample) * calcSVMGrad(sample);
}

template <SamplingSpace SamplingSpaceType>
bool RmapTraining<SamplingSpaceType>::predictOnceSVM(
    const SampleType& sample,
    double svm_thre) const
{
  return calcSVMValue(sample) >= svm_thre;
}

template <SamplingSpace SamplingSpaceType>
bool RmapTraining<SamplingSpaceType>::predictOnceOCNN(
    const SampleType& sample,
    double dist_ratio_thre) const
{
  return oneClassNearestNeighbor<sample_dim_>(sample, dist_ratio_thre, sample_list_);
}

template <SamplingSpace SamplingSpaceType>
bool RmapTraining<SamplingSpaceType>::predictOnceKNN(
    const SampleType& sample,
    size_t K) const
{
  return kNearestNeighbor<sample_dim_>(sample, K, sample_list_, reachability_list_);
}

template <SamplingSpace SamplingSpaceType>
bool RmapTraining<SamplingSpaceType>::predictOnceConvex(
    const SampleType& sample) const
{
  if constexpr (SamplingSpaceType == SamplingSpace::R2) {
      return convex_inside_class_->classify(sample);
    } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[predictOnceConvex] Unsupported SamplingSpace: {}", std::to_string(SamplingSpaceType));
  }

  return false;
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::predictOnSlicePlane()
{
  // Predict
  {
    auto start_time = std::chrono::system_clock::now();

    svm_node input_node[input_dim_ + 1];

    if constexpr (use_libsvm_prediction_) {
        setInputNode<SamplingSpaceType>(input_node, InputType::Zero());
      }

    size_t grid_idx = 0;
    SampleType origin_sample = poseToSample<SamplingSpaceType>(slice_origin_);
    for (grid_map::GridMapIterator it(*grid_map_); !it.isPastEnd(); ++it) {
      grid_map::Position pos;
      grid_map_->getPosition(*it, pos);

      SampleType sample = origin_sample;
      sample.x() = pos.x();
      if constexpr (sample_dim_ > 1) {
          sample.y() = pos.y();
        }

      double svm_value;
      if constexpr (use_libsvm_prediction_) {
          setInputNodeOnlyValue<SamplingSpaceType>(input_node, sampleToInput<SamplingSpaceType>(sample));
          svm_predict_values(svm_mo_, input_node, &svm_value);
        } else {
        svm_value = calcSVMValue(sample);
      }
      grid_map_->at("svm_value", *it) = config_.grid_map_height_scale * svm_value;

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

  sensor_msgs::PointCloud reachable_cloud_msg;
  sensor_msgs::PointCloud unreachable_cloud_msg;
  reachable_cloud_msg.header = header_msg;
  unreachable_cloud_msg.header = header_msg;
  for (size_t i = 0; i < sample_list_.size(); i++) {
    const SampleType& sample = sample_list_[i];

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
        double origin_z = slice_origin_.translation().z();
        Eigen::Quaterniond origin_quat(slice_origin_.rotation().transpose());
        Sample<SamplingSpace::SO3> sample_so3 =
            sample.template tail<sampleDim<SamplingSpace::SO3>()>();
        Eigen::Quaterniond sample_quat(sample_so3.w(), sample_so3.x(), sample_so3.y(), sample_so3.z());
        double theta_diff = std::fabs(Eigen::AngleAxisd(origin_quat.inverse() * sample_quat).angle());
        if (std::fabs(sample.z() - origin_z) > config_.slice_se3_z_thre ||
            theta_diff > mc_rtc::constants::toRad(config_.slice_se3_theta_thre)) {
          continue;
        }
      }

    if (reachability_list_[i]) {
      reachable_cloud_msg.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
    } else {
      unreachable_cloud_msg.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
    }
  }
  sliced_reachable_cloud_pub_.publish(reachable_cloud_msg);
  sliced_unreachable_cloud_pub_.publish(unreachable_cloud_msg);
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
  xy_plane_marker.color = OmgCore::toColorRGBAMsg({1.0, 1.0, 1.0, 1.0});
  xy_plane_marker.scale.x = 100.0;
  xy_plane_marker.scale.y = 100.0;
  xy_plane_marker.scale.z = plane_thickness;
  xy_plane_marker.pose = OmgCore::toPoseMsg(
      sva::PTransformd(Eigen::Vector3d(0, 0, svm_thre_manager_->value() - 0.5 * plane_thickness)));
  marker_arr_msg.markers.push_back(xy_plane_marker);

  marker_arr_pub_.publish(marker_arr_msg);
}

template <SamplingSpace SamplingSpaceType>
bool RmapTraining<SamplingSpaceType>::evaluateCallback(
    std_srvs::Empty::Request& req,
    std_srvs::Empty::Response& res)
{
  ROS_INFO("==== SVM ====");
  for (double svm_thre : config_.eval_svm_thre_list) {
    ROS_INFO_STREAM("- svm_thre: " << svm_thre);
    evaluateAccuracy(
        config_.eval_bag_path,
        std::bind(&RmapTraining<SamplingSpaceType>::predictOnceSVM,
                  this, std::placeholders::_1, svm_thre));
  }

  if (!contain_unreachable_sample_) {
    ROS_INFO("==== OCNN ====");
    for (double dist_ratio_thre : config_.ocnn_dist_ratio_thre_list) {
      ROS_INFO_STREAM("- dist_ratio_thre: " << dist_ratio_thre);
      evaluateAccuracy(
          config_.eval_bag_path,
          std::bind(&RmapTraining<SamplingSpaceType>::predictOnceOCNN,
                    this, std::placeholders::_1, dist_ratio_thre));
    }

    if constexpr (SamplingSpaceType == SamplingSpace::R2) {
        ROS_INFO("==== Convex ====");
        evaluateAccuracy(
            config_.eval_bag_path,
            std::bind(&RmapTraining<SamplingSpaceType>::predictOnceConvex,
                      this, std::placeholders::_1),
            [this]() {
              this->convex_inside_class_ =
                  std::make_shared<ConvexInsideClassification>(this->sample_list_);
            });
      }
  }

  if (contain_unreachable_sample_) {
    ROS_INFO("==== KNN ====");
    for (size_t K : config_.knn_K_list) {
      ROS_INFO_STREAM("- K: " << K);
      evaluateAccuracy(
          config_.eval_bag_path,
          std::bind(&RmapTraining<SamplingSpaceType>::predictOnceKNN,
                    this, std::placeholders::_1, K));
    }
  }

  return true;
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::testCalcSVMValue(
    double& svm_value_libsvm,
    double& svm_value_eigen,
    const SampleType& sample) const
{
  svm_node input_node[input_dim_ + 1];
  setInputNode<SamplingSpaceType>(input_node, sampleToInput<SamplingSpaceType>(sample));
  svm_predict_values(svm_mo_, input_node, &svm_value_libsvm);

  svm_value_eigen = calcSVMValue(sample);
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::testCalcSVMGrad(
    Eigen::Ref<Vel<SamplingSpaceType>> svm_grad_analytical,
    Eigen::Ref<Vel<SamplingSpaceType>> svm_grad_numerical,
    const SampleType& sample) const
{
  svm_grad_analytical = calcSVMGradWithVel(sample);

  double eps = 1e-6;
  for (int i = 0; i < velDim<SamplingSpaceType>(); i++) {
    Vel<SamplingSpaceType> vel = eps * Vel<SamplingSpaceType>::Unit(i);
    Sample<SamplingSpaceType> sample_plus = sample;
    integrateVelToSample<SamplingSpaceType>(sample_plus, vel);
    Sample<SamplingSpaceType> sample_minus = sample;
    integrateVelToSample<SamplingSpaceType>(sample_minus, -vel);

    svm_grad_numerical[i] =
        (calcSVMValue(sample_plus) - calcSVMValue(sample_minus)) / (2 * eps);
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::testCalcSVMGradRel(
    Eigen::Ref<Vel<SamplingSpaceType>> pre_grad_analytical,
    Eigen::Ref<Vel<SamplingSpaceType>> suc_grad_analytical,
    Eigen::Ref<Vel<SamplingSpaceType>> pre_grad_numerical,
    Eigen::Ref<Vel<SamplingSpaceType>> suc_grad_numerical,
    const SampleType& pre_sample,
    const SampleType& suc_sample) const
{
  SampleType rel_sample = relSample<SamplingSpaceType>(pre_sample, suc_sample);
  const VelType& svm_grad = calcSVMGradWithVel(rel_sample);

  pre_grad_analytical = relSampleToSampleMat<SamplingSpaceType>(pre_sample, suc_sample, false).transpose() * svm_grad;
  suc_grad_analytical = relSampleToSampleMat<SamplingSpaceType>(pre_sample, suc_sample, true).transpose() * svm_grad;

  double eps = 1e-6;
  for (bool wrt_suc : {false, true}) {
    for (int i = 0; i < velDim<SamplingSpaceType>(); i++) {
      Vel<SamplingSpaceType> vel = eps * Vel<SamplingSpaceType>::Unit(i);
      Sample<SamplingSpaceType> sample_plus = wrt_suc ? suc_sample : pre_sample;
      integrateVelToSample<SamplingSpaceType>(sample_plus, vel);
      Sample<SamplingSpaceType> sample_minus = wrt_suc ? suc_sample : pre_sample;
      integrateVelToSample<SamplingSpaceType>(sample_minus, -vel);

      SampleType rel_sample_plus;
      SampleType rel_sample_minus;
      if (wrt_suc) {
        rel_sample_plus = relSample<SamplingSpaceType>(pre_sample, sample_plus);
        rel_sample_minus = relSample<SamplingSpaceType>(pre_sample, sample_minus);
      } else {
        rel_sample_plus = relSample<SamplingSpaceType>(sample_plus, suc_sample);
        rel_sample_minus = relSample<SamplingSpaceType>(sample_minus, suc_sample);
      }

      double numerical_value =
          (calcSVMValue(rel_sample_plus) - calcSVMValue(rel_sample_minus)) / (2 * eps);
      if (wrt_suc) {
        suc_grad_numerical[i] = numerical_value;
      } else {
        pre_grad_numerical[i] = numerical_value;
      }
    }
  }
}

std::shared_ptr<RmapTrainingBase> DiffRmap::createRmapTraining(
    SamplingSpace sampling_space,
    const std::string& bag_path,
    const std::string& svm_path,
    bool load_svm)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapTraining<SamplingSpace::R2>>(bag_path, svm_path, load_svm);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapTraining<SamplingSpace::SO2>>(bag_path, svm_path, load_svm);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapTraining<SamplingSpace::SE2>>(bag_path, svm_path, load_svm);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapTraining<SamplingSpace::R3>>(bag_path, svm_path, load_svm);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapTraining<SamplingSpace::SO3>>(bag_path, svm_path, load_svm);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapTraining<SamplingSpace::SE3>>(bag_path, svm_path, load_svm);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapTraining] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapTraining<SamplingSpace::R2>;
template class RmapTraining<SamplingSpace::SO2>;
template class RmapTraining<SamplingSpace::SE2>;
template class RmapTraining<SamplingSpace::R3>;
template class RmapTraining<SamplingSpace::SO3>;
template class RmapTraining<SamplingSpace::SE3>;
