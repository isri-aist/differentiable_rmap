/* Author: Masaki Murooka */

#include <map>
#include <unordered_set>
#include <chrono>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud.h>
#include <differentiable_rmap/RmapSampleSet.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapTraining.h>
#include <differentiable_rmap/SVMUtils.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapTraining<SamplingSpaceType>::RmapTraining(const std::string& bag_path,
                                              const std::string& svm_path):
    svm_loaded_(bag_path.empty()),
    svm_path_(svm_path)
{
  // Setup ROS
  rmap_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("rmap_cloud", 1, true);
  grid_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);

  // Load
  if (svm_loaded_) {
    loadSVM();
  } else {
    loadBag(bag_path);
  }

  // Setup
  setupSVMParam();
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
void RmapTraining<SamplingSpaceType>::run()
{
  if (!svm_loaded_) {
    train();
  }

  predict();
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
void RmapTraining<SamplingSpaceType>::train()
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
    // \todo THis causes SEGV
    // svm_save_model(svm_path_.c_str(), svm_mo_);

    double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now() - start_time).count();
    ROS_INFO_STREAM("SVM save duration: " << duration << " [ms]");
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::predict()
{
  // Predict
  {
    auto start_time = std::chrono::system_clock::now();

    if constexpr (use_libsvm_prediction_) {
        setInputNode<SamplingSpaceType>(input_node_, InputVector::Zero());
      }

    size_t grid_idx = 0;
    SampleVector identity_sample = poseToSample<SamplingSpaceType>(sva::PTransformd::Identity());
    for (grid_map::GridMapIterator it(*grid_map_); !it.isPastEnd(); ++it) {
      grid_map::Position pos;
      grid_map_->getPosition(*it, pos);

      SampleVector sample = identity_sample;
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
    // publish is fast compared with other process
    // ROS_INFO_STREAM("SVM publish duration: " << duration << " [ms]");
  }
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
