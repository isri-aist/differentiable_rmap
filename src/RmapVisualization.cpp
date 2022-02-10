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
    const std::string& sample_bag_path,
    const std::string& svm_path)
{
  // Setup ROS
  marker_arr_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_arr", 1, true);
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
void RmapVisualization<SamplingSpaceType>::setup(const std::string& grid_bag_path,
                                                 bool load_grid)
{
  if (load_grid) {
    loadGridSet(grid_bag_path);
  } else {
    dumpGridSet(grid_bag_path);
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::runOnce()
{
  updateSliceOrigin();

  publishMarkerArray();
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::runLoop(const std::string& grid_bag_path,
                                                   bool load_grid)
{
  setup(grid_bag_path, load_grid);

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
void RmapVisualization<SamplingSpaceType>::loadGridSet(
    const std::string& grid_bag_path)
{
  ROS_INFO_STREAM("Load grid set from " << grid_bag_path);

  grid_set_msg_ = *loadBag<differentiable_rmap::RmapGridSet>(grid_bag_path);

  if (grid_set_msg_.type != static_cast<size_t>(SamplingSpaceType)) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "SamplingSpace does not match with message: {} != {}",
        grid_set_msg_.type, static_cast<size_t>(SamplingSpaceType));
  }

  for (int i = 0; i < sample_dim_; i++) {
    sample_min_[i] = grid_set_msg_.min[i];
    sample_max_[i] = grid_set_msg_.max[i];
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::dumpGridSet(
    const std::string& grid_bag_path)
{
  // Set number of division
  const GridPosType& grid_pos_range = getGridPosRange<SamplingSpaceType>(sample_min_, sample_max_);
  GridIdxs<SamplingSpaceType> divide_nums;
  GridPosType resolution;
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
  divide_nums = (grid_pos_range.array() / resolution.array()).ceil().template cast<int>().max(1);

  // Set grid set message
  const GridPosType& grid_pos_min = getGridPosMin<SamplingSpaceType>(sample_min_);
  const GridPosType& grid_pos_max = getGridPosMax<SamplingSpaceType>(sample_max_);
  grid_set_msg_.type = static_cast<size_t>(SamplingSpaceType);
  grid_set_msg_.divide_nums.resize(grid_dim_);
  grid_set_msg_.min.resize(grid_dim_);
  grid_set_msg_.max.resize(grid_dim_);
  int total_grid_num = 1;
  for (int i = 0; i < grid_dim_; i++) {
    grid_set_msg_.divide_nums[i] = divide_nums[i];
    grid_set_msg_.min[i] = grid_pos_min[i];
    grid_set_msg_.max[i] = grid_pos_max[i];
    total_grid_num *= (divide_nums[i] + 1);
  }
  grid_set_msg_.values.resize(total_grid_num);

  // Set grid value
  ROS_INFO_STREAM("Total grid num is " << total_grid_num);
  auto start_time = std::chrono::system_clock::now();
  loopGrid<SamplingSpaceType>(
      divide_nums,
      grid_pos_min,
      grid_pos_range,
      [&](int grid_idx, const GridPosType& grid_pos) {
        if (total_grid_num > 1e3 && grid_idx % static_cast<int>(total_grid_num / 100.0) == 0) {
          ROS_INFO_STREAM("Loop grid " << grid_idx << " / " << total_grid_num << ", grid_pos: " << grid_pos.transpose());
        }
        grid_set_msg_.values[grid_idx] = calcSVMValue<SamplingSpaceType>(
            gridPosToSample<SamplingSpaceType>(grid_pos),
            svm_mo_->param,
            svm_mo_,
            svm_coeff_vec_,
            svm_sv_mat_);
      });
  double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
      std::chrono::system_clock::now() - start_time).count();
  ROS_INFO_STREAM("SVM predict duration: " << duration << " [ms] (predict-one: " <<
                  duration / total_grid_num <<" [ms])");

  // Dump to ROS bag
  rosbag::Bag bag(grid_bag_path, rosbag::bagmode::Write);
  bag.write("/rmap_grid_set", ros::Time::now(), grid_set_msg_);
  ROS_INFO_STREAM("Dump grid set to " << grid_bag_path);
}

template <SamplingSpace SamplingSpaceType>
void RmapVisualization<SamplingSpaceType>::updateSliceOrigin()
{
  if (!(slice_roll_manager_->hasNewValue() ||
        slice_pitch_manager_->hasNewValue() ||
        slice_yaw_manager_->hasNewValue())) {
    return;
  }

  slice_origin_.rotation() =
      (Eigen::AngleAxisd(slice_roll_manager_->value(), Eigen::Vector3d::UnitX())
       * Eigen::AngleAxisd(slice_pitch_manager_->value(),  Eigen::Vector3d::UnitY())
       * Eigen::AngleAxisd(slice_yaw_manager_->value(), Eigen::Vector3d::UnitZ())).toRotationMatrix().transpose();

  slice_roll_manager_->update();
  slice_pitch_manager_->update();
  slice_yaw_manager_->update();
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

  const GridPos<SamplingSpaceType>& grid_pos_min = getGridPosMin<SamplingSpaceType>(sample_min_);
  const GridPos<SamplingSpaceType>& grid_pos_range = getGridPosRange<SamplingSpaceType>(sample_min_, sample_max_);

  // Reachable grids marker
  {
    visualization_msgs::Marker grids_marker;
    grids_marker.header = header_msg;
    grids_marker.ns = "reachable_grids";
    grids_marker.id = marker_arr_msg.markers.size();
    grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
    grids_marker.color = OmgCore::toColorRGBAMsg(config_.grid_color);
    grids_marker.scale = OmgCore::toVector3Msg(
        calcGridCubeScale<SamplingSpaceType>(grid_set_msg_.divide_nums, sample_max_ - sample_min_));
    grids_marker.pose = OmgCore::toPoseMsg(sva::PTransformd::Identity());
    loopGrid<SamplingSpaceType>(
        grid_set_msg_.divide_nums,
        grid_pos_min,
        grid_pos_range,
        [&](int grid_idx, const GridPosType& grid_pos) {
          if (grid_set_msg_.values[grid_idx] > config_.svm_thre) {
            grids_marker.points.push_back(OmgCore::toPointMsg(
                sampleToCloudPos<SamplingSpaceType>(gridPosToSample<SamplingSpaceType>(grid_pos))));
          }
        });
    marker_arr_msg.markers.push_back(grids_marker);
  }

  // Sliced reachable grids marker
  {
    visualization_msgs::Marker grids_marker;
    grids_marker.header = header_msg;
    grids_marker.ns = "reachable_grids_sliced";
    grids_marker.id = marker_arr_msg.markers.size();
    grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
    grids_marker.color = OmgCore::toColorRGBAMsg(config_.grid_color);
    grids_marker.scale = OmgCore::toVector3Msg(
        calcGridCubeScale<SamplingSpaceType>(grid_set_msg_.divide_nums, sample_max_ - sample_min_));
    grids_marker.pose = OmgCore::toPoseMsg(sva::PTransformd::Identity());

    const SampleType& slice_sample = poseToSample<SamplingSpaceType>(slice_origin_);
    GridIdxs<SamplingSpaceType> slice_divide_idxs;
    gridDivideRatiosToIdxs(
        slice_divide_idxs,
        (sampleToGridPos<SamplingSpaceType>(slice_sample) - grid_pos_min).array() / grid_pos_range.array(),
        grid_set_msg_.divide_nums);
    std::vector<int> slice_update_dims = {};
    if (SamplingSpaceType == SamplingSpace::R2 ||
        SamplingSpaceType == SamplingSpace::SE2) {
      slice_update_dims = {0, 1};
    } else if (SamplingSpaceType == SamplingSpace::R3 ||
               SamplingSpaceType == SamplingSpace::SE3) {
      slice_update_dims = {0, 1, 2};
    }

    loopGrid<SamplingSpaceType>(
        grid_set_msg_.divide_nums,
        grid_pos_min,
        grid_pos_range,
        [&](int grid_idx, const GridPosType& grid_pos) {
          if (grid_set_msg_.values[grid_idx] > config_.svm_thre) {
            Eigen::Vector3d pos = sampleToCloudPos<SamplingSpaceType>(gridPosToSample<SamplingSpaceType>(grid_pos));
            if (SamplingSpaceType == SamplingSpace::R2 ||
                SamplingSpaceType == SamplingSpace::SO2 ||
                SamplingSpaceType == SamplingSpace::SE2) {
              pos.z() = 0;
            }
            grids_marker.points.push_back(OmgCore::toPointMsg(pos));
          }
        },
        slice_update_dims,
        slice_divide_idxs);
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
