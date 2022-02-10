/* Author: Masaki Murooka */


namespace
{
/** \brief Get string from vector.
    \param vec vector
    \param delim delimiter string
*/
template <class T>
std::string vecToStr(const T& vec,
                     const std::string delim = ", ")
{
  std::string str = "";
  for (int i = 0; i < vec.size(); i++) {
    str += std::to_string(vec[i]);
    if (i < vec.size() - 1) {
      str += delim;
    }
  }
  return str;
}

/** \brief Update indices of grid divisions.
    \tparam DivideIdxsType type of divide_idxs
    \tparam DivideNumsType type of divide_nums
    \param[out] divide_idxs indices of grid divisions
    \param[in] divide_nums number of grid divisions (number of vertices is divide_nums + 1)
    \param[in] update_dims dimensions to update indices (empty to update all dimensions)
*/
template <class DivideIdxsType, class DivideNumsType>
bool updateGridDivideIdxs(DivideIdxsType& divide_idxs,
                          const DivideNumsType& divide_nums,
                          const std::vector<int>& update_dims)
{
  size_t i_end = update_dims.empty() ? divide_idxs.size() : update_dims.size();
  for (size_t i = 0; i < i_end; i++) {
    size_t k = update_dims.empty() ? i : update_dims[i];
    divide_idxs[k]++;
    if (divide_idxs[k] == divide_nums[k] + 1) {
      // If there is a carry, the current digit value is set to zero
      divide_idxs[k] = 0;
      // If there is a carry at the top, exit the outer loop
      if (i == i_end - 1) {
        return true;
      }
    } else {
      // If there is no carry, it will end
      break;
    }
  }
  return false;
}
}

namespace DiffRmap
{
template <class DivideIdxsType, class DivideNumsType>
int calcGridIdx(const DivideIdxsType& divide_idxs,
                const DivideNumsType& divide_nums)
{
  int grid_idx = 0;
  int stride = 1;
  for (size_t i = 0; i < divide_idxs.size(); i++) {
    grid_idx += stride * divide_idxs[i];
    stride *= (divide_nums[i] + 1);
  }
  return grid_idx;
}

template <class DivideRatiosType, class DivideIdxsType, class DivideNumsType>
void gridDivideIdxsToRatios(DivideRatiosType& divide_ratios,
                            const DivideIdxsType& divide_idxs,
                            const DivideNumsType& divide_nums)
{
  for (int i = 0; i < divide_ratios.size(); i++) {
    if (divide_nums[i] == 0) {
      divide_ratios[i] = 0.5;
    } else {
      divide_ratios[i] = std::min(std::max(static_cast<double>(divide_idxs[i]) / divide_nums[i], 0.0), 1.0);
    }
  }
}

template <class DivideIdxsType, class DivideRatiosType, class DivideNumsType>
void gridDivideRatiosToIdxs(DivideIdxsType& divide_idxs,
                            const DivideRatiosType& divide_ratios,
                            const DivideNumsType& divide_nums)
{
  for (int i = 0; i < divide_idxs.size(); i++) {
    divide_idxs[i] = std::min(std::max(
        static_cast<int>(std::round(divide_ratios[i] * divide_nums[i])), 0), divide_nums[i]);
  }
}

template <SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> gridPosToSample(const GridPos<SamplingSpaceType>& grid_pos)
{
  return grid_pos;
}

template <>
inline Sample<SamplingSpace::SO3> gridPosToSample<SamplingSpace::SO3>(
    const GridPos<SamplingSpace::SO3>& grid_pos)
{
  Sample<SamplingSpace::SO3> sample;
  sample <<
      Eigen::Quaterniond(
          Eigen::AngleAxisd(grid_pos.x(), Eigen::Vector3d::UnitX())
          * Eigen::AngleAxisd(grid_pos.y(),  Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(grid_pos.z(), Eigen::Vector3d::UnitZ())).coeffs();
  return sample;
}

template <>
inline Sample<SamplingSpace::SE3> gridPosToSample<SamplingSpace::SE3>(
    const GridPos<SamplingSpace::SE3>& grid_pos)
{
  Sample<SamplingSpace::SE3> sample;
  sample <<
      grid_pos.head<3>(),
      Eigen::Quaterniond(
          Eigen::AngleAxisd(grid_pos.tail<3>().x(), Eigen::Vector3d::UnitX())
          * Eigen::AngleAxisd(grid_pos.tail<3>().y(),  Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(grid_pos.tail<3>().z(), Eigen::Vector3d::UnitZ())).coeffs();
  return sample;
}

template <SamplingSpace SamplingSpaceType>
GridPos<SamplingSpaceType> getGridPosMin(const Sample<SamplingSpaceType>& sample_min)
{
  return sample_min;
}

template <>
inline GridPos<SamplingSpace::SO3> getGridPosMin<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample_min)
{
  GridPos<SamplingSpace::SO3> grid_pos_min;
  grid_pos_min << 0, -M_PI, -M_PI;
  return grid_pos_min;
}

template <>
inline GridPos<SamplingSpace::SE3> getGridPosMin<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample_min)
{
  GridPos<SamplingSpace::SE3> grid_pos_min;
  grid_pos_min <<
      getGridPosMin<SamplingSpace::R3>(sample_min.head<3>()), getGridPosMin<SamplingSpace::SO3>(sample_min.tail<4>());
  return grid_pos_min;
}

template <SamplingSpace SamplingSpaceType>
GridPos<SamplingSpaceType> getGridPosMax(const Sample<SamplingSpaceType>& sample_max)
{
  return sample_max;
}

template <>
inline GridPos<SamplingSpace::SO3> getGridPosMax<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample_max)
{
  GridPos<SamplingSpace::SO3> grid_pos_max;
  grid_pos_max << M_PI, M_PI, M_PI;
  return grid_pos_max;
}

template <>
inline GridPos<SamplingSpace::SE3> getGridPosMax<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample_max)
{
  GridPos<SamplingSpace::SE3> grid_pos_max;
  grid_pos_max <<
      getGridPosMax<SamplingSpace::R3>(sample_max.head<3>()), getGridPosMax<SamplingSpace::SO3>(sample_max.tail<4>());
  return grid_pos_max;
}

template <SamplingSpace SamplingSpaceType>
GridPos<SamplingSpaceType> getGridPosRange(const Sample<SamplingSpaceType>& sample_min,
                                           const Sample<SamplingSpaceType>& sample_max)
{
  return sample_max - sample_min;
}

template <>
inline GridPos<SamplingSpace::SO3> getGridPosRange<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample_min,
    const Sample<SamplingSpace::SO3>& sample_max)
{
  return getGridPosMax<SamplingSpace::SO3>(sample_max) - getGridPosMin<SamplingSpace::SO3>(sample_min);
}

template <>
inline GridPos<SamplingSpace::SE3> getGridPosRange<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample_min,
    const Sample<SamplingSpace::SE3>& sample_max)
{
  GridPos<SamplingSpace::SE3> grid_pos_range;
  grid_pos_range <<
      getGridPosRange<SamplingSpace::R3>(sample_min.head<3>(), sample_max.head<3>()),
      getGridPosRange<SamplingSpace::SO3>(sample_min.tail<4>(), sample_max.tail<4>());
  return grid_pos_range;
}

template <SamplingSpace SamplingSpaceType, class DivideNumsType>
void loopGrid(
    const DivideNumsType& divide_nums,
    const GridPos<SamplingSpaceType>& grid_pos_min,
    const GridPos<SamplingSpaceType>& grid_pos_range,
    const GridFuncType<SamplingSpaceType>& func,
    const std::vector<int>& update_dims,
    const GridIdxsType<SamplingSpaceType>& default_divide_idxs)
{
  // Initialize divide_idxs with default value
  GridIdxsType<SamplingSpaceType> divide_idxs = GridIdxsType<SamplingSpaceType>::Zero();
  if (!update_dims.empty()) {
    for (size_t i = 0; i < divide_idxs.size(); i++) {
      if (std::find(update_dims.begin(), update_dims.end(), i) == update_dims.end()) {
        if (default_divide_idxs[i] < 0 || default_divide_idxs[i] > divide_nums[i]) {
          mc_rtc::log::error_and_throw<std::runtime_error>(
              "[loopGrid] default_divide_idxs[{}] is invalid. It should be 0 <= {} <= {}",
              i, default_divide_idxs[i], divide_nums[i]);
        }
        divide_idxs[i] = default_divide_idxs[i];
      }
    }
  }

  // Loop
  GridPos<SamplingSpaceType> divide_ratios;
  bool break_flag = false;
  do {
    gridDivideIdxsToRatios(divide_ratios, divide_idxs, divide_nums);

    // ROS_INFO_STREAM("[loopGrid] idx: " << calcGridIdx(divide_idxs, divide_nums) <<
    //                 ", divide_idxs: [" << vecToStr(divide_idxs) << "]");
    func(calcGridIdx(divide_idxs, divide_nums),
         divide_ratios.cwiseProduct(grid_pos_range) + grid_pos_min);

    break_flag = updateGridDivideIdxs(divide_idxs, divide_nums, update_dims);

  } while (!break_flag);
}

template <SamplingSpace SamplingSpaceType, class DivideNumsType>
Eigen::Vector3d calcGridCubeScale(
    const DivideNumsType& divide_nums,
    const Sample<SamplingSpaceType>& sample_range,
    double margin,
    const Eigen::Vector3d& default_scale)
{
  Eigen::Vector3d scale = default_scale;
  for (int i = 0; i < 3; i++) {
    if (sampleDim<SamplingSpaceType>() > i) {
      scale[i] = (1 + margin) * sample_range[i] / divide_nums[i];
    }
  }
  return scale;
}
}
