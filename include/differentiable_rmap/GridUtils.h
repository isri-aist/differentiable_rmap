/* Author: Masaki Murooka */

#pragma once

#include <sstream>
#include <algorithm>
#include <functional>

#include <mc_rtc/logging.h>


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
}

namespace DiffRmap
{
/** \brief Calculate grid index.
    \tparam DivideIdxsType type of divide_idxs
    \tparam DivideNumsType type of divide_nums
    \param divide_idxs indices of grid divisions
    \param divide_nums number of grid divisions (number of vertices is divide_nums + 1)
*/
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

/** \brief Calculate ratios of grid divisions.
    \tparam DivideRatiosType type of divide_ratios
    \tparam DivideIdxsType type of divide_idxs
    \tparam DivideNumsType type of divide_nums
    \param[out] divide_ratios ratios of grid divisions
    \param[in] divide_idxs indices of grid divisions
    \param[in] divide_nums number of grid divisions (number of vertices is divide_nums + 1)
*/
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

/** \brief Calculate indices of grid divisions.
    \tparam DivideIdxsType type of divide_idxs
    \tparam DivideRatiosType type of divide_ratios
    \tparam DivideNumsType type of divide_nums
    \param[out] divide_idxs indices of grid divisions
    \param[in] divide_ratios ratios of grid divisions
    \param[in] divide_nums number of grid divisions (number of vertices is divide_nums + 1)
*/
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
                          const std::vector<int>& update_dims = {})
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

/** \brief Calculate cube scale for grid.
    \tparam SamplingSpaceType sampling space
    \tparam DivideNumsType type of divide_nums
    \param divide_nums number of grid divisions (number of vertices is divide_nums + 1)
    \param sample_range position range of sample
    \param margin rate of padding margin
    \param default_scale default cube scale
    \return cube scale
*/
template <SamplingSpace SamplingSpaceType, class DivideNumsType>
Eigen::Vector3d calcGridCubeScale(
    const DivideNumsType& divide_nums,
    const Sample<SamplingSpaceType>& sample_range,
    double margin = 0.0,
    const Eigen::Vector3d& default_scale = Eigen::Vector3d::Constant(0.01))
{
  Eigen::Vector3d scale = default_scale;
  for (int i = 0; i < 3; i++) {
    if (sampleDim<SamplingSpaceType>() > i) {
      scale[i] = (1 + margin) * sample_range[i] / divide_nums[i];
    }
  }
  return scale;
}

/*! \brief Type of grid indices. */
template <SamplingSpace SamplingSpaceType>
using GridIdxsType = Eigen::Matrix<int, sampleDim<SamplingSpaceType>(), 1>;

/*! \brief Type of function to be called for each grid. */
template <SamplingSpace SamplingSpaceType>
using GridFuncType = std::function<void(int, const Sample<SamplingSpaceType>&)>;

/** \brief Loop grid and call function for each grid.
    \tparam SamplingSpaceType sampling space
    \tparam DivideNumsType type of divide_nums
    \param divide_nums number of grid divisions (number of vertices is divide_nums + 1)
    \param sample_min min position of sample
    \param sample_range position range of sample
    \param func function to be called for each grid
    \param update_dims dimensions to update indices (empty to update all dimensions)
    \param default_divide_idxs default indices of grid divisions (used for non-updated indices according to update_dims)
*/
template <SamplingSpace SamplingSpaceType, class DivideNumsType>
void loopGrid(
    const DivideNumsType& divide_nums,
    const Sample<SamplingSpaceType>& sample_min,
    const Sample<SamplingSpaceType>& sample_range,
    const GridFuncType<SamplingSpaceType>& func,
    const std::vector<int>& update_dims = {},
    const GridIdxsType<SamplingSpaceType>& default_divide_idxs = GridIdxsType<SamplingSpaceType>::Zero())
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
  Sample<SamplingSpaceType> divide_ratios;
  bool break_flag = false;
  do {
    gridDivideIdxsToRatios(divide_ratios, divide_idxs, divide_nums);

    // ROS_INFO_STREAM("[loopGrid] idx: " << calcGridIdx(divide_idxs, divide_nums) <<
    //                 ", divide_idxs: [" << vecToStr(divide_idxs) << "]");
    func(calcGridIdx(divide_idxs, divide_nums),
         divide_ratios.cwiseProduct(sample_range) + sample_min);

    break_flag = updateGridDivideIdxs(divide_idxs, divide_nums, update_dims);

  } while (!break_flag);
}
}
