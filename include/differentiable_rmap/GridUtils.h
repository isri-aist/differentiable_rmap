/* Author: Masaki Murooka */

#pragma once

#include <functional>


namespace DiffRmap
{
/** \brief Calculate ratios of grid division
    \tparam DivideRatiosType type of divide_ratios
    \tparam DivideIdxsType type of divide_idxs
    \tparam DivideNumsType type of divide_nums
    \param[out] divide_ratios ratios of grid division
    \param[in] divide_idxs indices of grid division
    \param[in] divide_nums number of grid division
*/
template <class DivideRatiosType, class DivideIdxsType, class DivideNumsType>
inline void calcGridDivideRatios(
    DivideRatiosType& divide_ratios,
    const DivideIdxsType& divide_idxs,
    const DivideNumsType& divide_nums)
{
  for (int i = 0; i < divide_ratios.size(); i++) {
    if (divide_nums[i] == 1) {
      divide_ratios[i] = 0.5;
    } else {
      divide_ratios[i] = static_cast<double>(divide_idxs[i]) / (divide_nums[i] - 1);
    }
  }
}

/** \brief Update indices of grid division
    \tparam DivideIdxsType type of divide_idxs
    \tparam DivideNumsType type of divide_nums
    \param[out] divide_idxs indices of grid division
    \param[in] divide_nums number of grid division
*/
template <class DivideIdxsType, class DivideNumsType>
inline bool updateGridDivideIdxs(
    DivideIdxsType& divide_idxs,
    const DivideNumsType& divide_nums)
{
  for (size_t i = 0; i < divide_idxs.size(); i++) {
    divide_idxs[i]++;
    if (divide_idxs[i] == divide_nums[i]) {
      // If there is a carry, the current digit value is set to zero
      divide_idxs[i] = 0;
      // If there is a carry at the top, exit the outer loop
      if (i == divide_idxs.size() - 1) {
        return true;
      }
    } else {
      // If there is no carry, it will end
      break;
    }
  }
  return false;
}

/*! \brief Type of function to be called for each grid. */
template <SamplingSpace SamplingSpaceType>
using GridFuncType = std::function<void(int, const Sample<SamplingSpaceType>&)>;

/** \brief Loop grid and call function for each grid
    \tparam SamplingSpaceType sampling space
    \tparam DivideNumsType type of divide_nums
    \param divide_nums number of grid division
    \param sample_min min position of sample
    \param sample_range position range of sample
    \param func function to be called for each grid
*/
template <SamplingSpace SamplingSpaceType, class DivideNumsType>
inline void loopGrid(const DivideNumsType& divide_nums,
                     const Sample<SamplingSpaceType>& sample_min,
                     const Sample<SamplingSpaceType>& sample_range,
                     const GridFuncType<SamplingSpaceType>& func)
{
  Sample<SamplingSpaceType> divide_ratios;
  Eigen::Matrix<int, sampleDim<SamplingSpaceType>(), 1> divide_idxs =
      Eigen::Matrix<int, sampleDim<SamplingSpaceType>(), 1>::Zero();
  int grid_idx = 0;
  bool break_flag = false;
  do {
    calcGridDivideRatios(divide_ratios, divide_idxs, divide_nums);

    func(grid_idx, divide_ratios.cwiseProduct(sample_range) + sample_min);

    break_flag = updateGridDivideIdxs(divide_idxs, divide_nums);

    grid_idx++;
  } while (!break_flag);
}
}
