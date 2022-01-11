/* Author: Masaki Murooka */

#pragma once


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
}
