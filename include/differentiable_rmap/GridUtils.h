/* Author: Masaki Murooka */

#pragma once

#include <sstream>
#include <algorithm>
#include <functional>

#include <mc_rtc/logging.h>

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Get dimension of grid.
    \tparam SamplingSpaceType sampling space
*/
template <SamplingSpace SamplingSpaceType>
constexpr int gridDim()
{
  return velDim<SamplingSpaceType>();
}

/*! \brief Type of grid position. */
template <SamplingSpace SamplingSpaceType>
using GridPos = Eigen::Matrix<double, gridDim<SamplingSpaceType>(), 1>;

/*! \brief Type of grid indices. */
template <SamplingSpace SamplingSpaceType>
using GridIdxsType = Eigen::Matrix<int, gridDim<SamplingSpaceType>(), 1>;

/*! \brief Type of function to be called for each grid. */
template <SamplingSpace SamplingSpaceType>
using GridFuncType = std::function<void(int, const GridPos<SamplingSpaceType>&)>;

/** \brief Calculate grid index.
    \tparam DivideIdxsType type of divide_idxs
    \tparam DivideNumsType type of divide_nums
    \param divide_idxs indices of grid divisions
    \param divide_nums number of grid divisions (number of vertices is divide_nums + 1)
*/
template <class DivideIdxsType, class DivideNumsType>
int calcGridIdx(const DivideIdxsType& divide_idxs,
                const DivideNumsType& divide_nums);

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
                            const DivideNumsType& divide_nums);

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
                            const DivideNumsType& divide_nums);

/** \brief Convert grid position to sample.
    \tparam SamplingSpaceType sampling space
    \param grid pos grid position
    \return sample
*/
template <SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> gridPosToSample(const GridPos<SamplingSpaceType>& grid_pos);

/** \brief Convert min sample to min grid position.
    \tparam SamplingSpaceType sampling space
    \param sample_min min sample
    \return min grid position
*/
template <SamplingSpace SamplingSpaceType>
GridPos<SamplingSpaceType> getGridPosMin(const Sample<SamplingSpaceType>& sample_min);

/** \brief Convert max sample to max grid position.
    \tparam SamplingSpaceType sampling space
    \param sample_max max sample
    \return max grid position
*/
template <SamplingSpace SamplingSpaceType>
GridPos<SamplingSpaceType> getGridPosMax(const Sample<SamplingSpaceType>& sample_max);

/** \brief Convert min/max sample to range of grid position.
    \tparam SamplingSpaceType sampling space
    \param sample_min min sample
    \param sample_max max sample
    \return range of grid position
*/
template <SamplingSpace SamplingSpaceType>
GridPos<SamplingSpaceType> getGridPosRange(const Sample<SamplingSpaceType>& sample_min,
                                           const Sample<SamplingSpaceType>& sample_max);

/** \brief Loop grid and call function for each grid.
    \tparam SamplingSpaceType sampling space
    \tparam DivideNumsType type of divide_nums
    \param divide_nums number of grid divisions (number of vertices is divide_nums + 1)
    \param grid_pos_min min position of grid
    \param grid_pos_range position range of grid
    \param func function to be called for each grid
    \param update_dims dimensions to update indices (empty to update all dimensions)
    \param default_divide_idxs default indices of grid divisions (used for non-updated indices according to update_dims)
*/
template <SamplingSpace SamplingSpaceType, class DivideNumsType>
void loopGrid(
    const DivideNumsType& divide_nums,
    const GridPos<SamplingSpaceType>& grid_pos_min,
    const GridPos<SamplingSpaceType>& grid_pos_range,
    const GridFuncType<SamplingSpaceType>& func,
    const std::vector<int>& update_dims = {},
    const GridIdxsType<SamplingSpaceType>& default_divide_idxs = GridIdxsType<SamplingSpaceType>::Zero());

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
    const Eigen::Vector3d& default_scale = Eigen::Vector3d::Constant(0.01));
}

// See method 3 in https://www.codeproject.com/Articles/48575/How-to-Define-a-Template-Class-in-a-h-File-and-Imp
#include <differentiable_rmap/GridUtils.hpp>
