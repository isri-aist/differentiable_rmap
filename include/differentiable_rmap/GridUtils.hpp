/* Author: Masaki Murooka */


namespace DiffRmap
{
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
}
