/* Author: Masaki Murooka */

#pragma once

#include <cmath>

namespace DiffRmap
{
/** \brief Calculate yaw angle from rotation matrix
    \param rot input rotation matrix
    \note See https://stackoverflow.com/a/33920320 for formulation
    \return yaw angle
*/
inline double calcYawAngle(const Eigen::Matrix3d & rot)
{
  return std::atan2(Eigen::Vector3d::UnitZ().dot(Eigen::Vector3d::UnitX().cross(rot.col(0))),
                    Eigen::Vector3d::UnitX().dot(rot.col(0)));
}
} // namespace DiffRmap
