/* Author: Masaki Murooka */

#pragma once

#include <string>


namespace DiffRmap
{
/** \brief Limb. */
enum class Limb
{
  LeftFoot = 0,
  RightFoot,
  LeftHand,
  RightHand
};

namespace Limbs
{
//! All limbs
const std::set<Limb> all = {Limb::LeftFoot, Limb::RightFoot, Limb::LeftHand, Limb::RightHand};
//! Feet
const std::set<Limb> feet = {Limb::LeftFoot, Limb::RightFoot};
//! Hands
const std::set<Limb> hands = {Limb::LeftHand, Limb::RightHand};
}

/** \brief Convert string to limb. */
inline Limb strToLimb(const std::string& limb_str)
{
  if (limb_str == "LeftFoot") {
    return Limb::LeftFoot;
  } else if (limb_str == "RightFoot") {
    return Limb::RightFoot;
  } else if (limb_str == "LeftHand") {
    return Limb::LeftHand;
  } else if (limb_str == "RightHand") {
    return Limb::RightHand;
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[strToLimb] Unsupported Limb name: {}", limb_str);
  }
}
}

namespace std
{
using DiffRmap::Limb;

inline string to_string(Limb limb)
{
  if (limb == Limb::LeftFoot) {
    return std::string("LeftFoot");
  } else if (limb == Limb::RightFoot) {
    return std::string("RightFoot");
  } else if (limb == Limb::LeftHand) {
    return std::string("LeftHand");
  } else if (limb == Limb::RightHand) {
    return std::string("RightHand");
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[to_string] Unsupported Limb: {}", static_cast<int>(limb));
  }
}
}
