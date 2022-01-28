/* Author: Masaki Murooka */

#pragma once

#include <set>


namespace DiffRmap
{
/** \brief Classification of prediction result. */
enum class PredictResult
{
  TrueReachable = 0,
  TrueUnreachable,
  FalseReachable,
  FalseUnreachable
};

namespace PredictResults
{
//! All classifications of prediction result
std::set<PredictResult> all = {
  PredictResult::TrueReachable, PredictResult::TrueUnreachable,
  PredictResult::FalseReachable, PredictResult::FalseUnreachable
};
}
}

namespace std
{
using DiffRmap::PredictResult;

inline string to_string(PredictResult result)
{
  if (result == PredictResult::TrueReachable) {
    return std::string("TrueReachable");
  } else if (result == PredictResult::TrueUnreachable) {
    return std::string("TrueUnreachable");
  } else if (result == PredictResult::FalseReachable) {
    return std::string("FalseReachable");
  } else if (result == PredictResult::FalseUnreachable) {
    return std::string("FalseUnreachable");
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[to_string] Unsupported PredictResult: {}", static_cast<int>(result));
  }
}
}
