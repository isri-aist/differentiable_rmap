/* Author: Masaki Murooka */

#include <limits>

#include <mc_rtc/logging.h>


namespace DiffRmap
{
template <size_t N>
bool kNearestNeighbor(
    const Eigen::Matrix<double, N, 1>& test_sample,
    size_t K,
    const std::vector<Eigen::Matrix<double, N, 1>>& train_sample_list,
    const std::vector<bool>& class_list)
{
  // Set idx_dist_list which is top-K nearest samples and distances
  std::vector<std::pair<int, double>> idx_dist_list(
      K, std::make_pair<int, double>(-1, std::numeric_limits<double>::max()));
  for (size_t i = 0; i < train_sample_list.size(); i++) {
    double dist = (train_sample_list[i] - test_sample).squaredNorm();
    for (size_t j = 0; j < K; j++) {
      if (dist < idx_dist_list[j].second) {
        for (size_t k = K - 1; k > j; k--) {
          idx_dist_list[k] = idx_dist_list[k - 1];
        }
        idx_dist_list[j].first = static_cast<int>(i);
        idx_dist_list[j].second = dist;
        break;
      }
    }
  }

  // Determine class by majority vote of top-K samples
  int positive_num = 0;
  for (const auto& idx_dist : idx_dist_list) {
    if (idx_dist.first < 0) {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[kNearestNeighbor] idx_dist_list contains uninitialized elements.");
    }
    if (class_list[idx_dist.first]) {
      positive_num++;
    } else {
      positive_num--;
    }
  }
  if (positive_num == 0) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[kNearestNeighbor] Numbers of nearest points of positive and negative are the same.");
  }
  return positive_num > 0;
}
}
