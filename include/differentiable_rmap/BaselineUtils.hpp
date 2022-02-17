/* Author: Masaki Murooka */

#include <limits>
#include <set>

#include <mc_rtc/logging.h>

namespace DiffRmap
{
namespace
{
/** \brief Get nearest sample index.
    \tparam N sample dimension
    \param focused_sample focused sample
    \param sample_list sample list
    \param exclude_idx_list indices of sample list to exclude
*/
template<size_t N>
size_t getNearestSample(const Eigen::Matrix<double, N, 1> & focused_sample,
                        const std::vector<Eigen::Matrix<double, N, 1>> & sample_list,
                        const std::set<size_t> & exclude_idx_list = {})
{
  size_t nearest_idx = 0;
  double nearest_dist = std::numeric_limits<double>::max();
  for(size_t i = 0; i < sample_list.size(); i++)
  {
    if(exclude_idx_list.count(i) > 0)
    {
      continue;
    }
    double dist = (sample_list[i] - focused_sample).squaredNorm();
    if(dist < nearest_dist)
    {
      nearest_idx = i;
      nearest_dist = dist;
    }
  }
  return nearest_idx;
}
} // namespace

template<size_t N>
bool oneClassNearestNeighbor(const Eigen::Matrix<double, N, 1> & test_sample,
                             double dist_ratio_thre,
                             const std::vector<Eigen::Matrix<double, N, 1>> & train_sample_list)
{
  size_t nearest_sample_idx_to_test = getNearestSample<N>(test_sample, train_sample_list);
  size_t nearest_sample_idx_to_nearest = getNearestSample<N>(
      train_sample_list[nearest_sample_idx_to_test], train_sample_list, std::set<size_t>{nearest_sample_idx_to_test});

  double distance_test_nearest = (train_sample_list[nearest_sample_idx_to_test] - test_sample).norm();
  double distance_nearest_nearest =
      (train_sample_list[nearest_sample_idx_to_test] - train_sample_list[nearest_sample_idx_to_nearest]).norm();

  return distance_test_nearest / distance_nearest_nearest < dist_ratio_thre;
}

template<size_t N>
bool kNearestNeighbor(const Eigen::Matrix<double, N, 1> & test_sample,
                      size_t K,
                      const std::vector<Eigen::Matrix<double, N, 1>> & train_sample_list,
                      const std::vector<bool> & class_list)
{
  // Set idx_dist_list which is top-K nearest samples and distances
  std::vector<std::pair<int, double>> idx_dist_list(
      K, std::make_pair<int, double>(-1, std::numeric_limits<double>::max()));
  for(size_t i = 0; i < train_sample_list.size(); i++)
  {
    double dist = (train_sample_list[i] - test_sample).squaredNorm();
    for(size_t j = 0; j < K; j++)
    {
      if(dist < idx_dist_list[j].second)
      {
        for(size_t k = K - 1; k > j; k--)
        {
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
  for(const auto & idx_dist : idx_dist_list)
  {
    if(idx_dist.first < 0)
    {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[kNearestNeighbor] idx_dist_list contains uninitialized elements.");
    }
    if(class_list[idx_dist.first])
    {
      positive_num++;
    }
    else
    {
      positive_num--;
    }
  }
  if(positive_num == 0)
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[kNearestNeighbor] Numbers of nearest points of positive and negative are the same.");
  }
  return positive_num > 0;
}
} // namespace DiffRmap
