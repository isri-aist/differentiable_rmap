#include <boost/test/unit_test.hpp>

#include <mc_rtc/logging.h>

#include <differentiable_rmap/GridUtils.h>

using namespace DiffRmap;


BOOST_AUTO_TEST_CASE(TestGridUtilsIdxsRatios)
{
  int grid_dim = 5;
  Eigen::VectorXi divide_nums = Eigen::VectorXi(grid_dim);
  divide_nums << 12, 0, 5, 1, 3;

  int test_num = 1000;
  for (int i = 0; i < test_num; i++) {
    Eigen::VectorXi divide_idxs = Eigen::VectorXi::Random(grid_dim);
    for (int i = 0; i < grid_dim; i++) {
      int n = divide_nums[i] + 1;
      divide_idxs[i] = (divide_idxs[i] % n + n) % n;
    }

    // Convert divide_idxs -> divide_ratios -> restored_divide_idxs -> restored_divide_ratios, and check consistency
    Eigen::VectorXd divide_ratios(grid_dim);
    Eigen::VectorXi restored_divide_idxs(grid_dim);
    Eigen::VectorXd restored_divide_ratios(grid_dim);
    gridDivideIdxsToRatios(divide_ratios, divide_idxs, divide_nums);
    gridDivideRatiosToIdxs(restored_divide_idxs, divide_ratios, divide_nums);
    gridDivideIdxsToRatios(restored_divide_ratios, restored_divide_idxs, divide_nums);

    // std::cout << "[TestGridUtilsIdxsRatios]" << std::endl;
    // std::cout << "  divide_idxs: " << divide_idxs.transpose() << std::endl;
    // std::cout << "  restored_divide_idxs: " << restored_divide_idxs.transpose() << std::endl;
    // std::cout << "  equal: " << (divide_idxs == restored_divide_idxs) << std::endl;
    // std::cout << "  divide_ratios: " << divide_ratios.transpose() << std::endl;
    // std::cout << "  restored_divide_ratios: " << restored_divide_ratios.transpose() << std::endl;
    // std::cout << "  error: " << (divide_ratios - restored_divide_ratios).norm() << std::endl;

    BOOST_CHECK((divide_idxs == restored_divide_idxs));
    BOOST_CHECK((divide_ratios - restored_divide_ratios).norm() < 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(TestGridUtilsLoopGrid)
{
  GridIdxsType<SamplingSpace::SE2> divide_nums = GridIdxsType<SamplingSpace::SE2>(5, 2, 3);

  Sample<SamplingSpace::SE2> sample1 = Sample<SamplingSpace::SE2>::Random();
  Sample<SamplingSpace::SE2> sample2 = Sample<SamplingSpace::SE2>::Random();
  Sample<SamplingSpace::SE2> sample_min = sample1.cwiseMin(sample2);
  Sample<SamplingSpace::SE2> sample_max = sample1.cwiseMax(sample2);

  int prev_grid_idx = -1;
  int total_grid_num1 = 0;
  Sample<SamplingSpace::SE2> sample_first;
  Sample<SamplingSpace::SE2> sample_last;

  // std::cout << "[TestGridUtilsLoopGrid]" << std::endl;
  loopGrid<SamplingSpace::SE2>(
      divide_nums,
      sample_min,
      sample_max - sample_min,
      [&](int grid_idx, const Sample<SamplingSpace::SE2>& sample) {
        if (prev_grid_idx == -1) {
          sample_first = sample;
        }
        sample_last = sample;

        // std::cout << "  grid_idx: " << grid_idx << ", prev_grid_idx: " << prev_grid_idx << std::endl;
        // Check that index increases by one
        BOOST_CHECK(grid_idx - prev_grid_idx == 1);

        // std::cout << "sample: " << sample.transpose() << std::endl;
        // std::cout << "sample_min: " << sample_min.transpose() << std::endl;
        // std::cout << "sample_max: " << sample_max.transpose() << std::endl;
        // Check that sample is between sample_min and sample_max
        BOOST_CHECK(((sample - sample_min).array() >= -1e-10).all());
        BOOST_CHECK(((sample_max - sample).array() >= -1e-10).all());

        prev_grid_idx = grid_idx;
        total_grid_num1++;
      });

  int total_grid_num2 = 1;
  for (int i = 0; i < divide_nums.size(); i++) {
    total_grid_num2 *= (divide_nums[i] + 1);
  }
  // std::cout << "  total_grid_num1: " << total_grid_num1 << ", total_grid_num2: " << total_grid_num2 << std::endl;
  // Check that the number of grids accessed is correct
  BOOST_CHECK(total_grid_num1 == total_grid_num2);

  // std::cout << "  sample_min: " << sample_min.transpose() << std::endl;
  // std::cout << "  sample_first: " << sample_first.transpose() << std::endl;
  // std::cout << "  error: " << (sample_first - sample_min).norm() << std::endl;
  // std::cout << "  sample_max: " << sample_max.transpose() << std::endl;
  // std::cout << "  sample_last: " << sample_last.transpose() << std::endl;
  // std::cout << "  error: " << (sample_last - sample_max).norm() << std::endl;
  // Check that the loop starts with sample_min and ends with sample_max
  BOOST_CHECK((sample_first - sample_min).norm() < 1e-10);
  BOOST_CHECK((sample_last - sample_max).norm() < 1e-10);
}
