/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <mc_rtc/logging.h>

#include <differentiable_rmap/GridUtils.h>

using namespace DiffRmap;

TEST(TestGridUtils, IdxsRatios)
{
  int grid_dim = 5;
  Eigen::VectorXi divide_nums = Eigen::VectorXi(grid_dim);
  divide_nums << 12, 0, 5, 1, 3;

  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    Eigen::VectorXi divide_idxs = Eigen::VectorXi::Random(grid_dim);
    for(int i = 0; i < grid_dim; i++)
    {
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

    EXPECT_TRUE((divide_idxs == restored_divide_idxs));
    EXPECT_TRUE((divide_ratios - restored_divide_ratios).norm() < 1e-10);
  }
}

template<SamplingSpace SamplingSpaceType>
void testGridPos()
{
  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    sva::PTransformd pose = getRandomPose<SamplingSpaceType>();

    Sample<SamplingSpaceType> sample = poseToSample<SamplingSpaceType>(pose);
    GridPos<SamplingSpaceType> grid_pos = sampleToGridPos<SamplingSpaceType>(sample);
    Sample<SamplingSpaceType> restored_sample = gridPosToSample<SamplingSpaceType>(grid_pos);

    // std::cout << "[TestGridUtilsGridPos]" << std::endl;
    // std::cout << "  grid_pos: " << grid_pos.transpose() << std::endl;
    // std::cout << "  sample: " << sample.transpose() << std::endl;
    // std::cout << "  restored_sample: " << restored_sample.transpose() << std::endl;
    // std::cout << "  error: " << (sample - restored_sample).norm() << std::endl;

    if constexpr(SamplingSpaceType == SamplingSpace::SO3)
    {
      // Quaternion multiplied by -1 represents the same rotation
      EXPECT_TRUE((sample - restored_sample).norm() < 1e-10 || (sample + restored_sample).norm() < 1e-10);
    }
    else if constexpr(SamplingSpaceType == SamplingSpace::SE3)
    {
      EXPECT_TRUE((sample.template head<3>() - restored_sample.template head<3>()).norm() < 1e-10);
      // Quaternion multiplied by -1 represents the same rotation
      EXPECT_TRUE((sample.template tail<4>() - restored_sample.template tail<4>()).norm() < 1e-10
                  || (sample.template tail<4>() + restored_sample.template tail<4>()).norm() < 1e-10);
    }
    else
    {
      EXPECT_TRUE((sample - restored_sample).norm() < 1e-10);
    }
  }
}

TEST(TestGridUtils, GridPosR2)
{
  testGridPos<SamplingSpace::R2>();
}
TEST(TestGridUtils, GridPosSO2)
{
  testGridPos<SamplingSpace::SO2>();
}
TEST(TestGridUtils, GridPosSE2)
{
  testGridPos<SamplingSpace::SE2>();
}
TEST(TestGridUtils, GridPosR3)
{
  testGridPos<SamplingSpace::R3>();
}
TEST(TestGridUtils, GridPosSO3)
{
  testGridPos<SamplingSpace::SO3>();
}
TEST(TestGridUtils, GridPosSE3)
{
  testGridPos<SamplingSpace::SE3>();
}

TEST(TestGridUtils, LoopGrid)
{
  GridIdxs<SamplingSpace::SE2> divide_nums = GridIdxs<SamplingSpace::SE2>(5, 2, 3);

  Sample<SamplingSpace::SE2> sample1 = Sample<SamplingSpace::SE2>::Random();
  Sample<SamplingSpace::SE2> sample2 = Sample<SamplingSpace::SE2>::Random();
  Sample<SamplingSpace::SE2> sample_min = sample1.cwiseMin(sample2);
  Sample<SamplingSpace::SE2> sample_max = sample1.cwiseMax(sample2);

  int prev_grid_idx = -1;
  int total_grid_num1 = 0;
  Sample<SamplingSpace::SE2> sample_first;
  Sample<SamplingSpace::SE2> sample_last;

  // std::cout << "[TestGridUtilsLoopGrid]" << std::endl;
  loopGrid<SamplingSpace::SE2>(divide_nums, sample_min, sample_max - sample_min,
                               [&](int grid_idx, const Sample<SamplingSpace::SE2> & sample) {
                                 if(prev_grid_idx == -1)
                                 {
                                   sample_first = sample;
                                 }
                                 sample_last = sample;

                                 // std::cout << "  grid_idx: " << grid_idx << ", prev_grid_idx: " << prev_grid_idx <<
                                 // std::endl; Check that index increases by one
                                 EXPECT_TRUE(grid_idx - prev_grid_idx == 1);

                                 // std::cout << "sample: " << sample.transpose() << std::endl;
                                 // std::cout << "sample_min: " << sample_min.transpose() << std::endl;
                                 // std::cout << "sample_max: " << sample_max.transpose() << std::endl;
                                 // Check that sample is between sample_min and sample_max
                                 EXPECT_TRUE(((sample - sample_min).array() >= -1e-10).all());
                                 EXPECT_TRUE(((sample_max - sample).array() >= -1e-10).all());

                                 prev_grid_idx = grid_idx;
                                 total_grid_num1++;
                               });

  int total_grid_num2 = 1;
  for(int i = 0; i < divide_nums.size(); i++)
  {
    total_grid_num2 *= (divide_nums[i] + 1);
  }
  // std::cout << "  total_grid_num1: " << total_grid_num1 << ", total_grid_num2: " << total_grid_num2 << std::endl;
  // Check that the number of grids accessed is correct
  EXPECT_TRUE(total_grid_num1 == total_grid_num2);

  // std::cout << "  sample_min: " << sample_min.transpose() << std::endl;
  // std::cout << "  sample_first: " << sample_first.transpose() << std::endl;
  // std::cout << "  error: " << (sample_first - sample_min).norm() << std::endl;
  // std::cout << "  sample_max: " << sample_max.transpose() << std::endl;
  // std::cout << "  sample_last: " << sample_last.transpose() << std::endl;
  // std::cout << "  error: " << (sample_last - sample_max).norm() << std::endl;
  // Check that the loop starts with sample_min and ends with sample_max
  EXPECT_TRUE((sample_first - sample_min).norm() < 1e-10);
  EXPECT_TRUE((sample_last - sample_max).norm() < 1e-10);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
