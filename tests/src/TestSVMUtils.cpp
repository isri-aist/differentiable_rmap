/* Author: Masaki Murooka */

#include <gtest/gtest.h>

#include <ros/package.h>

#include <differentiable_rmap/RmapTraining.h>
#include <differentiable_rmap/SVMUtils.h>

using namespace DiffRmap;

template<SamplingSpace SamplingSpaceType>
std::shared_ptr<RmapTraining<SamplingSpaceType>> setupSVM(const std::string & bag_path)
{
  auto rmap_training = std::make_shared<RmapTraining<SamplingSpaceType>>(
      ros::package::getPath("differentiable_rmap") + "/tests/data/" + bag_path, "/tmp/rmap_svm_model.libsvm", false);

  mc_rtc::Configuration mc_rtc_config;
  // Set grid map resolution large to reduce the number of prediction
  mc_rtc_config.add("grid_map_resolution", 1.0);
  rmap_training->configure(mc_rtc_config);

  rmap_training->setup();
  rmap_training->runOnce();

  return rmap_training;
}

template<SamplingSpace SamplingSpaceType>
void testCalcSVMValue(const std::string & bag_path)
{
  auto rmap_sampling = setupSVM<SamplingSpaceType>(bag_path);

  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    double svm_value_libsvm;
    double svm_value_eigen;
    rmap_sampling->testCalcSVMValue(svm_value_libsvm, svm_value_eigen,
                                    poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>()));

    EXPECT_TRUE(std::fabs(svm_value_libsvm - svm_value_eigen) < 1e-10);
  }
}

TEST(TestSVMUtils, CalcSVMValueR2)
{
  testCalcSVMValue<SamplingSpace::R2>("rmap_sample_set_R2_test.bag");
}
TEST(TestSVMUtils, CalcSVMValueSE2)
{
  testCalcSVMValue<SamplingSpace::SE2>("rmap_sample_set_SE2_test.bag");
}
TEST(TestSVMUtils, CalcSVMValueR3)
{
  testCalcSVMValue<SamplingSpace::R3>("rmap_sample_set_R3_test.bag");
}
TEST(TestSVMUtils, CalcSVMValueSE3)
{
  testCalcSVMValue<SamplingSpace::SE3>("rmap_sample_set_SE3_test.bag");
}

TEST(TestSVMUtils, CalcSVMValueR2IK)
{
  testCalcSVMValue<SamplingSpace::R2>("rmap_sample_set_R2_test_ik.bag");
}
TEST(TestSVMUtils, CalcSVMValueSE2IK)
{
  testCalcSVMValue<SamplingSpace::SE2>("rmap_sample_set_SE2_test_ik.bag");
}
TEST(TestSVMUtils, CalcSVMValueR3IK)
{
  testCalcSVMValue<SamplingSpace::R3>("rmap_sample_set_R3_test_ik.bag");
}
TEST(TestSVMUtils, CalcSVMValueSE3IK)
{
  testCalcSVMValue<SamplingSpace::SE3>("rmap_sample_set_SE3_test_ik.bag");
}

template<SamplingSpace SamplingSpaceType>
void testCalcSVMGrad(const std::string & bag_path)
{
  auto rmap_sampling = setupSVM<SamplingSpaceType>(bag_path);

  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    Vel<SamplingSpaceType> svm_grad_analytical;
    Vel<SamplingSpaceType> svm_grad_numerical;
    rmap_sampling->testCalcSVMGrad(svm_grad_analytical, svm_grad_numerical,
                                   poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>()));

    // std::cout << "[testCalcSVMGrad]" << std::endl;
    // std::cout << "  svm_grad_analytical: " << svm_grad_analytical.transpose() << std::endl;
    // std::cout << "  svm_grad_numerical: " << svm_grad_numerical.transpose() << std::endl;
    // std::cout << "  error: " << (svm_grad_analytical - svm_grad_numerical).norm() << std::endl;

    EXPECT_TRUE((svm_grad_analytical - svm_grad_numerical).norm() < 1e-3);
  }
}

TEST(TestSVMUtils, CalcSVMGradR2)
{
  testCalcSVMGrad<SamplingSpace::R2>("rmap_sample_set_R2_test.bag");
}
TEST(TestSVMUtils, CalcSVMGradSE2)
{
  testCalcSVMGrad<SamplingSpace::SE2>("rmap_sample_set_SE2_test.bag");
}
TEST(TestSVMUtils, CalcSVMGradR3)
{
  testCalcSVMGrad<SamplingSpace::R3>("rmap_sample_set_R3_test.bag");
}
TEST(TestSVMUtils, CalcSVMGradSE3)
{
  testCalcSVMGrad<SamplingSpace::SE3>("rmap_sample_set_SE3_test.bag");
}

TEST(TestSVMUtils, CalcSVMGradR2IK)
{
  testCalcSVMGrad<SamplingSpace::R2>("rmap_sample_set_R2_test_ik.bag");
}
TEST(TestSVMUtils, CalcSVMGradSE2IK)
{
  testCalcSVMGrad<SamplingSpace::SE2>("rmap_sample_set_SE2_test_ik.bag");
}
TEST(TestSVMUtils, CalcSVMGradR3IK)
{
  testCalcSVMGrad<SamplingSpace::R3>("rmap_sample_set_R3_test_ik.bag");
}
TEST(TestSVMUtils, CalcSVMGradSE3IK)
{
  testCalcSVMGrad<SamplingSpace::SE3>("rmap_sample_set_SE3_test_ik.bag");
}

template<SamplingSpace SamplingSpaceType>
void testCalcSVMGradRel(const std::string & bag_path)
{
  auto rmap_sampling = setupSVM<SamplingSpaceType>(bag_path);

  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    Vel<SamplingSpaceType> pre_grad_analytical;
    Vel<SamplingSpaceType> suc_grad_analytical;
    Vel<SamplingSpaceType> pre_grad_numerical;
    Vel<SamplingSpaceType> suc_grad_numerical;
    rmap_sampling->testCalcSVMGradRel(pre_grad_analytical, suc_grad_analytical, pre_grad_numerical, suc_grad_numerical,
                                      poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>()),
                                      poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>()));

    // std::cout << "[testCalcSVMGradRel]" << std::endl;
    // std::cout << "  pre_grad_analytical: " << pre_grad_analytical.transpose() << std::endl;
    // std::cout << "  pre_grad_numerical: " << pre_grad_numerical.transpose() << std::endl;
    // std::cout << "  pre_error: " << (pre_grad_analytical - pre_grad_numerical).norm() << std::endl;
    // std::cout << "  suc_grad_analytical: " << suc_grad_analytical.transpose() << std::endl;
    // std::cout << "  suc_grad_numerical: " << suc_grad_numerical.transpose() << std::endl;
    // std::cout << "  suc_error: " << (suc_grad_analytical - suc_grad_numerical).norm() << std::endl;

    EXPECT_TRUE((pre_grad_analytical - pre_grad_numerical).norm() < 1e-3);
    EXPECT_TRUE((suc_grad_analytical - suc_grad_numerical).norm() < 1e-3);
  }
}

TEST(TestSVMUtils, CalcSVMGradRelR2)
{
  testCalcSVMGradRel<SamplingSpace::R2>("rmap_sample_set_R2_test.bag");
}
TEST(TestSVMUtils, CalcSVMGradRelSE2)
{
  testCalcSVMGradRel<SamplingSpace::SE2>("rmap_sample_set_SE2_test.bag");
}
TEST(TestSVMUtils, CalcSVMGradRelR3)
{
  testCalcSVMGradRel<SamplingSpace::R3>("rmap_sample_set_R3_test.bag");
}
TEST(TestSVMUtils, CalcSVMGradRelSE3)
{
  testCalcSVMGradRel<SamplingSpace::SE3>("rmap_sample_set_SE3_test.bag");
}

template<SamplingSpace SamplingSpaceType>
void testInputToVelMat()
{
  using InputToVelMat = Eigen::Matrix<double, velDim<SamplingSpaceType>(), inputDim<SamplingSpaceType>()>;

  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    sva::PTransformd pose = getRandomPose<SamplingSpaceType>();
    Sample<SamplingSpaceType> sample = poseToSample<SamplingSpaceType>(pose);

    InputToVelMat mat_analytical =
        sampleToVelMat<SamplingSpaceType>(sample) * inputToSampleMat<SamplingSpaceType>(sample);

    InputToVelMat mat_numerical;
    double eps = 1e-6;
    for(int j = 0; j < velDim<SamplingSpaceType>(); j++)
    {
      Vel<SamplingSpaceType> vel = eps * Vel<SamplingSpaceType>::Unit(j);
      Sample<SamplingSpaceType> sample_plus = sample;
      integrateVelToSample<SamplingSpaceType>(sample_plus, vel);
      Sample<SamplingSpaceType> sample_minus = sample;
      integrateVelToSample<SamplingSpaceType>(sample_minus, -vel);
      mat_numerical.row(j) =
          (sampleToInput<SamplingSpaceType>(sample_plus) - sampleToInput<SamplingSpaceType>(sample_minus)) / (2 * eps);
    }

    // std::cout << "[testInputToVelMat]" << std::endl;
    // std::cout << "  mat_analytical:\n" << mat_analytical << std::endl;
    // std::cout << "  mat_numerical:\n" << mat_numerical << std::endl;
    // std::cout << "  error: " << (mat_analytical - mat_numerical).norm() << std::endl;

    EXPECT_TRUE((mat_analytical - mat_numerical).norm() < 1e-8);
  }
}

TEST(TestSVMUtils, InputToVelMatR2)
{
  testInputToVelMat<SamplingSpace::R2>();
}
TEST(TestSVMUtils, InputToVelMatSO2)
{
  testInputToVelMat<SamplingSpace::SO2>();
}
TEST(TestSVMUtils, InputToVelMatSE2)
{
  testInputToVelMat<SamplingSpace::SE2>();
}
TEST(TestSVMUtils, InputToVelMatR3)
{
  testInputToVelMat<SamplingSpace::R3>();
}
TEST(TestSVMUtils, InputToVelMatSO3)
{
  testInputToVelMat<SamplingSpace::SO3>();
}
TEST(TestSVMUtils, InputToVelMatSE3)
{
  testInputToVelMat<SamplingSpace::SE3>();
}

template<SamplingSpace SamplingSpaceType>
void testRelSample()
{
  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    sva::PTransformd pre_pose = getRandomPose<SamplingSpaceType>();
    sva::PTransformd suc_pose = getRandomPose<SamplingSpaceType>();
    Sample<SamplingSpaceType> pre_sample = poseToSample<SamplingSpaceType>(pre_pose);
    Sample<SamplingSpaceType> suc_sample = poseToSample<SamplingSpaceType>(suc_pose);

    Sample<SamplingSpaceType> rel_sample1 = relSample<SamplingSpaceType>(pre_sample, suc_sample);
    Sample<SamplingSpaceType> rel_sample2 = poseToSample<SamplingSpaceType>(suc_pose * pre_pose.inv());

    // std::cout << "[testRelSample]" << std::endl;
    // std::cout << "  rel_sample1: " << rel_sample1.transpose() << std::endl;
    // std::cout << "  rel_sample2: " << rel_sample2.transpose() << std::endl;
    // std::cout << "  error: " << (rel_sample1 - rel_sample2).norm() << std::endl;

    if constexpr(SamplingSpaceType == SamplingSpace::SO3)
    {
      // Quaternion multiplied by -1 represents the same rotation
      EXPECT_TRUE((rel_sample1 - rel_sample2).norm() < 1e-8 || (rel_sample1 + rel_sample2).norm() < 1e-8);
    }
    else if constexpr(SamplingSpaceType == SamplingSpace::SE3)
    {
      EXPECT_TRUE((rel_sample1.template head<3>() - rel_sample2.template head<3>()).norm() < 1e-8);
      // Quaternion multiplied by -1 represents the same rotation
      EXPECT_TRUE((rel_sample1.template tail<4>() - rel_sample2.template tail<4>()).norm() < 1e-8
                  || (rel_sample1.template tail<4>() + rel_sample2.template tail<4>()).norm() < 1e-8);
    }
    else
    {
      EXPECT_TRUE((rel_sample1 - rel_sample2).norm() < 1e-8);
    }
  }
}

TEST(TestSVMUtils, RelSampleR2)
{
  testRelSample<SamplingSpace::R2>();
}
TEST(TestSVMUtils, RelSampleSO2)
{
  testRelSample<SamplingSpace::SO2>();
}
TEST(TestSVMUtils, RelSampleSE2)
{
  testRelSample<SamplingSpace::SE2>();
}
TEST(TestSVMUtils, RelSampleR3)
{
  testRelSample<SamplingSpace::R3>();
}
TEST(TestSVMUtils, RelSampleSO3)
{
  testRelSample<SamplingSpace::SO3>();
}
TEST(TestSVMUtils, RelSampleSE3)
{
  testRelSample<SamplingSpace::SE3>();
}

template<SamplingSpace SamplingSpaceType>
void testMidSample()
{
  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    sva::PTransformd pose1 = getRandomPose<SamplingSpaceType>();
    sva::PTransformd pose2 = getRandomPose<SamplingSpaceType>();
    Sample<SamplingSpaceType> sample1 = poseToSample<SamplingSpaceType>(pose1);
    Sample<SamplingSpaceType> sample2 = poseToSample<SamplingSpaceType>(pose2);

    Sample<SamplingSpaceType> mid_sample1 = midSample<SamplingSpaceType>(sample1, sample2);
    Sample<SamplingSpaceType> mid_sample2 = poseToSample<SamplingSpaceType>(sva::interpolate(pose1, pose2, 0.5));

    // std::cout << "[testMidSample]" << std::endl;
    // std::cout << "  mid_sample1: " << mid_sample1.transpose() << std::endl;
    // std::cout << "  mid_sample2: " << mid_sample2.transpose() << std::endl;
    // std::cout << "  error: " << (mid_sample1 - mid_sample2).norm() << std::endl;

    EXPECT_TRUE((mid_sample1 - mid_sample2).norm() < 1e-8);
  }
}

TEST(TestSVMUtils, MidSampleR2)
{
  testMidSample<SamplingSpace::R2>();
}
TEST(TestSVMUtils, MidSampleSO2)
{
  testMidSample<SamplingSpace::SO2>();
}
TEST(TestSVMUtils, MidSampleSE2)
{
  testMidSample<SamplingSpace::SE2>();
}
TEST(TestSVMUtils, MidSampleR3)
{
  testMidSample<SamplingSpace::R3>();
}
TEST(TestSVMUtils, MidSampleSO3)
{
  testMidSample<SamplingSpace::SO3>();
}
TEST(TestSVMUtils, MidSampleSE3)
{
  testMidSample<SamplingSpace::SE3>();
}

template<SamplingSpace SamplingSpaceType>
void testRelSampleToSampleMat()
{
  int test_num = 1000;
  for(int i = 0; i < test_num; i++)
  {
    sva::PTransformd pre_pose = getRandomPose<SamplingSpaceType>();
    sva::PTransformd suc_pose = getRandomPose<SamplingSpaceType>();
    Sample<SamplingSpaceType> pre_sample = poseToSample<SamplingSpaceType>(pre_pose);
    Sample<SamplingSpaceType> suc_sample = poseToSample<SamplingSpaceType>(suc_pose);

    double eps = 1e-6;

    SampleToSampleMat<SamplingSpaceType> pre_mat_analytical =
        relSampleToSampleMat<SamplingSpaceType>(pre_sample, suc_sample, false);
    SampleToSampleMat<SamplingSpaceType> pre_mat_numerical;
    for(int j = 0; j < sampleDim<SamplingSpaceType>(); j++)
    {
      Sample<SamplingSpaceType> pre_sample_plus = pre_sample + eps * Sample<SamplingSpaceType>::Unit(j);
      Sample<SamplingSpaceType> pre_sample_minus = pre_sample - eps * Sample<SamplingSpaceType>::Unit(j);
      pre_mat_numerical.col(j) = (relSample<SamplingSpaceType>(pre_sample_plus, suc_sample)
                                  - relSample<SamplingSpaceType>(pre_sample_minus, suc_sample))
                                 / (2 * eps);
    }

    SampleToSampleMat<SamplingSpaceType> suc_mat_analytical =
        relSampleToSampleMat<SamplingSpaceType>(pre_sample, suc_sample, true);
    SampleToSampleMat<SamplingSpaceType> suc_mat_numerical;
    for(int j = 0; j < sampleDim<SamplingSpaceType>(); j++)
    {
      Sample<SamplingSpaceType> suc_sample_plus = suc_sample + eps * Sample<SamplingSpaceType>::Unit(j);
      Sample<SamplingSpaceType> suc_sample_minus = suc_sample - eps * Sample<SamplingSpaceType>::Unit(j);
      suc_mat_numerical.col(j) = (relSample<SamplingSpaceType>(pre_sample, suc_sample_plus)
                                  - relSample<SamplingSpaceType>(pre_sample, suc_sample_minus))
                                 / (2 * eps);
    }

    // std::cout << "[testRelSampleToSampleMat]" << std::endl;
    // std::cout << "  pre_mat_analytical:\n" << pre_mat_analytical << std::endl;
    // std::cout << "  pre_mat_numerical:\n" << pre_mat_numerical << std::endl;
    // std::cout << "  pre_error: " << (pre_mat_analytical - pre_mat_numerical).norm() << std::endl;
    // std::cout << "  suc_mat_analytical:\n" << suc_mat_analytical << std::endl;
    // std::cout << "  suc_mat_numerical:\n" << suc_mat_numerical << std::endl;
    // std::cout << "  suc_error: " << (suc_mat_analytical - suc_mat_numerical).norm() << std::endl;

    EXPECT_TRUE((pre_mat_analytical - pre_mat_numerical).norm() < 1e-8);
    EXPECT_TRUE((suc_mat_analytical - suc_mat_numerical).norm() < 1e-8);
  }
}

TEST(TestSVMUtils, RelSampleToSampleMatR2)
{
  testRelSampleToSampleMat<SamplingSpace::R2>();
}
TEST(TestSVMUtils, RelSampleToSampleMatSO2)
{
  testRelSampleToSampleMat<SamplingSpace::SO2>();
}
TEST(TestSVMUtils, RelSampleToSampleMatSE2)
{
  testRelSampleToSampleMat<SamplingSpace::SE2>();
}
TEST(TestSVMUtils, RelSampleToSampleMatR3)
{
  testRelSampleToSampleMat<SamplingSpace::R3>();
}
TEST(TestSVMUtils, RelSampleToSampleMatSO3)
{
  testRelSampleToSampleMat<SamplingSpace::SO3>();
}
TEST(TestSVMUtils, RelSampleToSampleMatSE3)
{
  testRelSampleToSampleMat<SamplingSpace::SE3>();
}

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "test_svm_utils");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
