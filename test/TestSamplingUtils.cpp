#include <boost/test/unit_test.hpp>

#include <mc_rtc/logging.h>

#include <differentiable_rmap/SamplingUtils.h>
#include <differentiable_rmap/MathUtils.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
sva::PTransformd getRandomPose()
{
  sva::PTransformd pose = sva::PTransformd::Identity();
  if constexpr (SamplingSpaceType == SamplingSpace::R2) {
      pose.translation().head<2>().setRandom();
    } else if constexpr (SamplingSpaceType == SamplingSpace::SO2) {
      pose.rotation() = Eigen::AngleAxisd(
          M_PI * Eigen::Matrix<double, 1, 1>::Random()[0],
          Eigen::Vector3d::UnitZ()).toRotationMatrix();
    } else if constexpr (SamplingSpaceType == SamplingSpace::SE2) {
      pose = getRandomPose<SamplingSpace::SO2>() * getRandomPose<SamplingSpace::R2>();
    } else if constexpr (SamplingSpaceType == SamplingSpace::R3) {
      pose.translation().setRandom();
    } else if constexpr (SamplingSpaceType == SamplingSpace::SO3) {
      pose.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    } else if constexpr (SamplingSpaceType == SamplingSpace::SE3) {
      pose = getRandomPose<SamplingSpace::SO3>() * getRandomPose<SamplingSpace::R3>();
    } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[getRandomPose] unsupported for SamplingSpace {}", SamplingSpaceType);
  }
  return pose;
}

template <SamplingSpace SamplingSpaceType>
void test()
{
  int test_num = 1000;
  for (int i = 0; i < test_num; i++) {
    sva::PTransformd pose = getRandomPose<SamplingSpaceType>();

    Sample<SamplingSpaceType> sample = poseToSample<SamplingSpaceType>(pose);
    Input<SamplingSpaceType> input = sampleToInput<SamplingSpaceType>(sample);
    sva::PTransformd restored_pose = sampleToPose<SamplingSpaceType>(sample);
    Sample<SamplingSpaceType> restored_sample = inputToSample<SamplingSpaceType>(input);
    sva::PTransformd restored_pose2 = sampleToPose<SamplingSpaceType>(restored_sample);

    BOOST_CHECK((sample - restored_sample).norm() < 1e-10);
    BOOST_CHECK(sva::transformError(pose, restored_pose).vector().norm() < 1e-10);
    BOOST_CHECK(sva::transformError(pose, restored_pose2).vector().norm() < 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsR2)
{
  test<SamplingSpace::R2>();
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsSO2)
{
  test<SamplingSpace::SO2>();
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsSE2)
{
  test<SamplingSpace::SE2>();
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsR3)
{
  test<SamplingSpace::R3>();
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsSO3)
{
  test<SamplingSpace::SO3>();
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsSE3)
{
  test<SamplingSpace::SE3>();
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsRot3D)
{
  sva::PTransformd pose = getRandomPose<SamplingSpace::SO3>();
  const Eigen::Matrix3d& rot = pose.rotation().transpose();
  Sample<SamplingSpace::SO3> sample = poseToSample<SamplingSpace::SO3>(pose);
  Input<SamplingSpace::SO3> input = sampleToInput<SamplingSpace::SO3>(sample);

  // std::cout << "[TestSamplingUtils3DRot]" << std::endl;
  // std::cout << "  pose rotation:\n" << rot << std::endl;
  // std::cout << "  sample:\n" << sample.transpose() << std::endl;
  // std::cout << "  input:\n" << input.transpose() << std::endl;

  for (int i = 0; i < 3; i++) {
    BOOST_CHECK((rot.row(i).transpose() - input.segment<3>(3 * i)).norm() < 1e-10);
  }
}
