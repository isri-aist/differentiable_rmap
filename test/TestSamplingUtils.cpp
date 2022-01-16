#include <boost/test/unit_test.hpp>

#include <mc_rtc/logging.h>

#include <differentiable_rmap/SamplingUtils.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
void testConversion()
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

BOOST_AUTO_TEST_CASE(TestSamplingUtilsConversionR2) { testConversion<SamplingSpace::R2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsConversionSO2) { testConversion<SamplingSpace::SO2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsConversionSE2) { testConversion<SamplingSpace::SE2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsConversionR3) { testConversion<SamplingSpace::R3>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsConversionSO3) { testConversion<SamplingSpace::SO3>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsConversionSE3) { testConversion<SamplingSpace::SE3>(); }

BOOST_AUTO_TEST_CASE(TestSamplingUtilsConversionRot3D)
{
  int test_num = 1000;
  for (int i = 0; i < test_num; i++) {
    sva::PTransformd pose = getRandomPose<SamplingSpace::SO3>();
    const Eigen::Matrix3d& rot = pose.rotation().transpose();
    Sample<SamplingSpace::SO3> sample = poseToSample<SamplingSpace::SO3>(pose);
    Input<SamplingSpace::SO3> input = sampleToInput<SamplingSpace::SO3>(sample);

    // std::cout << "[TestSamplingUtilsConversion3DRot]" << std::endl;
    // std::cout << "  pose rotation:\n" << rot << std::endl;
    // std::cout << "  sample:\n" << sample.transpose() << std::endl;
    // std::cout << "  input:\n" << input.transpose() << std::endl;

    for (int i = 0; i < 3; i++) {
      BOOST_CHECK((rot.row(i).transpose() - input.segment<3>(3 * i)).norm() < 1e-10);
    }
  }
}

template <SamplingSpace SamplingSpaceType>
void testIntegrate()
{
  int test_num = 1000;

  // Compare sample integration and pose integration
  for (int i = 0; i < test_num; i++) {
    sva::PTransformd pose = getRandomPose<SamplingSpaceType>();
    Vel<SamplingSpaceType> vel = 1e-3 * Vel<SamplingSpaceType>::Random();

    Sample<SamplingSpaceType> integrated_sample = poseToSample<SamplingSpaceType>(pose);
    integrateVelToSample<SamplingSpaceType>(integrated_sample, vel);

    sva::PTransformd integrated_pose2 = pose;
    if constexpr (SamplingSpaceType == SamplingSpace::R2) {
        integrated_pose2.translation().template head<2>() += vel;
      } else if constexpr (SamplingSpaceType == SamplingSpace::SO2) {
        integrated_pose2.rotation() =
            (pose.rotation().transpose() *
             Eigen::AngleAxisd(vel[0], Eigen::Vector3d::UnitZ()).toRotationMatrix()).transpose();
      } else if constexpr (SamplingSpaceType == SamplingSpace::SE2) {
        integrated_pose2.translation().template head<2>() += vel.template head<2>();
        integrated_pose2.rotation() =
            (pose.rotation().transpose() *
             Eigen::AngleAxisd(vel[2], Eigen::Vector3d::UnitZ()).toRotationMatrix()).transpose();
      } else if constexpr (SamplingSpaceType == SamplingSpace::R3) {
        integrated_pose2.translation() += vel;
      } else if constexpr (SamplingSpaceType == SamplingSpace::SO3) {
        integrated_pose2.rotation() =
            (pose.rotation().transpose() *
             Eigen::AngleAxisd(vel.norm(), vel.normalized()).toRotationMatrix()).transpose();
      } else if constexpr (SamplingSpaceType == SamplingSpace::SE3) {
        integrated_pose2.translation() += vel.template head<3>();
        integrated_pose2.rotation() =
            (pose.rotation().transpose() *
             Eigen::AngleAxisd(vel.template tail<3>().norm(), vel.template tail<3>().normalized()
                               ).toRotationMatrix()).transpose();
      } else {
      static_assert(static_cast<bool>(static_cast<int>(SamplingSpaceType)) && false,
                    "[testIntegrate] Unsupported sampling space.");
    }
    Sample<SamplingSpaceType> integrated_sample2 = poseToSample<SamplingSpaceType>(integrated_pose2);

    // if ((integrated_sample - integrated_sample2).norm() >= 1e-10) {
    //   std::cout << "[testIntegrate]" << std::endl;
    //   std::cout << "  original_sample: " << poseToSample<SamplingSpaceType>(pose).transpose() << std::endl;
    //   std::cout << "  integrated_sample: " << integrated_sample.transpose() << std::endl;
    //   std::cout << "  integrated_sample2: " << integrated_sample2.transpose() << std::endl;
    //   std::cout << "  error: " << (integrated_sample - integrated_sample2).norm()
    //             << " / " << (integrated_sample + integrated_sample2).norm() << std::endl;
    // }

    if constexpr (SamplingSpaceType == SamplingSpace::SO2) {
        // Allow 2 pi deviation for Yaw
        BOOST_CHECK(std::fmod(integrated_sample[0] - integrated_sample2[0], 2 * M_PI) < 1e-10);
      } else if constexpr (SamplingSpaceType == SamplingSpace::SE2) {
        BOOST_CHECK((integrated_sample.template head<2>() - integrated_sample2.template head<2>()).norm() < 1e-10);
        // Allow 2 pi deviation for Yaw
        BOOST_CHECK(std::fmod(integrated_sample[2] - integrated_sample2[2], 2 * M_PI) < 1e-10);
      } else if constexpr (SamplingSpaceType == SamplingSpace::SO3) {
        // Quaternion multiplied by -1 represents the same rotation
        BOOST_CHECK((integrated_sample - integrated_sample2).norm() < 1e-10 ||
                    (integrated_sample + integrated_sample2).norm() < 1e-10);
      } else if constexpr (SamplingSpaceType == SamplingSpace::SE3) {
        BOOST_CHECK((integrated_sample.template head<3>() - integrated_sample2.template head<3>()).norm() < 1e-10);
        // Quaternion multiplied by -1 represents the same rotation
        BOOST_CHECK((integrated_sample.template tail<4>() - integrated_sample2.template tail<4>()).norm() < 1e-10 ||
                    (integrated_sample.template tail<4>() + integrated_sample2.template tail<4>()).norm() < 1e-10);
      } else {
      BOOST_CHECK((integrated_sample - integrated_sample2).norm() < 1e-10);
    }

    BOOST_CHECK(
        sva::transformError(sampleToPose<SamplingSpaceType>(integrated_sample),
                            sampleToPose<SamplingSpaceType>(integrated_sample2)).vector().norm() < 1e-10);
  }

  // Check zero velocity integration does not change sample
  for (int i = 0; i < test_num; i++) {
    Sample<SamplingSpaceType> sample = poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>());
    Sample<SamplingSpaceType> integrated_sample = sample;
    integrateVelToSample<SamplingSpaceType>(integrated_sample, Vel<SamplingSpaceType>::Zero());
    BOOST_CHECK((sample - integrated_sample).norm() < 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsIntegrateR2) { testIntegrate<SamplingSpace::R2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsIntegrateSO2) { testIntegrate<SamplingSpace::SO2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsIntegrateSE2) { testIntegrate<SamplingSpace::SE2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsIntegrateR3) { testIntegrate<SamplingSpace::R3>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsIntegrateSO3) { testIntegrate<SamplingSpace::SO3>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsIntegrateSE3) { testIntegrate<SamplingSpace::SE3>(); }

template <SamplingSpace SamplingSpaceType>
void testSampleError()
{
  int test_num = 1000;
  for (int i = 0; i < test_num; i++) {
    sva::PTransformd pose = getRandomPose<SamplingSpaceType>();
    Vel<SamplingSpaceType> vel = 1e-3 * Vel<SamplingSpaceType>::Random();
    Sample<SamplingSpaceType> sample = poseToSample<SamplingSpaceType>(pose);
    Sample<SamplingSpaceType> integrated_sample = sample;
    integrateVelToSample<SamplingSpaceType>(integrated_sample, vel);

    // std::cout << "[testSampleError]" << std::endl;
    // std::cout << "  vel: " << vel.transpose() << std::endl;
    // std::cout << "  sample_error: " << sampleError<SamplingSpaceType>(sample, integrated_sample).transpose() << std::endl;
    // std::cout << "  error: " << (vel - sampleError<SamplingSpaceType>(sample, integrated_sample)).norm() << std::endl;

    BOOST_CHECK((vel - sampleError<SamplingSpaceType>(sample, integrated_sample)).norm() < 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(TestSamplingUtilsSampleErrorR2) { testSampleError<SamplingSpace::R2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsSampleErrorSO2) { testSampleError<SamplingSpace::SO2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsSampleErrorSE2) { testSampleError<SamplingSpace::SE2>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsSampleErrorR3) { testSampleError<SamplingSpace::R3>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsSampleErrorSO3) { testSampleError<SamplingSpace::SO3>(); }
BOOST_AUTO_TEST_CASE(TestSamplingUtilsSampleErrorSE3) { testSampleError<SamplingSpace::SE3>(); }
