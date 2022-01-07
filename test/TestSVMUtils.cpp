#include <boost/test/unit_test.hpp>

#include <mc_rtc/logging.h>

#include <ros/package.h>

#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/RmapTraining.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
std::shared_ptr<RmapTraining<SamplingSpaceType>> setupSVM(const std::string& bag_path)
{
  auto rmap_training = std::make_shared<RmapTraining<SamplingSpaceType>>(
      ros::package::getPath("differentiable_rmap") + "/test/data/" + bag_path,
      "/tmp/rmap_svm_model.libsvm",
      false);

  rmap_training->setup();
  rmap_training->runOnce();

  return rmap_training;
}

template <SamplingSpace SamplingSpaceType>
void testCalcSVMValue(const std::string& bag_path)
{
  int argc = 0;
  char** argv = nullptr;
  ros::init(
      argc,
      argv,
      "test_calc_svm_value_" + bag_path.substr(0, bag_path.size() - std::string(".bag").size()));

  auto rmap_sampling = setupSVM<SamplingSpaceType>(bag_path);

  int test_num = 1000;
  for (int i = 0; i < test_num; i++) {
    Sample<SamplingSpaceType> sample =
        poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>());

    double svm_value_libsvm;
    double svm_value_eigen;
    rmap_sampling->testCalcSVMValue(svm_value_libsvm, svm_value_eigen, sample);

    BOOST_CHECK(std::fabs(svm_value_libsvm - svm_value_eigen) < 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR2)
{
  testCalcSVMValue<SamplingSpace::R2>("rmap_sample_set_R2_test.bag");
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE2)
{
  testCalcSVMValue<SamplingSpace::SE2>("rmap_sample_set_SE2_test.bag");
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR3)
{
  testCalcSVMValue<SamplingSpace::R3>("rmap_sample_set_R3_test.bag");
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE3)
{
  testCalcSVMValue<SamplingSpace::SE3>("rmap_sample_set_SE3_test.bag");
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR2IK)
{
  testCalcSVMValue<SamplingSpace::R2>("rmap_sample_set_R2_test_ik.bag");
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE2IK)
{
  testCalcSVMValue<SamplingSpace::SE2>("rmap_sample_set_SE2_test_ik.bag");
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR3IK)
{
  testCalcSVMValue<SamplingSpace::R3>("rmap_sample_set_R3_test_ik.bag");
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE3IK)
{
  testCalcSVMValue<SamplingSpace::SE3>("rmap_sample_set_SE3_test_ik.bag");
}
