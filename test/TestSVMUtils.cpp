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

  mc_rtc::Configuration mc_rtc_config;
  // Set grid map resolution large to reduce the number of prediction
  mc_rtc_config.add("grid_map_resolution", 1.0);
  rmap_training->configure(mc_rtc_config);

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
    double svm_value_libsvm;
    double svm_value_eigen;
    rmap_sampling->testCalcSVMValue(
        svm_value_libsvm,
        svm_value_eigen,
        poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>()));

    BOOST_CHECK(std::fabs(svm_value_libsvm - svm_value_eigen) < 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR2) { testCalcSVMValue<SamplingSpace::R2>("rmap_sample_set_R2_test.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE2) { testCalcSVMValue<SamplingSpace::SE2>("rmap_sample_set_SE2_test.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR3) { testCalcSVMValue<SamplingSpace::R3>("rmap_sample_set_R3_test.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE3) { testCalcSVMValue<SamplingSpace::SE3>("rmap_sample_set_SE3_test.bag"); }

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR2IK) { testCalcSVMValue<SamplingSpace::R2>("rmap_sample_set_R2_test_ik.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE2IK) { testCalcSVMValue<SamplingSpace::SE2>("rmap_sample_set_SE2_test_ik.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueR3IK) { testCalcSVMValue<SamplingSpace::R3>("rmap_sample_set_R3_test_ik.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMValueSE3IK) { testCalcSVMValue<SamplingSpace::SE3>("rmap_sample_set_SE3_test_ik.bag"); }

template <SamplingSpace SamplingSpaceType>
void testCalcSVMGrad(const std::string& bag_path)
{
  int argc = 0;
  char** argv = nullptr;
  ros::init(
      argc,
      argv,
      "test_calc_svm_grad_" + bag_path.substr(0, bag_path.size() - std::string(".bag").size()));

  auto rmap_sampling = setupSVM<SamplingSpaceType>(bag_path);

  int test_num = 1000;
  for (int i = 0; i < test_num; i++) {
    Vel<SamplingSpaceType> svm_grad_analytical;
    Vel<SamplingSpaceType> svm_grad_numerical;
    rmap_sampling->testCalcSVMGrad(
        svm_grad_analytical,
        svm_grad_numerical,
        poseToSample<SamplingSpaceType>(getRandomPose<SamplingSpaceType>()));

    // std::cout << "[testCalcSVMGrad]" << std::endl;
    // std::cout << "  svm_grad_analytical: " << svm_grad_analytical.transpose() << std::endl;
    // std::cout << "  svm_grad_numerical: " << svm_grad_numerical.transpose() << std::endl;
    // std::cout << "  error: " << (svm_grad_analytical - svm_grad_numerical).norm() << std::endl;

    BOOST_CHECK((svm_grad_analytical - svm_grad_numerical).norm() < 1e-3);
  }
}

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradR2) { testCalcSVMGrad<SamplingSpace::R2>("rmap_sample_set_R2_test.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradSE2) { testCalcSVMGrad<SamplingSpace::SE2>("rmap_sample_set_SE2_test.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradR3) { testCalcSVMGrad<SamplingSpace::R3>("rmap_sample_set_R3_test.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradSE3) { testCalcSVMGrad<SamplingSpace::SE3>("rmap_sample_set_SE3_test.bag"); }

BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradR2IK) { testCalcSVMGrad<SamplingSpace::R2>("rmap_sample_set_R2_test_ik.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradSE2IK) { testCalcSVMGrad<SamplingSpace::SE2>("rmap_sample_set_SE2_test_ik.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradR3IK) { testCalcSVMGrad<SamplingSpace::R3>("rmap_sample_set_R3_test_ik.bag"); }
BOOST_AUTO_TEST_CASE(TestSVMUtilsCalcSVMGradSE3IK) { testCalcSVMGrad<SamplingSpace::SE3>("rmap_sample_set_SE3_test_ik.bag"); }

template <SamplingSpace SamplingSpaceType>
void testInputToVelMat()
{
  int test_num = 1000;
  for (int i = 0; i < test_num; i++) {
    sva::PTransformd pose = getRandomPose<SamplingSpaceType>();
    Sample<SamplingSpaceType> sample = poseToSample<SamplingSpaceType>(pose);

    InputToVelMat<SamplingSpaceType> mat_analytical = inputToVelMat<SamplingSpaceType>(sample);

    InputToVelMat<SamplingSpaceType> mat_numerical;
    double eps = 1e-6;
    for (int j = 0; j < velDim<SamplingSpaceType>(); j++) {
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

    BOOST_CHECK((mat_analytical - mat_numerical).norm() < 1e-8);
  }
}

BOOST_AUTO_TEST_CASE(TestInputToVelMatR2) { testInputToVelMat<SamplingSpace::R2>(); }
BOOST_AUTO_TEST_CASE(TestInputToVelMatSO2) { testInputToVelMat<SamplingSpace::SO2>(); }
BOOST_AUTO_TEST_CASE(TestInputToVelMatSE2) { testInputToVelMat<SamplingSpace::SE2>(); }
BOOST_AUTO_TEST_CASE(TestInputToVelMatR3) { testInputToVelMat<SamplingSpace::R3>(); }
BOOST_AUTO_TEST_CASE(TestInputToVelMatSO3) { testInputToVelMat<SamplingSpace::SO3>(); }
BOOST_AUTO_TEST_CASE(TestInputToVelMatSE3) { testInputToVelMat<SamplingSpace::SE3>(); }
