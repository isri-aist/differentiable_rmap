/* Author: Masaki Murooka */


namespace DiffRmap
{
template <SamplingSpace SamplingSpaceType>
void setInputNode(
    svm_node* input_node,
    const Input<SamplingSpaceType>& input)
{
  for (int i = 0; i < inputDim<SamplingSpaceType>() + 1; i++) {
    if (i == inputDim<SamplingSpaceType>()) {
      input_node[i].index = -1; // last index must be -1
    } else {
      input_node[i].index = i + 1; // index starts from 1
      input_node[i].value = input[i];
    }
  }
}

template <SamplingSpace SamplingSpaceType>
void setInputNodeOnlyValue(
    svm_node* input_node,
    const Input<SamplingSpaceType>& input)
{
  for (int i = 0; i < inputDim<SamplingSpaceType>(); i++) {
    input_node[i].value = input[i];
  }
}

template <SamplingSpace SamplingSpaceType>
Input<SamplingSpaceType> svmNodeToEigenVec(
    const svm_node *input_node)
{
  Input<SamplingSpaceType> input;
  for (int i = 0; i < inputDim<SamplingSpaceType>(); i++) {
    input[i] = input_node[i].value;
  }
  return input;
}

template <SamplingSpace SamplingSpaceType>
double calcSVMValue(
    const Sample<SamplingSpaceType>& sample,
    const svm_parameter& svm_param,
    svm_model *svm_mo,
    const Eigen::VectorXd& svm_coeff_vec,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat)
{
  if (!(svm_mo->param.svm_type == ONE_CLASS || svm_mo->param.svm_type == NU_SVC)) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMValue] Only one-class or nu-svc SVM is supported: {}", svm_mo->param.svm_type);
  }

  if (svm_param.kernel_type != RBF) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMValue] Only RBF kernel is supported: {}", svm_param.kernel_type);
  }

  return svm_coeff_vec.dot(
      (-svm_param.gamma * (svm_sv_mat.colwise() - sampleToInput<SamplingSpaceType>(sample)).colwise().squaredNorm()).array().exp().matrix())
      - svm_mo->rho[0];
}

template <SamplingSpace SamplingSpaceType>
Vel<SamplingSpaceType> calcSVMGrad(
    const Sample<SamplingSpaceType>& sample,
    const svm_parameter& svm_param,
    svm_model *svm_mo,
    const Eigen::VectorXd& svm_coeff_vec,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat)
{
  if (!(svm_mo->param.svm_type == ONE_CLASS || svm_mo->param.svm_type == NU_SVC)) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMGrad] Only one-class or nu-svc SVM is supported: {}", svm_mo->param.svm_type);
  }

  if (svm_param.kernel_type != RBF) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMGrad] Only RBF kernel is supported: {}", svm_param.kernel_type);
  }

  Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic> sv_mat_minus_input =
      svm_sv_mat.colwise() - sampleToInput<SamplingSpaceType>(sample);

  return inputToVelMat<SamplingSpaceType>(sample) *
      2 * svm_param.gamma * sv_mat_minus_input *
      svm_coeff_vec.cwiseProduct((-svm_param.gamma * sv_mat_minus_input.colwise().squaredNorm()).array().exp().matrix().transpose());
}

template <SamplingSpace SamplingSpaceType>
Eigen::Matrix<double, velDim<SamplingSpaceType>(), inputDim<SamplingSpaceType>()>
inputToVelMat(const Sample<SamplingSpaceType>& sample)
{
  static_assert(velDim<SamplingSpaceType>() == inputDim<SamplingSpaceType>());

  return Eigen::Matrix<double, velDim<SamplingSpaceType>(), velDim<SamplingSpaceType>()>::Identity();
}

template <>
inline Eigen::Matrix<double, velDim<SamplingSpace::SO2>(), inputDim<SamplingSpace::SO2>()>
inputToVelMat<SamplingSpace::SO2>(const Sample<SamplingSpace::SO2>& sample)
{
  double cos = std::cos(sample.x());
  double sin = std::sin(sample.x());
  Eigen::Matrix<double, velDim<SamplingSpace::SO2>(), inputDim<SamplingSpace::SO2>()> mat;
  mat << -sin, -cos, cos, -sin;
  return mat;
}

template <>
inline Eigen::Matrix<double, velDim<SamplingSpace::SE2>(), inputDim<SamplingSpace::SE2>()>
inputToVelMat<SamplingSpace::SE2>(const Sample<SamplingSpace::SE2>& sample)
{
  Eigen::Matrix<double, velDim<SamplingSpace::SE2>(), inputDim<SamplingSpace::SE2>()> mat =
      Eigen::Matrix<double, velDim<SamplingSpace::SE2>(), inputDim<SamplingSpace::SE2>()>::Zero();
  mat.block<velDim<SamplingSpace::R2>(), inputDim<SamplingSpace::R2>()>(
      0, 0).diagonal().setConstant(1);
  mat.block<velDim<SamplingSpace::SO2>(), inputDim<SamplingSpace::SO2>()>(
      velDim<SamplingSpace::R2>(), inputDim<SamplingSpace::R2>()) =
            inputToVelMat<SamplingSpace::SO2>(sample.tail<sampleDim<SamplingSpace::SO2>()>());
  return mat;
}

template <>
inline Eigen::Matrix<double, velDim<SamplingSpace::SO3>(), inputDim<SamplingSpace::SO3>()>
inputToVelMat<SamplingSpace::SO3>(const Sample<SamplingSpace::SO3>& sample)
{
  return Eigen::Matrix<double, velDim<SamplingSpace::SO3>(), inputDim<SamplingSpace::SO3>()>::Zero();
}

template <>
inline Eigen::Matrix<double, velDim<SamplingSpace::SE3>(), inputDim<SamplingSpace::SE3>()>
inputToVelMat<SamplingSpace::SE3>(const Sample<SamplingSpace::SE3>& sample)
{
  return Eigen::Matrix<double, velDim<SamplingSpace::SE3>(), inputDim<SamplingSpace::SE3>()>::Zero();
}
}
