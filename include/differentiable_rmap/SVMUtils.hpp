/* Author: Masaki Murooka */


namespace DiffRmap
{
template <SamplingSpace SamplingSpaceType>
void setSVMPredictionMat(
    Eigen::Ref<Eigen::VectorXd> svm_coeff_vec,
    Eigen::Ref<Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>> svm_sv_mat,
    svm_model* svm_mo)
{
  int num_sv = svm_mo->l;
  for (int i = 0; i < num_sv; i++) {
    svm_coeff_vec[i] = svm_mo->sv_coef[0][i];
    svm_sv_mat.col(i) = svmNodeToEigenVec<SamplingSpaceType>(svm_mo->SV[i]);
  }
}

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
InputToVelMat<SamplingSpaceType> inputToVelMat(const Sample<SamplingSpaceType>& sample)
{
  static_assert(velDim<SamplingSpaceType>() == inputDim<SamplingSpaceType>());

  return InputToVelMat<SamplingSpaceType>::Identity();
}

template <>
inline InputToVelMat<SamplingSpace::SO2> inputToVelMat<SamplingSpace::SO2>(
    const Sample<SamplingSpace::SO2>& sample)
{
  double cos = std::cos(sample.x());
  double sin = std::sin(sample.x());
  InputToVelMat<SamplingSpace::SO2> mat;
  mat << -sin, -cos, cos, -sin;
  return mat;
}

template <>
inline InputToVelMat<SamplingSpace::SE2> inputToVelMat<SamplingSpace::SE2>(
    const Sample<SamplingSpace::SE2>& sample)
{
  InputToVelMat<SamplingSpace::SE2> mat = InputToVelMat<SamplingSpace::SE2>::Zero();
  mat.block<velDim<SamplingSpace::R2>(), inputDim<SamplingSpace::R2>()>(0, 0) =
      inputToVelMat<SamplingSpace::R2>(sample.head<sampleDim<SamplingSpace::R2>()>());
  mat.block<velDim<SamplingSpace::SO2>(), inputDim<SamplingSpace::SO2>()>(
      velDim<SamplingSpace::R2>(), inputDim<SamplingSpace::R2>()) =
      inputToVelMat<SamplingSpace::SO2>(sample.tail<sampleDim<SamplingSpace::SO2>()>());
  return mat;
}

template <>
inline InputToVelMat<SamplingSpace::SO3> inputToVelMat<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample)
{
  double qw = sample.w();
  double qx = sample.x();
  double qy = sample.y();
  double qz = sample.z();

  Eigen::Matrix<double, sampleDim<SamplingSpace::SO3>(), velDim<SamplingSpace::SO3>()>
      vel_to_sample_mat; // 4 x 3 matrix
  vel_to_sample_mat <<
      -qx, -qy, -qz, qw, -qz, qy, qz, qw, -qx, -qy, qx, qw;

  Eigen::Matrix<double, inputDim<SamplingSpace::SO3>(), sampleDim<SamplingSpace::SO3>()>
      sample_to_input_mat; // 9 x 4 matrix
  sample_to_input_mat <<
      0, 0, -2*qy, -2*qz,
      -qz, qy, qx, -qw,
      qy, qz, qw, qx,
      qz, qy, qx, qw,
      0, -2*qx, 0, -2*qz,
      -qx, -qw, qz, qy,
      -qy, qz, -qw, qx,
      qx, qw, qz, qy,
      0, -2*qx, -2*qy, 0;
  sample_to_input_mat *= 2;

  return vel_to_sample_mat.transpose() * sample_to_input_mat.transpose() / 2;
}

template <>
inline InputToVelMat<SamplingSpace::SE3> inputToVelMat<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample)
{
  InputToVelMat<SamplingSpace::SE3> mat = InputToVelMat<SamplingSpace::SE3>::Zero();
  mat.block<velDim<SamplingSpace::R3>(), inputDim<SamplingSpace::R3>()>(0, 0) =
      inputToVelMat<SamplingSpace::R3>(sample.head<sampleDim<SamplingSpace::R3>()>());
  mat.block<velDim<SamplingSpace::SO3>(), inputDim<SamplingSpace::SO3>()>(
      velDim<SamplingSpace::R3>(), inputDim<SamplingSpace::R3>()) =
      inputToVelMat<SamplingSpace::SO3>(sample.tail<sampleDim<SamplingSpace::SO3>()>());
  return mat;
}

template <SamplingSpace SamplingSpaceType>
void setSVMIneq(Eigen::Ref<Eigen::MatrixXd> ineq_mat,
                Eigen::Ref<Eigen::MatrixXd> ineq_vec,
                const Sample<SamplingSpaceType>& sample,
                const svm_parameter& svm_param,
                svm_model *svm_mo,
                const Eigen::VectorXd& svm_coeff_vec,
                const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat,
                double svm_thre)
{
  // There is a problem with receiving a fixed size block matrix with Ref, so we receive a dynamic size matrix.
  // See https://stackoverflow.com/a/54966664
  assert(ineq_mat.rows() == 1);
  assert(ineq_mat.cols() == velDim<SamplingSpaceType>());
  assert(ineq_vec.rows() == 1);
  assert(ineq_vec.cols() == 1);

  // input_mat is row vector
  ineq_mat = -1 * calcSVMGrad<SamplingSpaceType>(sample, svm_param, svm_mo, svm_coeff_vec, svm_sv_mat).transpose();
  // input_vec is 1 x 1 matrix
  ineq_vec << calcSVMValue<SamplingSpaceType>(sample, svm_param, svm_mo, svm_coeff_vec, svm_sv_mat) - svm_thre;
}
}
