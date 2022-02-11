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
Sample<SamplingSpaceType> calcSVMGrad(
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

  return inputToSampleMat<SamplingSpaceType>(sample) *
      2 * svm_param.gamma * sv_mat_minus_input *
      svm_coeff_vec.cwiseProduct((-svm_param.gamma * sv_mat_minus_input.colwise().squaredNorm()).array().exp().matrix().transpose());
}

template <SamplingSpace SamplingSpaceType>
InputToSampleMat<SamplingSpaceType> inputToSampleMat(const Sample<SamplingSpaceType>& sample)
{
  static_assert(sampleDim<SamplingSpaceType>() == inputDim<SamplingSpaceType>());

  return InputToSampleMat<SamplingSpaceType>::Identity();
}

template <>
inline InputToSampleMat<SamplingSpace::SO2> inputToSampleMat<SamplingSpace::SO2>(
    const Sample<SamplingSpace::SO2>& sample)
{
  double cos = std::cos(sample.x());
  double sin = std::sin(sample.x());
  InputToSampleMat<SamplingSpace::SO2> mat;
  mat << -sin, -cos, cos, -sin;
  return mat;
}

template <>
inline InputToSampleMat<SamplingSpace::SE2> inputToSampleMat<SamplingSpace::SE2>(
    const Sample<SamplingSpace::SE2>& sample)
{
  InputToSampleMat<SamplingSpace::SE2> mat = InputToSampleMat<SamplingSpace::SE2>::Zero();
  mat.block<sampleDim<SamplingSpace::R2>(), inputDim<SamplingSpace::R2>()>(0, 0) =
      inputToSampleMat<SamplingSpace::R2>(sample.head<sampleDim<SamplingSpace::R2>()>());
  mat.block<sampleDim<SamplingSpace::SO2>(), inputDim<SamplingSpace::SO2>()>(
      sampleDim<SamplingSpace::R2>(), inputDim<SamplingSpace::R2>()) =
      inputToSampleMat<SamplingSpace::SO2>(sample.tail<sampleDim<SamplingSpace::SO2>()>());
  return mat;
}

template <>
inline InputToSampleMat<SamplingSpace::SO3> inputToSampleMat<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample)
{
  double qw = sample.w();
  double qx = sample.x();
  double qy = sample.y();
  double qz = sample.z();

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

  return sample_to_input_mat.transpose();
}

template <>
inline InputToSampleMat<SamplingSpace::SE3> inputToSampleMat<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample)
{
  InputToSampleMat<SamplingSpace::SE3> mat = InputToSampleMat<SamplingSpace::SE3>::Zero();
  mat.block<sampleDim<SamplingSpace::R3>(), inputDim<SamplingSpace::R3>()>(0, 0) =
      inputToSampleMat<SamplingSpace::R3>(sample.head<sampleDim<SamplingSpace::R3>()>());
  mat.block<sampleDim<SamplingSpace::SO3>(), inputDim<SamplingSpace::SO3>()>(
      sampleDim<SamplingSpace::R3>(), inputDim<SamplingSpace::R3>()) =
      inputToSampleMat<SamplingSpace::SO3>(sample.tail<sampleDim<SamplingSpace::SO3>()>());
  return mat;
}

template <SamplingSpace SamplingSpaceType>
SampleToVelMat<SamplingSpaceType> sampleToVelMat(const Sample<SamplingSpaceType>& sample)
{
  static_assert(sampleDim<SamplingSpaceType>() == velDim<SamplingSpaceType>());

  return SampleToVelMat<SamplingSpaceType>::Identity();
}

template <>
inline SampleToVelMat<SamplingSpace::SO3> sampleToVelMat<SamplingSpace::SO3>(
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

  return vel_to_sample_mat.transpose() / 2;
}

template <>
inline SampleToVelMat<SamplingSpace::SE3> sampleToVelMat<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample)
{
  SampleToVelMat<SamplingSpace::SE3> mat = SampleToVelMat<SamplingSpace::SE3>::Zero();
  mat.block<velDim<SamplingSpace::R3>(), sampleDim<SamplingSpace::R3>()>(0, 0) =
      sampleToVelMat<SamplingSpace::R3>(sample.head<sampleDim<SamplingSpace::R3>()>());
  mat.block<velDim<SamplingSpace::SO3>(), sampleDim<SamplingSpace::SO3>()>(
      velDim<SamplingSpace::R3>(), sampleDim<SamplingSpace::R3>()) =
      sampleToVelMat<SamplingSpace::SO3>(sample.tail<sampleDim<SamplingSpace::SO3>()>());
  return mat;
}

template <SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> relSample(const Sample<SamplingSpaceType>& pre_sample,
                                    const Sample<SamplingSpaceType>& suc_sample)
{
    return sampleError<SamplingSpaceType>(pre_sample, suc_sample);
}

template <>
Sample<SamplingSpace::SO3> relSample<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& pre_sample,
    const Sample<SamplingSpace::SO3>& suc_sample)
{
  // In sampleError(), translation error is assumed to be represented in world frame.
  // In relSample(), on the other hand, it is assumed to be represented in pre_sample frame.
  // These assumptions lead to different results in SE2, SO3, and SE3, so sampleError() cannot be used.
  return (Eigen::Quaterniond(pre_sample.w(), pre_sample.x(), pre_sample.y(), pre_sample.z()).conjugate() *
          Eigen::Quaterniond(suc_sample.w(), suc_sample.x(), suc_sample.y(), suc_sample.z())).coeffs();
}

template <>
Sample<SamplingSpace::SE3> relSample<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& pre_sample,
    const Sample<SamplingSpace::SE3>& suc_sample)
{
  Sample<SamplingSpace::SE3> sample;
  sample <<
      Eigen::Quaterniond(pre_sample.tail<4>().w(),
                         pre_sample.tail<4>().x(),
                         pre_sample.tail<4>().y(),
                         pre_sample.tail<4>().z()).conjugate() *
      (suc_sample.head<3>() - pre_sample.head<3>()),
      (Eigen::Quaterniond(pre_sample.tail<4>().w(),
                          pre_sample.tail<4>().x(),
                          pre_sample.tail<4>().y(),
                          pre_sample.tail<4>().z()).conjugate() *
       Eigen::Quaterniond(suc_sample.tail<4>().w(),
                          suc_sample.tail<4>().x(),
                          suc_sample.tail<4>().y(),
                          suc_sample.tail<4>().z())).coeffs();
  return sample;

  // return poseToSample<SamplingSpaceType>(
  //     sampleToPose<SamplingSpaceType>(suc_sample) * sampleToPose<SamplingSpaceType>(pre_sample).inv());
}

template <>
inline Sample<SamplingSpace::SE2> relSample<SamplingSpace::SE2>(
    const Sample<SamplingSpace::SE2>& pre_sample,
    const Sample<SamplingSpace::SE2>& suc_sample)
{
  double cos = std::cos(pre_sample.z());
  double sin = std::sin(pre_sample.z());
  Vel<SamplingSpace::SE2> sample_error = sampleError<SamplingSpace::SE2>(pre_sample, suc_sample);

  Sample<SamplingSpace::SE2> rel_sample;
  rel_sample <<
      cos * sample_error.x() + sin * sample_error.y(),
      -sin * sample_error.x() + cos * sample_error.y(),
      sample_error.z();

  return rel_sample;
}

template <SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> midSample(const Sample<SamplingSpaceType>& sample1,
                                    const Sample<SamplingSpaceType>& sample2)
{
  if constexpr (SamplingSpaceType == SamplingSpace::R2 ||
                SamplingSpaceType == SamplingSpace::R3) {
      return (sample1 + sample2) / 2;
    } else {
    return poseToSample<SamplingSpaceType>(sva::interpolate(
        sampleToPose<SamplingSpaceType>(sample1), sampleToPose<SamplingSpaceType>(sample2), 0.5));
  }
}

template <SamplingSpace SamplingSpaceType>
SampleToSampleMat<SamplingSpaceType> relSampleToSampleMat(const Sample<SamplingSpaceType>& pre_sample,
                                                          const Sample<SamplingSpaceType>& suc_sample,
                                                          bool wrt_suc)
{
  if (wrt_suc) {
    return SampleToSampleMat<SamplingSpaceType>::Identity();
  } else {
    return -1 * SampleToSampleMat<SamplingSpaceType>::Identity();
  }
}

template <>
inline SampleToSampleMat<SamplingSpace::SE2> relSampleToSampleMat<SamplingSpace::SE2>(
    const Sample<SamplingSpace::SE2>& pre_sample,
    const Sample<SamplingSpace::SE2>& suc_sample,
    bool wrt_suc)
{
  double cos = std::cos(pre_sample.z());
  double sin = std::sin(pre_sample.z());

  SampleToSampleMat<SamplingSpace::SE2> mat;
  mat <<
      cos, sin, 0,
      -sin, cos, 0,
      0, 0, 1;

  if (!wrt_suc) {
    Vel<SamplingSpace::SE2> sample_error = sampleError<SamplingSpace::SE2>(pre_sample, suc_sample);
    mat *= -1;
    mat(0, 2) = -sin * sample_error.x() + cos * sample_error.y();
    mat(1, 2) = -cos * sample_error.x() - sin * sample_error.y();
  }

  return mat;
}

template <>
inline SampleToSampleMat<SamplingSpace::SO3> relSampleToSampleMat<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& pre_sample,
    const Sample<SamplingSpace::SO3>& suc_sample,
    bool wrt_suc)
{
  double qx1 = pre_sample.x();
  double qy1 = pre_sample.y();
  double qz1 = pre_sample.z();
  double qw1 = pre_sample.w();
  double qx2 = suc_sample.x();
  double qy2 = suc_sample.y();
  double qz2 = suc_sample.z();
  double qw2 = suc_sample.w();

  SampleToSampleMat<SamplingSpace::SO3> mat;

  if (wrt_suc) {
    mat <<
        qw1, qz1, -qy1, -qx1,
        -qz1, qw1, qx1, -qy1,
        qy1, -qx1, qw1, -qz1,
        qx1, qy1, qz1, qw1;
  } else {
    mat <<
        -qw2, -qz2, qy2, qx2,
        qz2, -qw2, -qx2, qy2,
        -qy2, qx2, -qw2, qz2,
        qx2, qy2, qz2, qw2;
  }

  return mat;
}

template <>
inline SampleToSampleMat<SamplingSpace::SE3> relSampleToSampleMat<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& pre_sample,
    const Sample<SamplingSpace::SE3>& suc_sample,
    bool wrt_suc)
{
  SampleToSampleMat<SamplingSpace::SE3> mat = SampleToSampleMat<SamplingSpace::SE3>::Zero();

  mat.block<sampleDim<SamplingSpace::R3>(), sampleDim<SamplingSpace::R3>()>(0, 0) =
      (wrt_suc ? 1 : -1) * Eigen::Quaterniond(
          pre_sample.tail<4>().w(),
          pre_sample.tail<4>().x(),
          pre_sample.tail<4>().y(),
          pre_sample.tail<4>().z()).toRotationMatrix().transpose();
  mat.block<sampleDim<SamplingSpace::SO3>(), sampleDim<SamplingSpace::SO3>()>(
      sampleDim<SamplingSpace::R3>(), sampleDim<SamplingSpace::R3>()) =
      relSampleToSampleMat<SamplingSpace::SO3>(pre_sample.tail<4>(), suc_sample.tail<4>(), wrt_suc);

  if (!wrt_suc) {
    double qx = pre_sample.tail<4>().x();
    double qy = pre_sample.tail<4>().y();
    double qz = pre_sample.tail<4>().z();
    double qw = pre_sample.tail<4>().w();

    Eigen::MatrixXd quat_mat(9, 4);
    quat_mat <<
        0, -2*qy, -2*qz, 0,
        qy, qx, -qw, -qz,
        qz, qw, qx, qy,
        qy, qx, qw, qz,
        -2*qx, 0, -2*qz, 0,
        -qw, qz, qy, -qx,
        qz, -qw, qx, -qy,
        qw, qz, qy, qx,
        -2*qx, -2*qy, 0, 0;
    quat_mat *= 2;

    Eigen::Vector3d rel_pos = suc_sample.head<3>() - pre_sample.head<3>();

    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        mat.block<sampleDim<SamplingSpace::R3>(), sampleDim<SamplingSpace::SO3>()>(
            0, sampleDim<SamplingSpace::R3>()).row(j) += rel_pos[k] * quat_mat.row(j + k * 3);
      }
    }
  }

  return mat;
}
}
