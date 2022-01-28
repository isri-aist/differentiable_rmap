#include <iostream>
#include <fstream>

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include <differentiable_rmap/BaselineUtils.h>

using namespace DiffRmap;


BOOST_AUTO_TEST_CASE(TestKNN)
{
  constexpr size_t N = 2;

  auto getClass = [](double x, double y) -> bool {
    return y < std::sin(4.0 * x);
  };

  // Generate training samples
  size_t train_sample_num = 1000;
  std::vector<Eigen::Matrix<double, N, 1>> train_sample_list(train_sample_num);
  std::vector<bool> class_list(train_sample_num);
  std::ofstream train_ofs("/tmp/train_data_knn.txt");
  for (size_t i = 0; i < train_sample_num; i++) {
    train_sample_list[i].setRandom();
    class_list[i] = getClass(train_sample_list[i].x(), train_sample_list[i].y());
    if (i % 50 == 0) {
      class_list[i] = !class_list[i];
    }
    train_ofs << train_sample_list[i].x() << " " << train_sample_list[i].y()
              << " " << (class_list[i] ? 1 : 0) << std::endl;
  }

  // Predict test samples
  size_t test_sample_num = 100;
  std::ofstream test_ofs("/tmp/test_data_knn.txt");
  for (size_t K : std::vector<size_t>{1, 3, 5, 7, 9}) {
    size_t failure_num = 0;
    for (size_t i = 0; i < test_sample_num; i++) {
      Eigen::Matrix<double, N, 1> test_sample = Eigen::Matrix<double, N, 1>::Random();
      bool class_gt = getClass(test_sample.x(), test_sample.y());
      bool class_pred = kNearestNeighbor<N>(test_sample, K, train_sample_list, class_list);
      if (class_gt != class_pred) {
        failure_num++;
      }
      test_ofs << test_sample.x() << " " << test_sample.y() << " " << (class_pred ? 1 : 0) << std::endl;
    }
    double failure_rate = static_cast<double>(failure_num) / test_sample_num;
    // std::cout << "[TestKNN] K: " << K << ", failure_rate: " << failure_rate << std::endl;
    BOOST_CHECK(failure_rate < 0.05);
  }

  // std::cout << "[TestKNN] Plot samples by the following gnuplot commands:" << std::endl;
  // std::cout << "  unset colorbox" << std::endl;
  // std::cout << "  set palette model RGB defined ( 0 'red', 1 'green' )" << std::endl;
  // std::cout << "  plot \"/tmp/train_data_knn.txt\" u 1:2:3 with points pt 7 ps 1 palette" << std::endl;
  // std::cout << "  replot \"/tmp/test_data_knn.txt\" u 1:2:3 with points pt 8 ps 1 palette" << std::endl;
  // std::cout << "  replot sin(4 * x) lw 4" << std::endl;
}
