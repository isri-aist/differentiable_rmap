#include <fstream>
#include <iostream>

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include <differentiable_rmap/BaselineUtils.h>

using namespace DiffRmap;

BOOST_AUTO_TEST_CASE(TestOCNN)
{
  constexpr size_t N = 2;

  auto getClass = [](double x, double y) -> bool { return y < std::sin(4.0 * x); };

  srand(1);

  // Generate training samples
  size_t train_sample_num = 1000;
  std::vector<Eigen::Matrix<double, N, 1>> train_sample_list(train_sample_num);
  std::ofstream train_ofs("/tmp/train_data_ocnn.txt");
  for(size_t i = 0; i < train_sample_num; i++)
  {
    do
    {
      train_sample_list[i].setRandom();
    } while(!getClass(train_sample_list[i].x(), train_sample_list[i].y()));
    train_ofs << train_sample_list[i].x() << " " << train_sample_list[i].y() << " 1" << std::endl;
  }

  // Predict test samples
  size_t test_sample_num = 100;
  std::vector<double> dist_ratio_thre_list = {0.5, 1.0, 2.0, 3.0};
  for(size_t i = 0; i < dist_ratio_thre_list.size(); i++)
  {
    std::ofstream test_ofs("/tmp/test_data_ocnn_" + std::to_string(i) + ".txt");
    double dist_ratio_thre = dist_ratio_thre_list[i];
    size_t failure_num = 0;
    for(size_t j = 0; j < test_sample_num; j++)
    {
      Eigen::Matrix<double, N, 1> test_sample = Eigen::Matrix<double, N, 1>::Random();
      bool class_gt = getClass(test_sample.x(), test_sample.y());
      bool class_pred = oneClassNearestNeighbor<N>(test_sample, dist_ratio_thre, train_sample_list);
      if(class_gt != class_pred)
      {
        failure_num++;
      }
      test_ofs << test_sample.x() << " " << test_sample.y() << " " << (class_pred ? 1 : 0) << std::endl;
    }
    double failure_rate = static_cast<double>(failure_num) / test_sample_num;
    // std::cout << "[TestOCNN] thre: " << dist_ratio_thre << ", failure_rate: " << failure_rate << std::endl;
    BOOST_CHECK(failure_rate < 0.3);
  }

  // std::cout << "[TestOCNN] Plot samples by the following gnuplot commands:" << std::endl;
  // std::cout << "  unset colorbox" << std::endl;
  // std::cout << "  set palette model RGB defined ( 0 'red', 1 'green' )" << std::endl;
  // std::cout << "  plot \"/tmp/train_data_ocnn.txt\" u 1:2:3 with points pt 7 ps 1 palette" << std::endl;
  // std::cout << "  replot \"/tmp/test_data_ocnn_0.txt\" u 1:2:3 with points pt 4 ps 2 palette" << std::endl;
  // std::cout << "  replot sin(4 * x) lw 4" << std::endl;
}

BOOST_AUTO_TEST_CASE(TestKNN)
{
  constexpr size_t N = 2;

  auto getClass = [](double x, double y) -> bool { return y < std::sin(4.0 * x); };

  srand(1);

  // Generate training samples
  size_t train_sample_num = 1000;
  std::vector<Eigen::Matrix<double, N, 1>> train_sample_list(train_sample_num);
  std::vector<bool> class_list(train_sample_num);
  std::ofstream train_ofs("/tmp/train_data_knn.txt");
  for(size_t i = 0; i < train_sample_num; i++)
  {
    train_sample_list[i].setRandom();
    class_list[i] = getClass(train_sample_list[i].x(), train_sample_list[i].y());
    if(i % 100 == 0)
    {
      class_list[i] = !class_list[i];
    }
    train_ofs << train_sample_list[i].x() << " " << train_sample_list[i].y() << " " << (class_list[i] ? 1 : 0)
              << std::endl;
  }

  // Predict test samples
  size_t test_sample_num = 100;
  std::vector<size_t> K_list = {1, 3, 5, 7, 9};
  for(size_t i = 0; i < K_list.size(); i++)
  {
    std::ofstream test_ofs("/tmp/test_data_knn_" + std::to_string(i) + ".txt");
    size_t K = K_list[i];
    size_t failure_num = 0;
    for(size_t j = 0; j < test_sample_num; j++)
    {
      Eigen::Matrix<double, N, 1> test_sample = Eigen::Matrix<double, N, 1>::Random();
      bool class_gt = getClass(test_sample.x(), test_sample.y());
      bool class_pred = kNearestNeighbor<N>(test_sample, K, train_sample_list, class_list);
      if(class_gt != class_pred)
      {
        failure_num++;
      }
      test_ofs << test_sample.x() << " " << test_sample.y() << " " << (class_pred ? 1 : 0) << std::endl;
    }
    double failure_rate = static_cast<double>(failure_num) / test_sample_num;
    // std::cout << "[TestKNN] K: " << K << ", failure_rate: " << failure_rate << std::endl;
    BOOST_CHECK(failure_rate < 0.1);
  }

  // std::cout << "[TestKNN] Plot samples by the following gnuplot commands:" << std::endl;
  // std::cout << "  unset colorbox" << std::endl;
  // std::cout << "  set palette model RGB defined ( 0 'red', 1 'green' )" << std::endl;
  // std::cout << "  plot \"/tmp/train_data_knn.txt\" u 1:2:3 with points pt 7 ps 1 palette" << std::endl;
  // std::cout << "  replot \"/tmp/test_data_knn_0.txt\" u 1:2:3 with points pt 4 ps 2 palette" << std::endl;
  // std::cout << "  replot sin(4 * x) lw 4" << std::endl;
}

BOOST_AUTO_TEST_CASE(TestConvexInsideClassification)
{
  std::ofstream ofs("/tmp/data_convex_inside_classification.txt");

  // Generate training points
  std::vector<Eigen::Vector2d> points;
  for(int i = 0; i < 20; i++)
  {
    const Eigen::Vector2d & point = Eigen::Vector2d::Random();
    points.push_back(point);
    ofs << point.transpose() << std::endl;
  }
  ofs << std::endl;

  // Generate classification instance
  ConvexInsideClassification convex_inside_class(points);

  // Dump convex vertices
  for(const auto & vertex : convex_inside_class.convex_vertices_)
  {
    ofs << vertex.transpose() << std::endl;
  }
  ofs << std::endl;

  // Check inside convex
  for(int i = 0; i < 1000; i++)
  {
    Eigen::Vector2d point = Eigen::Vector2d::Random();
    ofs << point.transpose() << " " << static_cast<int>(convex_inside_class.classify(point)) << std::endl;
  }

  BOOST_CHECK(convex_inside_class.classify(Eigen::Vector2d::Zero()));
  BOOST_CHECK(!convex_inside_class.classify(Eigen::Vector2d(1.1, 0)));
  BOOST_CHECK(!convex_inside_class.classify(Eigen::Vector2d(-1.1, 0)));
  BOOST_CHECK(!convex_inside_class.classify(Eigen::Vector2d(0, 1.1)));
  BOOST_CHECK(!convex_inside_class.classify(Eigen::Vector2d(0, -1.1)));

  // std::cout
  //     << "[TestConvexInsideClassification] Plot samples by the following gnuplot commands:" << std::endl
  //     << "  set palette model RGB defined ( 0 'red', 1 'green' )" << std::endl
  //     << "  unset colorbox" << std::endl
  //     << "  plot \"/tmp/data_convex_inside_classification.txt\" every :::0::0 using 1:2 t \"train\" pt 4 ps 2 lc rgb
  //     \"blue\"" << std::endl
  //     << "  replot \"\" every :::1::1 u 1:2 with lines t \"convex\" lw 2 lc rgb \"blue\"" << std::endl
  //     << "  replot \"\" every :::2::2 u 1:2:3 t \"test\" pt 7 ps 2 palette" << std::endl;
}
