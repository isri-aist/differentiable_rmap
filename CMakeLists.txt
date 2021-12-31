cmake_minimum_required(VERSION 3.1)
project(differentiable_rmap)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rosbag
  rviz
  urdf
  tf2
  std_msgs
  std_srvs
  eigen_conversions
  geometry_msgs
  visualization_msgs
  sensor_msgs
  grid_map_ros
  optmotiongen
  )

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# SpaceVecAlg
find_package(SpaceVecAlg REQUIRED)

# RBDyn
find_package(RBDyn REQUIRED)

# sch-core
find_package(sch-core REQUIRED)

# mc_rtc
find_package(mc_rtc REQUIRED)

# eigen-quadprog
find_package(eigen-quadprog REQUIRED)

add_message_files(
  FILES
  RmapSample.msg
  RmapSampleSet.msg
  )

generate_messages(
  DEPENDENCIES std_msgs geometry_msgs visualization_msgs sensor_msgs
  )


catkin_package(
  CATKIN_DEPENDS
  message_runtime
  roscpp
  rosbag
  rviz
  urdf
  tf2
  std_msgs
  std_srvs
  eigen_conversions
  geometry_msgs
  visualization_msgs
  sensor_msgs
  grid_map_ros
  optmotiongen
  DEPENDS EIGEN3
  INCLUDE_DIRS include
  LIBRARIES OmgCore
  )

include_directories(include ${catkin_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(node)
