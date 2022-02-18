# differentiable_rmap

[![CI](https://github.com/isri-aist/differentiable_rmap/actions/workflows/ci.yaml/badge.svg)](https://github.com/isri-aist/differentiable_rmap/actions/workflows/ci.yaml)
[![Documentation](https://img.shields.io/badge/doxygen-online-brightgreen?logo=read-the-docs&style=flat)](https://isri-aist.github.io/differentiable_rmap/)

![eval-all](doc/images/eval-all.gif)

## Summary
This is a library for representing the kinematic reachability of robots: differentiable reachability map.

This is a scalar-valued function in task space that is positive only in the region reachable by the robot's end-effector. The main feature is that the scalar-valued function is continuous and differentiable with respect to task-space coordinates. This allows us to formulate the reachability conditions of the robot's end-effectors using reachability maps in continuous optimization for motion planning. The differentiable reachability map is learned using a support vector machine from a sample set of end-effector poses generated from a robot kinematic model.

## Install

### Dependencies

#### Packages not supported by rosdep
- [mc_rtc](https://jrl-umi3218.github.io/mc_rtc)
- [jrl-qp (topic/BlockStructure branch)](https://github.com/jrl-umi3218/jrl-qp/tree/topic/BlockStructure)
- [optmotiongen](https://github.com/isri-aist/optmotiongen)

#### Packages supported by rosdep
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

### Installation procedure
It is assumed that ROS is installed.

1. Install mc_rtc

Installation via apt is recommended. See [here](https://jrl-umi3218.github.io/mc_rtc/tutorials/introduction/installation-guide.html#ubuntu-lts-1804-2004).

2. Install jrl-qp

```bash
$ git clone https://github.com/jrl-umi3218/jrl-qp -b topic/BlockStructure --recursive
$ mkdir build
$ cd build
$ cmake .. -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF
$ make
$ make install
```

3. Setup catkin workspace and build
```bash
$ mkdir -p ~/ros/ws_differentiable_rmap/src
$ cd ~/ros/ws_differentiable_rmap
$ wstool init src
$ wstool set -t src isri-aist/optmotiongen git@github.com:isri-aist/optmotiongen.git -v ver2 --git -y
$ wstool set -t src isri-aist/differentiable_rmap git@github.com:isri-aist/differentiable_rmap.git --git -y
$ wstool update -t src

$ source /opt/ros/${ROS_DISTRO}/setup.bash
$ rosdep install -y -r --from-paths src --ignore-src

$ catkin build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_JRLQP=ON
```

## Example with simple 2D manipulator
You can reproduce the results of this [video](https://www.dropbox.com/s/7vp0zq7yxj37t5v/eval-all.mp4?dl=0).

### Sample set generation
Run either FK-based or IK-based sampling.

FK-based sampling:
```bash
$ roslaunch differentiable_rmap rmap_sampling_simple_2dof_manipulator.launch
```

IK-based sampling:
```bash
$ roslaunch differentiable_rmap rmap_sampling_simple_2dof_manipulator.launch use_ik:=true
```

### Reachability map learning
```bash
$ roslaunch differentiable_rmap rmap_training.launch sampling_space:=R2
```

### Saving a grid set of reachability map
```bash
$ roslaunch differentiable_rmap rmap_visualization.launch sampling_space:=R2
```

### Optimization with reachability constraint
```bash
$ roslaunch differentiable_rmap rmap_planning.launch sampling_space:=R2
```

## Standalone script for scalar field learning examples
```bash
$ rosrun differentiable_rmap JointSpaceUniformSampling.py
$ rosrun differentiable_rmap TaskSpaceDensityEstimation.py
```
The following image will be displayed.

![TaskSpaceDensityEstimation](doc/images/TaskSpaceDensityEstimation.png)
