# differentiable_rmap
Differentiable reachability map

[![CI](https://github.com/isri-aist/differentiable_rmap/actions/workflows/ci.yaml/badge.svg)](https://github.com/isri-aist/differentiable_rmap/actions/workflows/ci.yaml)

## Install

### Dependencies

#### Packages not supported by rosdep
- [mc_rtc](https://github.com/jrl-umi3218/mc_rtc)
- [jrl-qp (topic/BlockStructure branch)](https://github.com/jrl-umi3218/jrl-qp/tree/topic/BlockStructure)
- [optmotiongen](https://github.com/isri-aist/optmotiongen)

#### Packages supported by rosdep
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)


## Samples

### Sample set generation

#### FK-based sampling
```bash
$ roslaunch differentiable_rmap rmap_sampling_simple_2dof_manipulator.launch
```

#### IK-based sampling
```bash
$ roslaunch differentiable_rmap rmap_sampling_simple_2dof_manipulator.launch use_ik:=true
```
