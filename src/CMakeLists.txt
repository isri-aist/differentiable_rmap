add_library(DiffRmap
  SamplingUtils.cpp
  BaselineUtils.cpp
  RmapSampling.cpp
  RmapSamplingIK.cpp
  RmapSamplingFootstep.cpp
  RmapTraining.cpp
  RmapVisualization.cpp
  RmapPlanning.cpp
  RmapPlanningFootstep.cpp
  RmapPlanningPlacement.cpp
  RmapPlanningMulticontact.cpp
  RmapPlanningLocomanip.cpp
  )
# Library depends on message generation
add_dependencies(DiffRmap
  ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(DiffRmap PUBLIC
  SpaceVecAlg::SpaceVecAlg
  RBDyn::RBDyn
  RBDyn::Parsers
  sch-core::sch-core
  mc_rtc::mc_rtc_utils
  svm
  ${catkin_LIBRARIES}
  )

target_compile_definitions(DiffRmap PUBLIC OPTMOTIONGEN_V2) # for optmotiongen-ver2

# target_compile_options(DiffRmap PUBLIC -march=native)
# target_link_options(DiffRmap PUBLIC -march=native)
