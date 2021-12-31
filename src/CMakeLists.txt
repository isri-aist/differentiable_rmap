add_library(DiffRmap
  SamplingUtils.cpp
  RmapSampling.cpp
  )
target_link_libraries(DiffRmap PUBLIC
  SpaceVecAlg::SpaceVecAlg
  RBDyn::RBDyn
  RBDyn::Parsers
  sch-core::sch-core
  mc_rtc::mc_rtc_utils
  ${catkin_LIBRARIES}
  )
