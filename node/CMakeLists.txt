add_executable(NodeRmapSampling
  NodeRmapSampling.cpp)
target_link_libraries(NodeRmapSampling PRIVATE
  DiffRmap)

add_executable(NodeRmapSamplingFootstep
  NodeRmapSamplingFootstep.cpp)
target_link_libraries(NodeRmapSamplingFootstep PRIVATE
  DiffRmap)

add_executable(NodeRmapTraining
  NodeRmapTraining.cpp)
target_link_libraries(NodeRmapTraining PRIVATE
  DiffRmap)
