add_executable(NodeRmapSampling
  NodeRmapSampling.cpp)
target_link_libraries(NodeRmapSampling PRIVATE
  DiffRmap)

add_executable(NodeRmapTraining
  NodeRmapTraining.cpp)
target_link_libraries(NodeRmapTraining PRIVATE
  DiffRmap)
