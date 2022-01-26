set(differentiable_rmap_node_list
  NodeRmapSampling
  NodeRmapSamplingFootstep
  NodeRmapSamplingLocomanip
  NodeRmapTraining
  NodeRmapVisualization
  NodeRmapPlanning
  NodeRmapPlanningFootstep
  NodeRmapPlanningPlacement
  NodeRmapPlanningMulticontact
  NodeRmapPlanningLocomanip
  )

foreach(NAME IN LISTS differentiable_rmap_node_list)
  add_executable(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} PRIVATE DiffRmap)
endforeach()
