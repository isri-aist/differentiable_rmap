set(differentiable_rmap_test_list
  TestSamplingUtils
  # TestSVMUtils
  TestGridUtils
  TestBaselineUtils
  )

foreach(NAME IN LISTS differentiable_rmap_test_list)
  catkin_add_gtest(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} DiffRmap)
endforeach()
