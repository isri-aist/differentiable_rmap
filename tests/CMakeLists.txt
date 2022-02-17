find_package(rostest REQUIRED)

set(differentiable_rmap_gtest_list
  TestSamplingUtils
  TestGridUtils
  TestBaselineUtils
  )

set(differentiable_rmap_rostest_list
  TestSVMUtils
  )

foreach(NAME IN LISTS differentiable_rmap_gtest_list)
  catkin_add_gtest(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} DiffRmap)
endforeach()

foreach(NAME IN LISTS differentiable_rmap_rostest_list)
  add_rostest_gtest(${NAME} ${NAME}.test ${NAME}.cpp)
  target_link_libraries(${NAME} DiffRmap)
endforeach()
