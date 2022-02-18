find_package(rostest REQUIRED)

set(differentiable_rmap_gtest_list
  TestSamplingUtils
  TestGridUtils
  TestBaselineUtils
  )

set(differentiable_rmap_rostest_list
  TestSVMUtils
  TestRmapSampling
  )

foreach(NAME IN LISTS differentiable_rmap_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} DiffRmap)
endforeach()

foreach(NAME IN LISTS differentiable_rmap_rostest_list)
  add_rostest_gtest(${NAME} test/${NAME}.test src/${NAME}.cpp)
  target_link_libraries(${NAME} DiffRmap)
endforeach()
