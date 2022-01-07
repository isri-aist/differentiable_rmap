find_package(Boost REQUIRED COMPONENTS unit_test_framework)

set(differentiable_rmap_test_list
  TestSamplingUtils
  )

foreach(NAME IN LISTS differentiable_rmap_test_list)
  add_executable(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} PUBLIC Boost::unit_test_framework Boost::disable_autolinking DiffRmap)
  target_compile_definitions(${NAME} PRIVATE -DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)
  add_test(${NAME} ${NAME})
endforeach()
