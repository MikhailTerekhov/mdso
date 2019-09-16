set(cam_test_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/robotcar/cam_test/main.cpp)
  add_executable(cam_test ${cam_test_SOURCE_FILES})
target_link_libraries(cam_test dso)
