set(stat_epipolar_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/mfov/stat/stat_epipolar/main.cpp)
  add_executable(stat_epipolar ${stat_epipolar_SOURCE_FILES})
target_link_libraries(stat_epipolar reader)
target_link_libraries(stat_epipolar dso)

