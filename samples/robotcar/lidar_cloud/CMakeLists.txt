set(lidar_cloud_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/robotcar/lidar_cloud/main.cpp)
  add_executable(lidar_cloud ${lidar_cloud_SOURCE_FILES})
target_link_libraries(lidar_cloud robotcar_reader)
