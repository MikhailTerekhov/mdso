set(kitti_HEADER_FILES
  ${PROJECT_SOURCE_DIR}/samples/kitti/KittiReader.h)
set(kitti_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/kitti/main.cpp
    ${PROJECT_SOURCE_DIR}/samples/kitti/KittiReader.cpp)
add_executable(kitti ${kitti_HEADER_FILES} ${kitti_SOURCE_FILES})

target_link_libraries(kitti dso)

