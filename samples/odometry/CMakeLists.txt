set(odometry_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/samples/odometry/main.cpp)
add_executable(odometry ${odometry_SOURCE_FILES})
target_link_libraries(odometry dso)
