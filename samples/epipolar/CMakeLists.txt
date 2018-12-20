set(epipolar_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/epipolar/main.cpp)
add_executable(epipolar ${epipolar_SOURCE_FILES})
target_link_libraries(epipolar dso)
