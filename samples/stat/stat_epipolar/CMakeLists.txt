set(stat_epipolar_SOURCE_FILES
        main.cpp)
add_executable(stat_epipolar ${stat_epipolar_SOURCE_FILES})
target_link_libraries(stat_epipolar dso)

