set(depthmaps_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/samples/mfov_render/depthmaps/main.cpp)
add_executable(depthmaps ${depthmaps_SOURCE_FILES})
target_link_libraries(depthmaps dso)
