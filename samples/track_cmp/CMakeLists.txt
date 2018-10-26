set(track_cmp_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/track_cmp/main.cpp)
add_executable(track_cmp ${track_cmp_SOURCE_FILES})
target_link_libraries(track_cmp dso)
