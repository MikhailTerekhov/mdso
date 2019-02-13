set(stat_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/mfov/stat/main.cpp)
add_executable(stat ${stat_SOURCE_FILES})
target_link_libraries(stat reader)
target_link_libraries(stat dso)

