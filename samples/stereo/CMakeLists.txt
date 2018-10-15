set(stereo_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/stereo/main.cpp)
add_executable(stereo ${stereo_SOURCE_FILES})

target_link_libraries(stereo dso)
