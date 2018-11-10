set(multi_fov_HEADER_FILES
  ${PROJECT_SOURCE_DIR}/samples/multi_fov/MultiFovReader.h)
set(multi_fov_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/multi_fov/main.cpp
    ${PROJECT_SOURCE_DIR}/samples/multi_fov/MultiFovReader.cpp)
add_executable(multi_fov ${multi_fov_HEADER_FILES} ${multi_fov_SOURCE_FILES})

target_link_libraries(multi_fov dso)

