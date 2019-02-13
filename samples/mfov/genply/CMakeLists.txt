set(genply_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/mfov/genply/main.cpp)
  add_executable(genply ${genply_SOURCE_FILES})
target_link_libraries(genply reader)
target_link_libraries(genply dso)

