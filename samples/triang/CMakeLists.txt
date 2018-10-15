set(triang_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/triang/main.cpp)
add_executable(triang ${triang_SOURCE_FILES})
target_link_libraries(triang dso)
