set(distmap_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/distmap/main.cpp)
add_executable(distmap ${distmap_SOURCE_FILES})

target_link_libraries(distmap dso)

