set(selectpix_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/selectpix/main.cpp)
add_executable(selectpix ${selectpix_SOURCE_FILES})

target_link_libraries(selectpix dso)
