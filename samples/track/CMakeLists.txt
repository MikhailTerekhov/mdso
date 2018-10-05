set(track_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/track/main.cpp)
add_executable(track ${track_SOURCE_FILES})
target_link_libraries(track ${OpenCV_LIBS} dso)
