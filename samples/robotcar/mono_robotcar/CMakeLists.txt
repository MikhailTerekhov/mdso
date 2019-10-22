set(mono_robotcar_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/robotcar/mono_robotcar/main.cpp)
add_executable(mono_robotcar ${mono_robotcar_SOURCE_FILES})
target_link_libraries(mono_robotcar robotcar_reader)
