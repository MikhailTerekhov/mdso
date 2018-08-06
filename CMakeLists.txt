cmake_minimum_required(VERSION 2.8)

project(dso-fisheye-ceres)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

set(dso_HEADER_FILES
)

set(dso_SOURCE_FILES 
)

add_executable(dso ${dso_SOURCE_FILES} ${dso_HEADER_FILES})
target_compile_definitions(dso PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_link_libraries(dso ${OpenCV_LIBS} ${CERES_LIBRARIES})
