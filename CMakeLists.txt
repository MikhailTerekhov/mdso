cmake_minimum_required(VERSION 2.8)

project(dso-fisheye-ceres)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
)

set(dso_HEADER_FILES
    ${PROJECT_SOURCE_DIR}/src/util/defs.h
    ${PROJECT_SOURCE_DIR}/src/util/settings.h
    ${PROJECT_SOURCE_DIR}/src/frontend/frontend.h
    ${PROJECT_SOURCE_DIR}/src/system/dsosystem.h
    ${PROJECT_SOURCE_DIR}/src/system/keyframe.h
    ${PROJECT_SOURCE_DIR}/src/system/interestpoint.h
)

set(dso_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
    ${PROJECT_SOURCE_DIR}/src/util/util.cpp
    ${PROJECT_SOURCE_DIR}/src/frontend/frontend.cpp
    ${PROJECT_SOURCE_DIR}/src/system/dsosystem.cpp
    ${PROJECT_SOURCE_DIR}/src/system/keyframe.cpp
    ${PROJECT_SOURCE_DIR}/src/system/interestpoint.cpp
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

add_executable(dso ${dso_SOURCE_FILES} ${dso_HEADER_FILES})
target_compile_definitions(dso PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_link_libraries(dso ${OpenCV_LIBS} ${CERES_LIBRARIES})
