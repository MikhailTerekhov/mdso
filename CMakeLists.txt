cmake_minimum_required(VERSION 2.8)

project(dso-fisheye-ceres)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/googletest)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver
)

set(dso_HEADER_FILES
    ${PROJECT_SOURCE_DIR}/src/util/util.h
    ${PROJECT_SOURCE_DIR}/src/util/types.h
    ${PROJECT_SOURCE_DIR}/src/util/defs.h
    ${PROJECT_SOURCE_DIR}/src/util/settings.h
    ${PROJECT_SOURCE_DIR}/src/util/triangulation.h
    ${PROJECT_SOURCE_DIR}/src/util/sphericaltriangulation.h
    ${PROJECT_SOURCE_DIR}/src/util/sphericalterrain.h
    ${PROJECT_SOURCE_DIR}/src/system/terrain.h
    ${PROJECT_SOURCE_DIR}/src/system/dsosystem.h
    ${PROJECT_SOURCE_DIR}/src/system/dsoinitializer.h
    ${PROJECT_SOURCE_DIR}/src/system/keyframe.h
    ${PROJECT_SOURCE_DIR}/src/system/interestpoint.h
    ${PROJECT_SOURCE_DIR}/src/system/cameramodel.h
    ${PROJECT_SOURCE_DIR}/src/system/stereomatcher.h
    ${PROJECT_SOURCE_DIR}/src/system/stereogeometryestimator.h
)

set(dso_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/util/util.cpp
    ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
    ${PROJECT_SOURCE_DIR}/src/util/triangulation.cpp
    ${PROJECT_SOURCE_DIR}/src/util/sphericaltriangulation.cpp
    ${PROJECT_SOURCE_DIR}/src/util/sphericalterrain.cpp
    ${PROJECT_SOURCE_DIR}/src/system/terrain.cpp
    ${PROJECT_SOURCE_DIR}/src/system/dsosystem.cpp
    ${PROJECT_SOURCE_DIR}/src/system/dsoinitializer.cpp
    ${PROJECT_SOURCE_DIR}/src/system/keyframe.cpp
    ${PROJECT_SOURCE_DIR}/src/system/cameramodel.cpp
    ${PROJECT_SOURCE_DIR}/src/system/stereomatcher.cpp
    ${PROJECT_SOURCE_DIR}/src/system/stereogeometryestimator.cpp
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

set(essietial_5pt_estimator_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver/relativeposeestimator_double.cpp
)

add_library(dso ${dso_SOURCE_FILES} ${dso_HEADER_FILES}
    ${essietial_5pt_estimator_SOURCE_FILES}
)
target_compile_definitions(dso PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_link_libraries(dso ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_subdirectory(samples)

enable_testing()
add_subdirectory(test)
