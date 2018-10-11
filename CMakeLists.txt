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
    ${PROJECT_SOURCE_DIR}/src/util/Triangulation.h
    ${PROJECT_SOURCE_DIR}/src/util/Terrain.h
    ${PROJECT_SOURCE_DIR}/src/util/SphericalTriangulation.h
    ${PROJECT_SOURCE_DIR}/src/util/SphericalTerrain.h
    ${PROJECT_SOURCE_DIR}/src/system/AffineLightTransform.h
    ${PROJECT_SOURCE_DIR}/src/system/DsoSystem.h
    ${PROJECT_SOURCE_DIR}/src/system/DsoInitializer.h
    ${PROJECT_SOURCE_DIR}/src/system/PreKeyFrame.h
    ${PROJECT_SOURCE_DIR}/src/system/KeyFrame.h
    ${PROJECT_SOURCE_DIR}/src/system/InterestPoint.h
    ${PROJECT_SOURCE_DIR}/src/system/CameraModel.h
    ${PROJECT_SOURCE_DIR}/src/system/StereoMatcher.h
    ${PROJECT_SOURCE_DIR}/src/system/StereoGeometryEstimator.h
    ${PROJECT_SOURCE_DIR}/src/system/FrameTracker.h
)

set(dso_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/util/util.cpp
    ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
    ${PROJECT_SOURCE_DIR}/src/util/Triangulation.cpp
    ${PROJECT_SOURCE_DIR}/src/util/Terrain.cpp
    ${PROJECT_SOURCE_DIR}/src/util/SphericalTriangulation.cpp
    ${PROJECT_SOURCE_DIR}/src/util/SphericalTerrain.cpp
    ${PROJECT_SOURCE_DIR}/src/system/DsoSystem.cpp
    ${PROJECT_SOURCE_DIR}/src/system/DsoInitializer.cpp
    ${PROJECT_SOURCE_DIR}/src/system/PreKeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/src/system/KeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/src/system/CameraModel.cpp
    ${PROJECT_SOURCE_DIR}/src/system/StereoMatcher.cpp
    ${PROJECT_SOURCE_DIR}/src/system/StereoGeometryEstimator.cpp
    ${PROJECT_SOURCE_DIR}/src/system/FrameTracker.cpp
)

set(essietial_5pt_estimator_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver/relativeposeestimator_double.cpp
)

add_library(dso ${dso_SOURCE_FILES} ${dso_HEADER_FILES}
    ${essietial_5pt_estimator_SOURCE_FILES}
)
target_link_libraries(dso ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_subdirectory(samples)

enable_testing()
add_subdirectory(test)
