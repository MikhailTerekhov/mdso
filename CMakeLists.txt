cmake_minimum_required(VERSION 2.8)

project(dso-fisheye-ceres)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/googletest)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver
)

set(dso_HEADER_FILES
    ${PROJECT_SOURCE_DIR}/include/util/util.h
    ${PROJECT_SOURCE_DIR}/include/util/types.h
    ${PROJECT_SOURCE_DIR}/include/util/defs.h
    ${PROJECT_SOURCE_DIR}/include/util/settings.h
    ${PROJECT_SOURCE_DIR}/include/util/Triangulation.h
    ${PROJECT_SOURCE_DIR}/include/util/Terrain.h
    ${PROJECT_SOURCE_DIR}/include/util/SphericalTriangulation.h
    ${PROJECT_SOURCE_DIR}/include/util/SphericalTerrain.h
    ${PROJECT_SOURCE_DIR}/include/system/AffineLightTransform.h
    ${PROJECT_SOURCE_DIR}/include/system/DsoSystem.h
    ${PROJECT_SOURCE_DIR}/include/system/DsoInitializer.h
    ${PROJECT_SOURCE_DIR}/include/system/PreKeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/KeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/InterestPoint.h
    ${PROJECT_SOURCE_DIR}/include/system/CameraModel.h
    ${PROJECT_SOURCE_DIR}/include/system/StereoMatcher.h
    ${PROJECT_SOURCE_DIR}/include/system/StereoGeometryEstimator.h
    ${PROJECT_SOURCE_DIR}/include/system/FrameTracker.h
)

set(dso_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/source/util/util.cpp
    ${PROJECT_SOURCE_DIR}/source/util/settings.cpp
    ${PROJECT_SOURCE_DIR}/source/util/Triangulation.cpp
    ${PROJECT_SOURCE_DIR}/source/util/Terrain.cpp
    ${PROJECT_SOURCE_DIR}/source/util/SphericalTriangulation.cpp
    ${PROJECT_SOURCE_DIR}/source/util/SphericalTerrain.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DsoSystem.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DsoInitializer.cpp
    ${PROJECT_SOURCE_DIR}/source/system/PreKeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/KeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/CameraModel.cpp
    ${PROJECT_SOURCE_DIR}/source/system/StereoMatcher.cpp
    ${PROJECT_SOURCE_DIR}/source/system/StereoGeometryEstimator.cpp
    ${PROJECT_SOURCE_DIR}/source/system/FrameTracker.cpp
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
