cmake_minimum_required(VERSION 3.0.2)

project(dso-fisheye-ceres)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

if (ASAN_BUILD)
    add_compile_options("-fsanitize=address")
    link_libraries("-fsanitize=address")
endif()

find_package(TBB REQUIRED)
find_package(gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/googletest)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver
)

set(dso_HEADER_FILES
    ${PROJECT_SOURCE_DIR}/include/util/util.h
    ${PROJECT_SOURCE_DIR}/include/util/types.h
    ${PROJECT_SOURCE_DIR}/include/util/defs.h
    ${PROJECT_SOURCE_DIR}/include/util/settings.h
    ${PROJECT_SOURCE_DIR}/include/util/geometry.h
    ${PROJECT_SOURCE_DIR}/include/util/Triangulation.h
    ${PROJECT_SOURCE_DIR}/include/util/Terrain.h
    ${PROJECT_SOURCE_DIR}/include/util/SphericalTriangulation.h
    ${PROJECT_SOURCE_DIR}/include/util/SphericalTerrain.h
    ${PROJECT_SOURCE_DIR}/include/util/ImagePyramid.h
    ${PROJECT_SOURCE_DIR}/include/util/DepthedImagePyramid.h
    ${PROJECT_SOURCE_DIR}/include/util/ObjectPool.h
    ${PROJECT_SOURCE_DIR}/include/util/PixelSelector.h
    ${PROJECT_SOURCE_DIR}/include/util/DistanceMap.h
    ${PROJECT_SOURCE_DIR}/include/system/AffineLightTransform.h
    ${PROJECT_SOURCE_DIR}/include/system/SphericalPlus.h
    ${PROJECT_SOURCE_DIR}/include/system/DsoSystem.h
    ${PROJECT_SOURCE_DIR}/include/system/DsoInitializer.h
    ${PROJECT_SOURCE_DIR}/include/system/DelaunayDsoInitializer.h
    ${PROJECT_SOURCE_DIR}/include/system/PreKeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/KeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/ImmaturePoint.h
    ${PROJECT_SOURCE_DIR}/include/system/OptimizedPoint.h
    ${PROJECT_SOURCE_DIR}/include/system/CameraModel.h
    ${PROJECT_SOURCE_DIR}/include/system/StereoMatcher.h
    ${PROJECT_SOURCE_DIR}/include/system/StereoGeometryEstimator.h
    ${PROJECT_SOURCE_DIR}/include/system/FrameTracker.h
    ${PROJECT_SOURCE_DIR}/include/system/BundleAdjuster.h
)

set(dso_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/source/util/util.cpp
    ${PROJECT_SOURCE_DIR}/source/util/settings.cpp
    ${PROJECT_SOURCE_DIR}/source/util/geometry.cpp
    ${PROJECT_SOURCE_DIR}/source/util/Triangulation.cpp
    ${PROJECT_SOURCE_DIR}/source/util/Terrain.cpp
    ${PROJECT_SOURCE_DIR}/source/util/SphericalTriangulation.cpp
    ${PROJECT_SOURCE_DIR}/source/util/SphericalTerrain.cpp
    ${PROJECT_SOURCE_DIR}/source/util/ImagePyramid.cpp
    ${PROJECT_SOURCE_DIR}/source/util/DepthedImagePyramid.cpp
    ${PROJECT_SOURCE_DIR}/source/util/PixelSelector.cpp
    ${PROJECT_SOURCE_DIR}/source/util/DistanceMap.cpp
    ${PROJECT_SOURCE_DIR}/source/system/SphericalPlus.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DsoSystem.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DelaunayDsoInitializer.cpp
    ${PROJECT_SOURCE_DIR}/source/system/PreKeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/KeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/ImmaturePoint.cpp
    ${PROJECT_SOURCE_DIR}/source/system/CameraModel.cpp
    ${PROJECT_SOURCE_DIR}/source/system/StereoMatcher.cpp
    ${PROJECT_SOURCE_DIR}/source/system/StereoGeometryEstimator.cpp
    ${PROJECT_SOURCE_DIR}/source/system/FrameTracker.cpp
    ${PROJECT_SOURCE_DIR}/source/system/BundleAdjuster.cpp
)

set(essietial_5pt_estimator_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver/relativeposeestimator_double.cpp
)

set_source_files_properties(essietial_5pt_estimator_SOURCE_FILES
                           PROPERTIES COMPILE_FLAGS -O3
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")

add_library(dso ${dso_SOURCE_FILES} ${dso_HEADER_FILES}
    ${essietial_5pt_estimator_SOURCE_FILES}
)

target_link_libraries(dso ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIBRARIES} ${TBB_LIBRARIES})

add_subdirectory(samples)

enable_testing()
add_subdirectory(test)
