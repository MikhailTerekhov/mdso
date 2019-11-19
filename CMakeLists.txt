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

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/googletest)

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
    ${PROJECT_SOURCE_DIR}/include/util/PixelSelector.h
    ${PROJECT_SOURCE_DIR}/include/util/DistanceMap.h
    ${PROJECT_SOURCE_DIR}/include/util/PlyHolder.h
    ${PROJECT_SOURCE_DIR}/include/util/Sim3Aligner.h
    ${PROJECT_SOURCE_DIR}/include/util/flags.h
    ${PROJECT_SOURCE_DIR}/include/util/TrajectoryHolder.h

    ${PROJECT_SOURCE_DIR}/include/output/Observers.h
    ${PROJECT_SOURCE_DIR}/include/output/DsoObserver.h
    ${PROJECT_SOURCE_DIR}/include/output/DebugImageDrawer.h
    ${PROJECT_SOURCE_DIR}/include/output/TrajectoryWriter.h
    ${PROJECT_SOURCE_DIR}/include/output/TrajectoryWriterDso.h
    ${PROJECT_SOURCE_DIR}/include/output/TrajectoryWriterGT.h
    ${PROJECT_SOURCE_DIR}/include/output/TrajectoryWriterPredict.h
    ${PROJECT_SOURCE_DIR}/include/output/CloudWriter.h
    ${PROJECT_SOURCE_DIR}/include/output/CloudWriterGT.h
    ${PROJECT_SOURCE_DIR}/include/output/DelaunayInitializerObserver.h
    ${PROJECT_SOURCE_DIR}/include/output/InterpolationDrawer.h
    ${PROJECT_SOURCE_DIR}/include/output/FrameTrackerObserver.h
    ${PROJECT_SOURCE_DIR}/include/output/TrackingDebugImageDrawer.h
    ${PROJECT_SOURCE_DIR}/include/output/DepthPyramidDrawer.h

    ${PROJECT_SOURCE_DIR}/include/optimize/EnergyFunction.h
    ${PROJECT_SOURCE_DIR}/include/optimize/Parametrization.h
    ${PROJECT_SOURCE_DIR}/include/optimize/Residual.h

    ${PROJECT_SOURCE_DIR}/include/system/AffineLightTransform.h
    ${PROJECT_SOURCE_DIR}/include/system/SphericalPlus.h
    ${PROJECT_SOURCE_DIR}/include/system/DsoSystem.h
    ${PROJECT_SOURCE_DIR}/include/system/DsoInitializer.h
    ${PROJECT_SOURCE_DIR}/include/system/DelaunayDsoInitializer.h
    ${PROJECT_SOURCE_DIR}/include/system/PreKeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/KeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/MarginalizedKeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/MarginalizedPreKeyFrame.h
    ${PROJECT_SOURCE_DIR}/include/system/ImmaturePoint.h
    ${PROJECT_SOURCE_DIR}/include/system/OptimizedPoint.h
    ${PROJECT_SOURCE_DIR}/include/system/CameraModel.h
    ${PROJECT_SOURCE_DIR}/include/system/CameraBundle.h
    ${PROJECT_SOURCE_DIR}/include/system/StereoMatcher.h
    ${PROJECT_SOURCE_DIR}/include/system/StereoGeometryEstimator.h
    ${PROJECT_SOURCE_DIR}/include/system/FrameTracker.h
#    ${PROJECT_SOURCE_DIR}/include/system/BundleAdjuster.h
    ${PROJECT_SOURCE_DIR}/include/system/Preprocessor.h
    ${PROJECT_SOURCE_DIR}/include/system/IdentityPreprocessor.h
    ${PROJECT_SOURCE_DIR}/include/system/DoGPreprocessor.h
    ${PROJECT_SOURCE_DIR}/include/system/TrackingPredictor.h
    ${PROJECT_SOURCE_DIR}/include/system/TrackingPredictorRot.h
    ${PROJECT_SOURCE_DIR}/include/system/TrackingPredictorScrew.h
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
    ${PROJECT_SOURCE_DIR}/source/util/PlyHolder.cpp
    ${PROJECT_SOURCE_DIR}/source/util/Sim3Aligner.cpp
    ${PROJECT_SOURCE_DIR}/source/util/flags.cpp
    ${PROJECT_SOURCE_DIR}/source/util/TrajectoryHolder.cpp

    ${PROJECT_SOURCE_DIR}/source/output/DsoObserver.cpp
    ${PROJECT_SOURCE_DIR}/source/output/DebugImageDrawer.cpp
    ${PROJECT_SOURCE_DIR}/source/output/TrajectoryWriter.cpp
    ${PROJECT_SOURCE_DIR}/source/output/TrajectoryWriterDso.cpp
    ${PROJECT_SOURCE_DIR}/source/output/TrajectoryWriterGT.cpp
    ${PROJECT_SOURCE_DIR}/source/output/TrajectoryWriterPredict.cpp
    ${PROJECT_SOURCE_DIR}/source/output/CloudWriter.cpp
    ${PROJECT_SOURCE_DIR}/source/output/CloudWriterGT.cpp
    ${PROJECT_SOURCE_DIR}/source/output/DelaunayInitializerObserver.cpp
    ${PROJECT_SOURCE_DIR}/source/output/InterpolationDrawer.cpp
    ${PROJECT_SOURCE_DIR}/source/output/FrameTrackerObserver.cpp
    ${PROJECT_SOURCE_DIR}/source/output/TrackingDebugImageDrawer.cpp
    ${PROJECT_SOURCE_DIR}/source/output/DepthPyramidDrawer.cpp

    ${PROJECT_SOURCE_DIR}/source/optimize/EnergyFunction.cpp
    ${PROJECT_SOURCE_DIR}/source/optimize/Residual.cpp

    ${PROJECT_SOURCE_DIR}/source/system/SphericalPlus.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DsoSystem.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DsoInitializer.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DelaunayDsoInitializer.cpp
    ${PROJECT_SOURCE_DIR}/source/system/PreKeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/KeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/MarginalizedKeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/MarginalizedPreKeyFrame.cpp
    ${PROJECT_SOURCE_DIR}/source/system/ImmaturePoint.cpp
    ${PROJECT_SOURCE_DIR}/source/system/CameraModel.cpp
    ${PROJECT_SOURCE_DIR}/source/system/CameraBundle.cpp
    ${PROJECT_SOURCE_DIR}/source/system/StereoMatcher.cpp
    ${PROJECT_SOURCE_DIR}/source/system/StereoGeometryEstimator.cpp
    ${PROJECT_SOURCE_DIR}/source/system/FrameTracker.cpp
#    ${PROJECT_SOURCE_DIR}/source/system/BundleAdjuster.cpp
    ${PROJECT_SOURCE_DIR}/source/system/Preprocessor.cpp
    ${PROJECT_SOURCE_DIR}/source/system/IdentityPreprocessor.cpp
    ${PROJECT_SOURCE_DIR}/source/system/DoGPreprocessor.cpp
    ${PROJECT_SOURCE_DIR}/source/system/TrackingPredictor.cpp
    ${PROJECT_SOURCE_DIR}/source/system/TrackingPredictorRot.cpp
    ${PROJECT_SOURCE_DIR}/source/system/TrackingPredictorScrew.cpp
)

set(dso_internal_HEADER_FILES
        ${PROJECT_SOURCE_DIR}/internal/include/PreKeyFrameEntryInternals.h
)

set(dso_internal_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/internal/source/PreKeyFrameEntryInternals.cpp
)


set(essietial_5pt_estimator_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver/relativeposeestimator_double.cpp
)

set_source_files_properties(essietial_5pt_estimator_SOURCE_FILES
                           PROPERTIES COMPILE_FLAGS -O3
)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG -march=native")

add_library(dso ${dso_SOURCE_FILES} ${dso_HEADER_FILES}
    ${dso_internal_SOURCE_FILES} ${dso_internal_HEADER_FILES}
    ${essietial_5pt_estimator_SOURCE_FILES}
)

target_include_directories(dso
  PUBLIC
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  PRIVATE
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/internal/include
    ${PROJECT_SOURCE_DIR}/thirdparty/relative_pose_solver
)


target_link_libraries(dso
  PUBLIC
    stdc++fs
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
  PRIVATE
    ${CERES_LIBRARIES}
    ${TBB_LIBRARIES}
)

add_subdirectory(samples)

enable_testing()
add_subdirectory(test)
