set(robotcar_reader_HEADER_FILES
  ${PROJECT_SOURCE_DIR}/samples/robotcar/reader/RobotcarReader.h)
set(robotcar_reader_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/robotcar/reader/RobotcarReader.cpp)
add_library(
  robotcar_reader 
    ${robotcar_reader_HEADER_FILES} 
    ${robotcar_reader_SOURCE_FILES})
target_link_libraries(robotcar_reader dso)
