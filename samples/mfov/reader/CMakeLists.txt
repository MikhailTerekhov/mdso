set(mfov_reader_HEADER_FILES
  ${PROJECT_SOURCE_DIR}/samples/mfov/reader/MultiFovReader.h)
set(mfov_reader_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/mfov/reader/MultiFovReader.cpp)
add_library(mfov_reader ${mfov_reader_HEADER_FILES} ${mfov_reader_SOURCE_FILES})
target_link_libraries(mfov_reader dso)
