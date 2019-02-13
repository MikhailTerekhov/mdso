set(reader_HEADER_FILES
  ${PROJECT_SOURCE_DIR}/samples/mfov/reader/MultiFovReader.h)
set(reader_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/samples/mfov/reader/MultiFovReader.cpp)
add_library(reader ${reader_HEADER_FILES} ${reader_SOURCE_FILES})
target_link_libraries(reader dso)
