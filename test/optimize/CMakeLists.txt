set(TESTS test_motionderivatives test_residual test_parametrizations test_energyfunction)

foreach(CUR_TEST ${TESTS})
  add_executable(${CUR_TEST} ${CUR_TEST}.cpp)
  target_link_libraries(${CUR_TEST} gtest dso)
endforeach(CUR_TEST)

foreach(CUR_TEST ${TESTS})
  add_test(${CUR_TEST} ${CUR_TEST})
endforeach(CUR_TEST)

target_include_directories(test_residual
  PUBLIC
    ${PROJECT_SOURCE_DIR}/samples/mfov/reader
    ${PROJECT_SOURCE_DIR}/internal/include
)
target_include_directories(test_energyfunction
  PUBLIC
    ${PROJECT_SOURCE_DIR}/samples/mfov/reader
    ${PROJECT_SOURCE_DIR}/internal/include
)

target_link_libraries(test_residual mfov_reader)
target_link_libraries(test_energyfunction mfov_reader)
