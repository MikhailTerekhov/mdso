set(TESTS test_motionderivatives test_residual test_parametrizations)

foreach(CUR_TEST ${TESTS})
    add_executable(${CUR_TEST} ${CUR_TEST}.cpp)
    target_link_libraries(${CUR_TEST} gtest dso)
endforeach(CUR_TEST)

foreach(CUR_TEST ${TESTS})
    add_test(${CUR_TEST} ${CUR_TEST})
endforeach(CUR_TEST)

target_link_libraries(test_residual mfov_reader)
