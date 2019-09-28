set(TESTS test_stereo)

foreach(CUR_TEST ${TESTS})
    add_executable(${CUR_TEST} ${CUR_TEST}.cpp)
    target_link_libraries(${CUR_TEST} gtest dso)
    add_test(${CUR_TEST} ${CUR_TEST})
endforeach(CUR_TEST)
