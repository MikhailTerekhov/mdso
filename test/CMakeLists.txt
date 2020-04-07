set(TESTS test_cameramodel test_stereo test_triangulation test_geometry test_util test_serialization)

foreach(CUR_TEST ${TESTS})
    add_executable(${CUR_TEST} ${CUR_TEST}.cpp)
    target_link_libraries(${CUR_TEST} gtest dso)
endforeach(CUR_TEST)

target_link_libraries(test_serialization reader)

foreach(CUR_TEST ${TESTS})
    add_test(${CUR_TEST} ${CUR_TEST})
endforeach(CUR_TEST)
