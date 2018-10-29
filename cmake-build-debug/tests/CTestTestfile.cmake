# CMake generated Testfile for 
# Source directory: /work/juan/proj_others/V4PCS/tests
# Build directory: /work/juan/proj_others/V4PCS/cmake-build-debug/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(externalAppTest "Super4PCS-externalAppTest")
add_test(pair_extraction "/work/juan/proj_others/V4PCS/cmake-build-debug/tests/pair_extraction")
add_test(matching_0 "/work/juan/proj_others/V4PCS/cmake-build-debug/tests/matching" "0")
set_tests_properties(matching_0 PROPERTIES  TIMEOUT "600")
add_test(matching_1 "/work/juan/proj_others/V4PCS/cmake-build-debug/tests/matching" "1")
set_tests_properties(matching_1 PROPERTIES  TIMEOUT "600")
