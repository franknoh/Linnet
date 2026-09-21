add_library(linnet_test_support STATIC support/test_main.cpp)
target_include_directories(linnet_test_support PUBLIC support)
target_link_libraries(linnet_test_support PRIVATE linnet_build_options)

# linnet_add_test(<name> <sources...>) builds one unit-test executable and
# registers it with CTest.
function(linnet_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE linnet::core linnet_test_support linnet_build_options)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_test(NAME cli_version COMMAND linnet --version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^linnet [0-9]+\\.[0-9]+\\.[0-9]+")

add_test(NAME cli_unknown_command COMMAND linnet no-such-command)
set_tests_properties(cli_unknown_command PROPERTIES WILL_FAIL TRUE)

linnet_add_test(source_manager_test unit/source_manager_test.cpp)
linnet_add_test(diagnostic_test unit/diagnostic_test.cpp)
