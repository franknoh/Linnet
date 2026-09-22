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
linnet_add_test(doc_test unit/doc_test.cpp)
linnet_add_test(lexer_test unit/lexer_test.cpp)
linnet_add_test(parser_test unit/parser_test.cpp)

linnet_add_test(spec_syntax_test spec/spec_syntax_test.cpp)
target_compile_definitions(spec_syntax_test PRIVATE
    LINNET_SPEC_TESTS_DIR="${PROJECT_SOURCE_DIR}/spec-tests")
linnet_add_test(formatter_test unit/formatter_test.cpp)

add_test(NAME cli_fmt_check_clean
    COMMAND linnet fmt --check ${PROJECT_SOURCE_DIR}/spec-tests/valid ${PROJECT_SOURCE_DIR}/spec-tests/fixtures)
add_test(NAME cli_fmt_rejects_syntax_errors
    COMMAND linnet fmt --check ${PROJECT_SOURCE_DIR}/spec-tests/invalid/112_reserved_identifier.linnet)
set_tests_properties(cli_fmt_rejects_syntax_errors PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_inspect_ast
    COMMAND linnet inspect --ast ${PROJECT_SOURCE_DIR}/spec-tests/valid/001_minimal.linnet)
set_tests_properties(cli_inspect_ast PROPERTIES PASS_REGULAR_EXPRESSION "pub fn add_one")
linnet_add_test(shape_test unit/shape_test.cpp)
linnet_add_test(sema_test unit/sema_test.cpp)

linnet_add_test(spec_semantic_test spec/spec_semantic_test.cpp)
target_compile_definitions(spec_semantic_test PRIVATE
    LINNET_SPEC_TESTS_DIR="${PROJECT_SOURCE_DIR}/spec-tests")

add_test(NAME cli_check_examples COMMAND linnet check ${PROJECT_SOURCE_DIR}/examples)

linnet_add_test(loader_test unit/loader_test.cpp)
target_compile_definitions(loader_test PRIVATE
    LINNET_SPEC_TESTS_DIR="${PROJECT_SOURCE_DIR}/spec-tests")

add_test(NAME cli_check_package
    COMMAND linnet check --std ${PROJECT_SOURCE_DIR}/spec-tests/fixtures/std
            ${PROJECT_SOURCE_DIR}/spec-tests/fixtures/pkg)

linnet_add_test(json_test unit/json_test.cpp)

add_test(NAME cli_check_warning_passes
    COMMAND linnet check ${CMAKE_CURRENT_SOURCE_DIR}/cli/unused_local.linnet)
add_test(NAME cli_check_strict_fails_on_warning
    COMMAND linnet check --strict ${CMAKE_CURRENT_SOURCE_DIR}/cli/unused_local.linnet)
set_tests_properties(cli_check_strict_fails_on_warning PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_check_json
    COMMAND linnet check --json ${CMAKE_CURRENT_SOURCE_DIR}/cli/unused_local.linnet)
set_tests_properties(cli_check_json PROPERTIES
    PASS_REGULAR_EXPRESSION "\"code\":\"W1002\".*\"summary\":{\"errors\":0,\"warnings\":1}")

add_test(NAME cli_inspect_parameters
    COMMAND linnet inspect --parameters ${PROJECT_SOURCE_DIR}/examples/05-block-and-weights/model.linnet)
set_tests_properties(cli_inspect_parameters PROPERTIES
    PASS_REGULAR_EXPRESSION "param layers\\[\\*\\].up.weight: Tensor\\[Inner, H; T\\] x Layers")

linnet_add_test(toml_test unit/toml_test.cpp)

add_test(NAME cli_spec_test
    COMMAND linnet spec-test --std ${PROJECT_SOURCE_DIR}/spec-tests/fixtures/std ${PROJECT_SOURCE_DIR}/spec-tests)
add_test(NAME cli_lint_fails_on_warning
    COMMAND linnet lint ${CMAKE_CURRENT_SOURCE_DIR}/cli/unused_local.linnet)
set_tests_properties(cli_lint_fails_on_warning PROPERTIES WILL_FAIL TRUE)

linnet_add_test(manifest_test unit/manifest_test.cpp)

if(NOT WIN32)
    add_test(NAME cli_init_and_check
        COMMAND ${CMAKE_COMMAND} -E env LINNET_BIN=$<TARGET_FILE:linnet>
                bash ${CMAKE_CURRENT_SOURCE_DIR}/cli/init_and_check.sh
                ${CMAKE_CURRENT_BINARY_DIR}/init_scratch)
endif()
linnet_add_test(lsp_test unit/lsp_test.cpp)

add_test(NAME stdlib_checks COMMAND linnet lint --std ${PROJECT_SOURCE_DIR}/stdlib ${PROJECT_SOURCE_DIR}/stdlib)
add_test(NAME stdlib_is_formatted COMMAND linnet fmt --check ${PROJECT_SOURCE_DIR}/stdlib)
add_test(NAME example_transformer_checks
    COMMAND linnet lint --std ${PROJECT_SOURCE_DIR}/stdlib ${PROJECT_SOURCE_DIR}/examples/09-tiny-transformer)
linnet_add_test(ir_test unit/ir_test.cpp)
add_test(NAME cli_core_ir_stdlib
    COMMAND linnet inspect --core-ir --std ${PROJECT_SOURCE_DIR}/stdlib ${PROJECT_SOURCE_DIR}/stdlib/nn/attention.linnet)
add_test(NAME cli_core_ir_transformer
    COMMAND linnet inspect --core-ir --std ${PROJECT_SOURCE_DIR}/stdlib ${PROJECT_SOURCE_DIR}/examples/09-tiny-transformer/src/lib.linnet)
add_test(NAME cli_plan_transformer
    COMMAND linnet plan --std ${PROJECT_SOURCE_DIR}/stdlib ${PROJECT_SOURCE_DIR}/examples/09-tiny-transformer/src/lib.linnet)
set_tests_properties(cli_plan_transformer PROPERTIES PASS_REGULAR_EXPRESSION "\"version\":1,\"root\":\\{\"name\":\"Model\"")
