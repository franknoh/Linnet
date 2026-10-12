cmake_minimum_required(VERSION 3.25)

project(linnet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LINNET_WARNINGS_AS_ERRORS "Treat compiler warnings in project code as errors" OFF)
option(LINNET_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LINNET_BUILD_FUZZERS "Build libFuzzer targets (requires Clang)" OFF)
option(LINNET_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})

# Interface target carrying the warning and sanitizer policy for project code.
add_library(linnet_build_options INTERFACE)
if(MSVC)
    target_compile_options(linnet_build_options INTERFACE /W4 /permissive- /utf-8 /Zc:preprocessor)
    # Standard functions such as std::getenv are not deprecated in C++.
    target_compile_definitions(linnet_build_options INTERFACE _CRT_SECURE_NO_WARNINGS)
    # The 8 MiB stack Linux and macOS give a main thread, not Windows' 1 MiB:
    # the parser's nesting limit is sized for it, and a Debug build's frames
    # at that limit overflow 1 MiB.
    target_link_options(linnet_build_options INTERFACE /STACK:8388608)
    if(LINNET_WARNINGS_AS_ERRORS)
        target_compile_options(linnet_build_options INTERFACE /WX)
    endif()
else()
    target_compile_options(linnet_build_options INTERFACE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
        -Wnon-virtual-dtor -Wold-style-cast -Wimplicit-fallthrough)
    if(LINNET_WARNINGS_AS_ERRORS)
        target_compile_options(linnet_build_options INTERFACE -Werror)
    endif()
    if(LINNET_BUILD_FUZZERS)
        # Instruments all project code; only fuzz targets link the fuzzer runtime.
        target_compile_options(linnet_build_options INTERFACE
            -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer)
        target_link_options(linnet_build_options INTERFACE -fsanitize=address,undefined)
    endif()
    if(LINNET_SANITIZE)
        target_compile_options(linnet_build_options INTERFACE
            -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
        target_link_options(linnet_build_options INTERFACE -fsanitize=address,undefined)
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/linnet/version.hpp
    @ONLY)

# The prelude's dtype declarations, compiled into the checker; editing the
# file configures the build again.
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/stdlib/prelude.linnet LINNET_PRELUDE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/stdlib/prelude.linnet)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prelude.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/linnet/prelude.hpp
    @ONLY)

add_library(linnet_core STATIC
    src/ast/ast.cpp
    src/ast/dump.cpp
    src/backend/graph_export.cpp
    src/backend/pallas.cpp
    src/backend/gradient.cpp
    src/backend/tile_target.cpp
    src/backend/triton.cpp
    src/backend/onnx.cpp
    src/backend/plan.cpp
    src/backend/plan_reader.cpp
    src/backend/python_target.cpp
    src/backend/recorded_graph.cpp
    src/backend/stablehlo.cpp
    src/backend/torch_source.cpp
    src/backend/jax_source.cpp
    src/diagnostic/diagnostic.cpp
    src/diagnostic/json.cpp
    src/diagnostic/render.cpp
    src/emit/source.cpp
    src/format/doc.cpp
    src/format/formatter.cpp
    src/ir/ir.cpp
    src/ir/lower.cpp
    src/lsp/json.cpp
    src/lsp/server.cpp
    src/module/loader.cpp
    src/opt/candidates.cpp
    src/opt/egraph.cpp
    src/opt/passes.cpp
    src/package/fetch.cpp
    src/package/manifest.cpp
    src/package/spec_manifest.cpp
    src/package/toml.cpp
    src/sema/checker_calls.cpp
    src/sema/checker_decls.cpp
    src/sema/checker_exprs.cpp
    src/sema/checker_kernels.cpp
    src/sema/manifest.cpp
    src/sema/prelude.cpp
    src/sema/symbols.cpp
    src/sema/types.cpp
    src/shape/dim.cpp
    src/shape/solver.cpp
    src/source/source_manager.cpp
    src/source/utf8.cpp
    src/support/files.cpp
    src/support/text.cpp
    src/syntax/lexer.cpp
    src/syntax/parser.cpp)
add_library(linnet::core ALIAS linnet_core)
target_include_directories(linnet_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(linnet_core PRIVATE linnet_build_options)

# The runtime that runs execution plans: a C library with no CUDA
# dependency at build time (src/runtime/driver.cpp loads the driver).
add_library(linnet_runtime SHARED
    src/runtime/runtime.cpp src/runtime/driver.cpp src/runtime/cublaslt.cpp)
target_include_directories(linnet_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(linnet_runtime PRIVATE linnet_build_options ${CMAKE_DL_LIBS})
target_compile_definitions(linnet_runtime PRIVATE LINNET_RUNTIME_BUILD)
set_target_properties(linnet_runtime PROPERTIES
    C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

add_executable(linnet src/cli/main.cpp)
target_link_libraries(linnet PRIVATE linnet::core linnet_build_options)

include(GNUInstallDirs)
install(TARGETS linnet RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# Beside the standard library, where the Python package's wheel carries
# share/linnet and `linnet.runtime` looks for it.
install(TARGETS linnet_runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_DATADIR}/linnet/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/linnet/lib)
# The executable looks for the standard library at share/linnet/stdlib
# relative to its own location.
install(DIRECTORY stdlib/ DESTINATION ${CMAKE_INSTALL_DATADIR}/linnet/stdlib
        FILES_MATCHING PATTERN "*.linnet")
install(DIRECTORY editors/vim/ DESTINATION ${CMAKE_INSTALL_DATADIR}/linnet/vim)
install(FILES LICENSE README.md LANGUAGE.md DESTINATION ${CMAKE_INSTALL_DATADIR}/linnet)

if(LINNET_BUILD_FUZZERS)
    add_subdirectory(fuzz)
endif()

if(LINNET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
