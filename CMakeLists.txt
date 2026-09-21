cmake_minimum_required(VERSION 3.25)

project(linnet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LINNET_WARNINGS_AS_ERRORS "Treat compiler warnings in project code as errors" OFF)
option(LINNET_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LINNET_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})

# Interface target carrying the warning and sanitizer policy for project code.
add_library(linnet_build_options INTERFACE)
if(MSVC)
    target_compile_options(linnet_build_options INTERFACE /W4 /permissive- /utf-8)
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

add_library(linnet_core STATIC
    src/ast/ast.cpp
    src/ast/dump.cpp
    src/diagnostic/diagnostic.cpp
    src/diagnostic/render.cpp
    src/format/doc.cpp
    src/format/formatter.cpp
    src/source/source_manager.cpp
    src/source/utf8.cpp
    src/syntax/lexer.cpp
    src/syntax/parser.cpp)
add_library(linnet::core ALIAS linnet_core)
target_include_directories(linnet_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(linnet_core PRIVATE linnet_build_options)

add_executable(linnet src/cli/main.cpp)
target_link_libraries(linnet PRIVATE linnet::core linnet_build_options)

if(LINNET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
