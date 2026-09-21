if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "LINNET_BUILD_FUZZERS requires Clang (libFuzzer)")
endif()

add_executable(fuzz_frontend fuzz_frontend.cpp)
target_link_libraries(fuzz_frontend PRIVATE linnet::core linnet_build_options)
target_link_options(fuzz_frontend PRIVATE -fsanitize=fuzzer)
