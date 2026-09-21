#include "linnet/version.hpp"

#include <cstdio>
#include <string_view>

namespace {

// Exit statuses are part of the CLI contract and must stay stable.
constexpr int exit_success = 0;
constexpr int exit_usage = 2;

void print_usage(std::FILE* out) {
    std::fputs("Usage: linnet <command> [options]\n"
               "\n"
               "Options:\n"
               "  -h, --help       Show this help\n"
               "  -V, --version    Show the toolchain version\n",
               out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(stderr);
        return exit_usage;
    }

    const std::string_view command = argv[1];
    if (command == "-h" || command == "--help" || command == "help") {
        print_usage(stdout);
        return exit_success;
    }
    if (command == "-V" || command == "--version") {
        std::printf("linnet %s\n", linnet::version_string);
        return exit_success;
    }

    std::fprintf(stderr, "linnet: unknown command '%s'\n\n", argv[1]);
    print_usage(stderr);
    return exit_usage;
}
