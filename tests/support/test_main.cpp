#include "test.hpp"

#include <cstdio>
#include <exception>

namespace linnet::test {

namespace {
int current_failures = 0;
}

std::vector<TestCase>& registry() {
    static std::vector<TestCase> cases;
    return cases;
}

void report_failure(const char* file, int line, const std::string& message) {
    ++current_failures;
    std::fprintf(stderr, "%s:%d: %s\n", file, line, message.c_str());
}

} // namespace linnet::test

int main() {
    using namespace linnet::test;

    int failed_cases = 0;
    for (const TestCase& test : registry()) {
        current_failures = 0;
        try {
            test.run();
        } catch (const std::exception& error) {
            report_failure(test.name, 0, std::string("unexpected exception: ") + error.what());
        }
        if (current_failures != 0) {
            ++failed_cases;
            std::fprintf(stderr, "FAILED: %s\n", test.name);
        }
    }

    std::printf("%zu test cases, %d failed\n", registry().size(), failed_cases);
    return failed_cases == 0 ? 0 : 1;
}
