#pragma once

// Minimal self-registering test harness. Each test executable defines cases
// with TEST(...) and links tests/support/test_main.cpp.

#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace linnet::test {

struct TestCase {
    const char* name;
    void (*run)();
};

std::vector<TestCase>& registry();
void report_failure(const char* file, int line, const std::string& message);

struct Registrar {
    // Runs during static initialization, where an exception could not be caught.
    Registrar(const char* name, void (*run)()) noexcept {
        try {
            registry().push_back({name, run});
        } catch (...) {
            std::abort();
        }
    }
};

template <typename T>
std::string describe(const T& value) {
    if constexpr (std::is_enum_v<T>) {
        return std::to_string(static_cast<long long>(value));
    } else if constexpr (std::is_same_v<T, char8_t> || std::is_same_v<T, char16_t> ||
                         std::is_same_v<T, char32_t>) {
        return "U+" + std::to_string(static_cast<unsigned long>(value));
    } else if constexpr (std::is_convertible_v<const T&, std::string_view>) {
        return '"' + std::string(std::string_view(value)) + '"';
    } else if constexpr (requires(std::ostream& out) { out << value; }) {
        std::ostringstream out;
        out << value;
        return out.str();
    } else {
        return "<unprintable>";
    }
}

template <typename A, typename B>
void check_equal(
    const A& actual, const B& expected, const char* expression, const char* file, int line) {
    if (!(actual == expected)) {
        report_failure(file,
                       line,
                       std::string(expression) + "\n    actual:   " + describe(actual) +
                           "\n    expected: " + describe(expected));
    }
}

} // namespace linnet::test

#define LINNET_TEST_CONCAT_INNER(a, b) a##b
#define LINNET_TEST_CONCAT(a, b) LINNET_TEST_CONCAT_INNER(a, b)

#define TEST(name)                                                                                 \
    static void LINNET_TEST_CONCAT(linnet_test_fn_, __LINE__)();                                   \
    static const ::linnet::test::Registrar LINNET_TEST_CONCAT(linnet_test_reg_, __LINE__){         \
        name, &LINNET_TEST_CONCAT(linnet_test_fn_, __LINE__)};                                     \
    static void LINNET_TEST_CONCAT(linnet_test_fn_, __LINE__)()

// Variadic so that conditions containing top-level commas (braced
// initializers) can be written without extra parentheses.
#define CHECK(...)                                                                                 \
    do {                                                                                           \
        if (!(__VA_ARGS__)) {                                                                      \
            ::linnet::test::report_failure(__FILE__, __LINE__, "CHECK(" #__VA_ARGS__ ")");         \
        }                                                                                          \
    } while (false)

#define CHECK_EQ(actual, expected)                                                                 \
    ::linnet::test::check_equal(                                                                   \
        (actual), (expected), "CHECK_EQ(" #actual ", " #expected ")", __FILE__, __LINE__)
