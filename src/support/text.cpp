#include "linnet/support/text.hpp"

#include <cstdlib>
#include <sstream>

namespace linnet {

std::string shortest_float(double value) {
    std::string text;
    for (int precision = 6; precision <= 17; ++precision) {
        std::ostringstream stream;
        stream.precision(precision);
        stream << value;
        text = stream.str();
        if (precision == 17 || std::strtod(text.c_str(), nullptr) == value) {
            break;
        }
    }
    if (text.find_first_of(".eEni") == std::string::npos) {
        text += ".0";
    }
    return text;
}

std::string decimal_float(double value) {
    std::string text = shortest_float(value);
    const std::size_t exponent = text.find_first_of("eE");
    if (text.find('.') == std::string::npos && exponent != std::string::npos) {
        text.insert(exponent, ".0");
    }
    return text;
}

std::string indent_lines(std::string_view text, std::string_view prefix) {
    std::string out;
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = text.find('\n', start);
        end = end == std::string_view::npos ? text.size() : end + 1;
        const std::string_view line = text.substr(start, end - start);
        if (line != "\n") {
            out += prefix;
        }
        out += line;
        start = end;
    }
    return out;
}

std::string command_line_word(std::string_view argument) {
    if (!argument.empty() && argument.find_first_of(" \t\n\v\"") == std::string_view::npos) {
        return std::string(argument);
    }
    std::string out = "\"";
    std::size_t backslashes = 0;
    for (const char c : argument) {
        if (c == '\\') {
            ++backslashes;
            continue;
        }
        out.append(c == '"' ? (2 * backslashes) + 1 : backslashes, '\\');
        out += c;
        backslashes = 0;
    }
    out.append(2 * backslashes, '\\');
    out += '"';
    return out;
}

} // namespace linnet
