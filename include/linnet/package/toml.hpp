#pragma once

#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace linnet::toml {

// A TOML document, as much of the format as the project's own files need:
// tables `[a.b]`, arrays of tables `[[a]]`, dotted and quoted keys, basic and
// literal strings, integers, booleans, arrays, and inline tables. Dates,
// floats, and multi-line strings are rejected with a message rather than
// misread. Reading a document never executes anything.

struct Value;
using Table = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    std::variant<std::string, std::int64_t, bool, Array, Table> data;
    std::uint32_t line = 0; // 1-based line of the value, for messages

    const std::string* as_string() const { return std::get_if<std::string>(&data); }
    const std::int64_t* as_integer() const { return std::get_if<std::int64_t>(&data); }
    const bool* as_bool() const { return std::get_if<bool>(&data); }
    const Array* as_array() const { return std::get_if<Array>(&data); }
    const Table* as_table() const { return std::get_if<Table>(&data); }

    // Nested lookup; null when any key is missing or not a table on the way.
    const Value* find(std::string_view key) const;
};

struct ParseError {
    std::uint32_t line;
    std::string message;
};

std::expected<Table, ParseError> parse(std::string_view text);

} // namespace linnet::toml
