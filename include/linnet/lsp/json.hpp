#pragma once

#include <cstdint>
#include <expected>
#include <initializer_list>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace linnet::lsp {

// A JSON value with the shape JSON-RPC and LSP need. Numbers are kept as
// doubles; integers up to 2^53 round-trip exactly, which covers LSP.
class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;

    Json() = default; // null
    Json(std::nullptr_t) {}
    Json(bool value) : data_(value) {}
    Json(int value) : data_(static_cast<double>(value)) {}
    Json(std::int64_t value) : data_(static_cast<double>(value)) {}
    Json(std::uint32_t value) : data_(static_cast<double>(value)) {}
    Json(double value) : data_(value) {}
    Json(const char* value) : data_(std::string(value)) {}
    Json(std::string value) : data_(std::move(value)) {}
    Json(std::string_view value) : data_(std::string(value)) {}
    Json(Array value) : data_(std::move(value)) {}
    Json(Object value) : data_(std::move(value)) {}

    static Json array(std::initializer_list<Json> items) { return Array(items); }
    static Json object(std::initializer_list<std::pair<const std::string, Json>> items) {
        return Object(items);
    }

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data_); }
    bool is_string() const { return std::holds_alternative<std::string>(data_); }
    bool is_number() const { return std::holds_alternative<double>(data_); }
    bool is_object() const { return std::holds_alternative<Object>(data_); }
    bool is_array() const { return std::holds_alternative<Array>(data_); }

    // Accessors return a default when the value has another type, so that
    // malformed client messages degrade instead of crashing.
    const std::string& as_string() const;
    double as_number() const;
    std::int64_t as_int() const { return static_cast<std::int64_t>(as_number()); }
    bool as_bool() const;
    const Array& as_array() const;
    const Object& as_object() const;

    // Member of an object; null for anything else or when absent.
    const Json& operator[](std::string_view key) const;
    Json& set(std::string key, Json value);

    std::string dump() const;

private:
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data_;
};

std::expected<Json, std::string> parse_json(std::string_view text);

} // namespace linnet::lsp
