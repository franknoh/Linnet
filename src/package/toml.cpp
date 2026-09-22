#include "linnet/package/toml.hpp"

#include <limits>
#include <optional>
#include <set>
#include <utility>

namespace linnet::toml {

const Value* Value::find(std::string_view key) const {
    const Table* table = as_table();
    if (table == nullptr) {
        return nullptr;
    }
    const auto found = table->find(std::string(key));
    return found == table->end() ? nullptr : &found->second;
}

namespace {

constexpr bool is_bare_key_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
           c == '-';
}

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    std::expected<Table, ParseError> run() {
        Table root;
        Table* current = &root;
        while (skip_blank_lines(), pos_ < text_.size()) {
            if (peek() == '[') {
                auto table = header(root);
                if (!table) {
                    return std::unexpected(table.error());
                }
                current = *table;
            } else {
                if (auto error = key_value(*current)) {
                    return std::unexpected(*error);
                }
            }
            if (auto error = end_of_line()) {
                return std::unexpected(*error);
            }
        }
        return root;
    }

private:
    using Error = std::optional<ParseError>;

    char peek(std::size_t ahead = 0) const {
        return pos_ + ahead < text_.size() ? text_[pos_ + ahead] : '\0';
    }

    ParseError fail(std::string message) const { return {line_, std::move(message)}; }

    void skip_spaces() {
        while (peek() == ' ' || peek() == '\t') {
            ++pos_;
        }
    }

    void skip_comment() {
        if (peek() == '#') {
            while (pos_ < text_.size() && peek() != '\n') {
                ++pos_;
            }
        }
    }

    bool newline() {
        if (peek() == '\r' && peek(1) == '\n') {
            pos_ += 2;
        } else if (peek() == '\n') {
            ++pos_;
        } else {
            return false;
        }
        ++line_;
        return true;
    }

    void skip_blank_lines() {
        while (true) {
            skip_spaces();
            skip_comment();
            if (!newline()) {
                return;
            }
        }
    }

    Error end_of_line() {
        skip_spaces();
        skip_comment();
        if (pos_ < text_.size() && !newline()) {
            return fail("expected end of line");
        }
        return std::nullopt;
    }

    // ------------------------------------------------------------------- keys

    std::expected<std::string, ParseError> simple_key() {
        if (peek() == '"' || peek() == '\'') {
            auto text = string_value();
            if (!text) {
                return std::unexpected(text.error());
            }
            return std::get<std::string>(text->data);
        }
        const std::size_t begin = pos_;
        while (is_bare_key_char(peek())) {
            ++pos_;
        }
        if (pos_ == begin) {
            return std::unexpected(fail("expected a key"));
        }
        return std::string(text_.substr(begin, pos_ - begin));
    }

    std::expected<std::vector<std::string>, ParseError> dotted_key() {
        std::vector<std::string> parts;
        while (true) {
            skip_spaces();
            auto part = simple_key();
            if (!part) {
                return std::unexpected(part.error());
            }
            parts.push_back(std::move(*part));
            skip_spaces();
            if (peek() != '.') {
                return parts;
            }
            ++pos_;
        }
    }

    // Walks (creating when needed) the tables named by `parts`.
    std::expected<Table*, ParseError> descend(Table& root, const std::vector<std::string>& parts) {
        Table* table = &root;
        for (const std::string& part : parts) {
            auto [entry, inserted] = table->try_emplace(part, Value{Table{}, line_});
            if (Array* array = std::get_if<Array>(&entry->second.data);
                array != nullptr && !array->empty() && array->back().as_table() != nullptr) {
                table = &std::get<Table>(array->back().data); // latest [[entry]]
            } else if (Table* nested = std::get_if<Table>(&entry->second.data)) {
                table = nested;
            } else {
                return std::unexpected(fail("`" + part + "` is not a table"));
            }
        }
        return table;
    }

    std::expected<Table*, ParseError> header(Table& root) {
        const bool is_array = peek(1) == '[';
        pos_ += is_array ? 2 : 1;
        auto parts = dotted_key();
        if (!parts) {
            return std::unexpected(parts.error());
        }
        skip_spaces();
        if (peek() != ']' || (is_array && peek(1) != ']')) {
            return std::unexpected(fail("expected `]` after the table name"));
        }
        pos_ += is_array ? 2 : 1;

        const std::string last = parts->back();
        parts->pop_back();
        auto parent = descend(root, *parts);
        if (!parent) {
            return std::unexpected(parent.error());
        }
        if (is_array) {
            auto [entry, inserted] = (*parent)->try_emplace(last, Value{Array{}, line_});
            Array* array = std::get_if<Array>(&entry->second.data);
            if (array == nullptr) {
                return std::unexpected(fail("`" + last + "` is not an array of tables"));
            }
            array->push_back(Value{Table{}, line_});
            return &std::get<Table>(array->back().data);
        }
        auto [entry, inserted] = (*parent)->try_emplace(last, Value{Table{}, line_});
        Table* table = std::get_if<Table>(&entry->second.data);
        if (table == nullptr) {
            return std::unexpected(fail("`" + last + "` is not a table"));
        }
        if (!inserted && defined_tables_.contains(table)) {
            return std::unexpected(fail("table `" + last + "` is defined twice"));
        }
        defined_tables_.insert(table);
        return table;
    }

    Error key_value(Table& table) {
        auto parts = dotted_key();
        if (!parts) {
            return parts.error();
        }
        skip_spaces();
        if (peek() != '=') {
            return fail("expected `=` after the key");
        }
        ++pos_;
        skip_spaces();
        auto value = this->value();
        if (!value) {
            return value.error();
        }
        const std::string last = parts->back();
        parts->pop_back();
        auto target = descend(table, *parts);
        if (!target) {
            return target.error();
        }
        if (!(*target)->try_emplace(last, std::move(*value)).second) {
            return fail("key `" + last + "` is defined twice");
        }
        return std::nullopt;
    }

    // ----------------------------------------------------------------- values

    std::expected<Value, ParseError> value() {
        const char c = peek();
        if (c == '"' || c == '\'') {
            return string_value();
        }
        if (c == '[') {
            return array_value();
        }
        if (c == '{') {
            return inline_table();
        }
        if (text_.substr(pos_).starts_with("true") && !is_bare_key_char(peek(4))) {
            pos_ += 4;
            return Value{true, line_};
        }
        if (text_.substr(pos_).starts_with("false") && !is_bare_key_char(peek(5))) {
            pos_ += 5;
            return Value{false, line_};
        }
        if (c == '-' || c == '+' || (c >= '0' && c <= '9')) {
            return integer_value();
        }
        return std::unexpected(fail("expected a value"));
    }

    std::expected<Value, ParseError> integer_value() {
        const std::size_t begin = pos_;
        const bool is_negative = peek() == '-';
        if (peek() == '-' || peek() == '+') {
            ++pos_;
        }
        std::int64_t magnitude = 0;
        bool has_digit = false;
        while ((peek() >= '0' && peek() <= '9') || peek() == '_') {
            if (peek() != '_') {
                const int digit = peek() - '0';
                if (magnitude > (std::numeric_limits<std::int64_t>::max() - digit) / 10) {
                    return std::unexpected(fail("integer is too large"));
                }
                magnitude = magnitude * 10 + digit;
                has_digit = true;
            }
            ++pos_;
        }
        if (!has_digit || peek() == '.' || peek() == 'e' || peek() == 'E' || peek() == ':' ||
            peek() == '-' || is_bare_key_char(peek())) {
            pos_ = begin;
            return std::unexpected(fail("only integers, strings, booleans, arrays, and inline "
                                        "tables are supported as values"));
        }
        return Value{is_negative ? -magnitude : magnitude, line_};
    }

    std::expected<Value, ParseError> string_value() {
        const char quote = peek();
        if (peek(1) == quote && peek(2) == quote) {
            return std::unexpected(fail("multi-line strings are not supported"));
        }
        ++pos_;
        std::string out;
        while (true) {
            const char c = peek();
            if (pos_ >= text_.size() || c == '\n' || c == '\r') {
                return std::unexpected(fail("unterminated string"));
            }
            ++pos_;
            if (c == quote) {
                return Value{std::move(out), line_};
            }
            if (c != '\\' || quote == '\'') {
                out += c;
                continue;
            }
            const char escaped = peek();
            ++pos_;
            switch (escaped) {
            case 'n':
                out += '\n';
                break;
            case 't':
                out += '\t';
                break;
            case 'r':
                out += '\r';
                break;
            case '"':
            case '\\':
                out += escaped;
                break;
            default:
                return std::unexpected(fail("unsupported escape sequence in string"));
            }
        }
    }

    std::expected<Value, ParseError> array_value() {
        ++pos_;
        Array items;
        while (true) {
            skip_blank_lines();
            if (peek() == ']') {
                ++pos_;
                return Value{std::move(items), line_};
            }
            auto item = value();
            if (!item) {
                return std::unexpected(item.error());
            }
            items.push_back(std::move(*item));
            skip_blank_lines();
            if (peek() == ',') {
                ++pos_;
            } else if (peek() != ']') {
                return std::unexpected(fail("expected `,` or `]` in array"));
            }
        }
    }

    std::expected<Value, ParseError> inline_table() {
        ++pos_;
        Table table;
        skip_spaces();
        if (peek() == '}') {
            ++pos_;
            return Value{std::move(table), line_};
        }
        while (true) {
            skip_spaces();
            if (auto error = key_value(table)) {
                return std::unexpected(*error);
            }
            skip_spaces();
            if (peek() == '}') {
                ++pos_;
                return Value{std::move(table), line_};
            }
            if (peek() != ',') {
                return std::unexpected(fail("expected `,` or `}` in inline table"));
            }
            ++pos_;
        }
    }

    std::string_view text_;
    std::size_t pos_ = 0;
    std::uint32_t line_ = 1;
    std::set<const Table*> defined_tables_;
};

} // namespace

std::expected<Table, ParseError> parse(std::string_view text) {
    return Parser(text).run();
}

} // namespace linnet::toml
