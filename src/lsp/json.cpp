#include "linnet/lsp/json.hpp"

#include "linnet/diagnostic/json.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace linnet::lsp {

namespace {

const std::string empty_string;
const Json::Array empty_array;
const Json::Object empty_object;
const Json null_json;

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    std::expected<Json, std::string> run() {
        auto value = parse_value(0);
        if (!value) {
            return value;
        }
        skip_whitespace();
        if (pos_ != text_.size()) {
            return std::unexpected(fail("trailing characters after the document"));
        }
        return value;
    }

private:
    static constexpr int max_depth = 256;

    char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

    std::string fail(std::string_view message) const {
        return "JSON at byte " + std::to_string(pos_) + ": " + std::string(message);
    }

    void skip_whitespace() {
        while (peek() == ' ' || peek() == '\t' || peek() == '\n' || peek() == '\r') {
            ++pos_;
        }
    }

    bool consume(std::string_view literal) {
        if (text_.substr(pos_).starts_with(literal)) {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    std::expected<Json, std::string> parse_value(int depth) {
        if (depth > max_depth) {
            return std::unexpected(fail("nesting is too deep"));
        }
        skip_whitespace();
        const char c = peek();
        if (c == '{') {
            return parse_object(depth);
        }
        if (c == '[') {
            return parse_array(depth);
        }
        if (c == '"') {
            auto text = parse_string();
            if (!text) {
                return std::unexpected(text.error());
            }
            return Json(std::move(*text));
        }
        if (consume("true")) {
            return Json(true);
        }
        if (consume("false")) {
            return Json(false);
        }
        if (consume("null")) {
            return Json(nullptr);
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number();
        }
        return std::unexpected(fail("expected a value"));
    }

    std::expected<Json, std::string> parse_number() {
        const std::size_t begin = pos_;
        while (pos_ < text_.size() &&
               (std::isdigit(static_cast<unsigned char>(peek())) != 0 || peek() == '-' ||
                peek() == '+' || peek() == '.' || peek() == 'e' || peek() == 'E')) {
            ++pos_;
        }
        // strtod accepts more than JSON (hex, infinity), but the scan above
        // only admits JSON's number characters. Apple's libc++ lacks the
        // floating-point std::from_chars, so strtod is the portable choice.
        const std::string digits(text_.substr(begin, pos_ - begin));
        char* end = nullptr;
        const double value = std::strtod(digits.c_str(), &end);
        if (digits.empty() || end != digits.c_str() + digits.size() || !std::isfinite(value)) {
            return std::unexpected(fail("malformed number"));
        }
        return Json(value);
    }

    static void append_utf8(std::string& out, std::uint32_t code_point) {
        if (code_point < 0x80) {
            out += static_cast<char>(code_point);
        } else if (code_point < 0x800) {
            out += static_cast<char>(0xC0 | (code_point >> 6));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        } else if (code_point < 0x10000) {
            out += static_cast<char>(0xE0 | (code_point >> 12));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code_point >> 18));
            out += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code_point & 0x3F));
        }
    }

    std::expected<std::uint32_t, std::string> parse_hex4() {
        if (pos_ + 4 > text_.size()) {
            return std::unexpected(fail("truncated \\u escape"));
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            const int digit = c >= '0' && c <= '9'   ? c - '0'
                              : c >= 'a' && c <= 'f' ? c - 'a' + 10
                              : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                     : -1;
            if (digit < 0) {
                return std::unexpected(fail("malformed \\u escape"));
            }
            value = value * 16 + static_cast<std::uint32_t>(digit);
        }
        return value;
    }

    std::expected<std::string, std::string> parse_string() {
        ++pos_; // opening quote
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) {
                return std::unexpected(fail("unterminated string"));
            }
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                return std::unexpected(fail("control character in string"));
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= text_.size()) {
                return std::unexpected(fail("unterminated escape"));
            }
            const char escaped = text_[pos_++];
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                out += escaped;
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                auto unit = parse_hex4();
                if (!unit) {
                    return std::unexpected(unit.error());
                }
                std::uint32_t code_point = *unit;
                if (code_point >= 0xD800 && code_point <= 0xDBFF && consume("\\u")) {
                    auto low = parse_hex4();
                    if (!low) {
                        return std::unexpected(low.error());
                    }
                    if (*low >= 0xDC00 && *low <= 0xDFFF) {
                        code_point = 0x10000 + ((code_point - 0xD800) << 10) + (*low - 0xDC00);
                    } else {
                        code_point = 0xFFFD;
                    }
                } else if (code_point >= 0xD800 && code_point <= 0xDFFF) {
                    code_point = 0xFFFD;
                }
                append_utf8(out, code_point);
                break;
            }
            default:
                return std::unexpected(fail("unknown escape"));
            }
        }
    }

    std::expected<Json, std::string> parse_array(int depth) {
        ++pos_;
        Json::Array items;
        skip_whitespace();
        if (peek() == ']') {
            ++pos_;
            return Json(std::move(items));
        }
        while (true) {
            auto item = parse_value(depth + 1);
            if (!item) {
                return item;
            }
            items.push_back(std::move(*item));
            skip_whitespace();
            if (peek() == ',') {
                ++pos_;
            } else if (peek() == ']') {
                ++pos_;
                return Json(std::move(items));
            } else {
                return std::unexpected(fail("expected `,` or `]`"));
            }
        }
    }

    std::expected<Json, std::string> parse_object(int depth) {
        ++pos_;
        Json::Object members;
        skip_whitespace();
        if (peek() == '}') {
            ++pos_;
            return Json(std::move(members));
        }
        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                return std::unexpected(fail("expected a string key"));
            }
            auto key = parse_string();
            if (!key) {
                return std::unexpected(key.error());
            }
            skip_whitespace();
            if (peek() != ':') {
                return std::unexpected(fail("expected `:`"));
            }
            ++pos_;
            auto value = parse_value(depth + 1);
            if (!value) {
                return value;
            }
            members[std::move(*key)] = std::move(*value);
            skip_whitespace();
            if (peek() == ',') {
                ++pos_;
            } else if (peek() == '}') {
                ++pos_;
                return Json(std::move(members));
            } else {
                return std::unexpected(fail("expected `,` or `}`"));
            }
        }
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

} // namespace

const std::string& Json::as_string() const {
    const auto* value = std::get_if<std::string>(&data_);
    return value != nullptr ? *value : empty_string;
}

double Json::as_number() const {
    const auto* value = std::get_if<double>(&data_);
    return value != nullptr ? *value : 0.0;
}

bool Json::as_bool() const {
    const auto* value = std::get_if<bool>(&data_);
    return value != nullptr && *value;
}

const Json::Array& Json::as_array() const {
    const auto* value = std::get_if<Array>(&data_);
    return value != nullptr ? *value : empty_array;
}

const Json::Object& Json::as_object() const {
    const auto* value = std::get_if<Object>(&data_);
    return value != nullptr ? *value : empty_object;
}

const Json& Json::operator[](std::string_view key) const {
    const auto* object = std::get_if<Object>(&data_);
    if (object == nullptr) {
        return null_json;
    }
    const auto found = object->find(std::string(key));
    return found == object->end() ? null_json : found->second;
}

Json& Json::set(std::string key, Json value) {
    if (!is_object()) {
        data_ = Object{};
    }
    std::get<Object>(data_)[std::move(key)] = std::move(value);
    return *this;
}

std::string Json::dump() const {
    return std::visit(
        [](const auto& value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::nullptr_t>) {
                return "null";
            } else if constexpr (std::is_same_v<T, bool>) {
                return value ? "true" : "false";
            } else if constexpr (std::is_same_v<T, double>) {
                if (value == std::floor(value) && std::fabs(value) < 9007199254740992.0) {
                    return std::to_string(static_cast<std::int64_t>(value));
                }
                std::array<char, 32> buffer{};
                const auto result =
                    std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
                return std::string(buffer.data(), result.ptr);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return json_string(value);
            } else if constexpr (std::is_same_v<T, Array>) {
                std::string out = "[";
                for (const Json& item : value) {
                    out += out.size() == 1 ? "" : ",";
                    out += item.dump();
                }
                return out + "]";
            } else {
                std::string out = "{";
                for (const auto& [key, item] : value) {
                    out += out.size() == 1 ? "" : ",";
                    out += json_string(key) + ":" + item.dump();
                }
                return out + "}";
            }
        },
        data_);
}

std::expected<Json, std::string> parse_json(std::string_view text) {
    return Parser(text).run();
}

} // namespace linnet::lsp
