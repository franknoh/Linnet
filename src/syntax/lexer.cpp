#include "linnet/syntax/lexer.hpp"

#include "linnet/diagnostic/codes.hpp"
#include "linnet/source/utf8.hpp"

#include <array>
#include <string>
#include <utility>

namespace linnet {

std::string_view token_kind_name(TokenKind kind) {
    static constexpr auto names = std::to_array<std::string_view>({
#define LINNET_TOKEN_NAME(name, spelling) spelling,
        LINNET_TOKEN_KINDS(LINNET_TOKEN_NAME)
#undef LINNET_TOKEN_NAME
    });
    return names[static_cast<std::size_t>(kind)];
}

namespace {

constexpr bool is_digit(char c) {
    return c >= '0' && c <= '9';
}
constexpr bool is_hex_digit(char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
constexpr bool is_identifier_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
constexpr bool is_identifier_continue(char c) {
    return is_identifier_start(c) || is_digit(c);
}
constexpr bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

constexpr auto reserved_words = std::to_array<std::string_view>({
    "async",
    "await",
    "effect",
    "unsafe",
    "macro",
    "trait",
    "impl",
    "derive",
    "rng",
    "mut",
    "ref",
    "yield",
    "kernel",
    "device",
});

// Validates `digits` as one or more digits with `_` allowed only between digits.
template <typename IsDigit>
bool valid_digit_run(std::string_view digits, IsDigit is_valid_digit) {
    if (digits.empty() || digits.front() == '_' || digits.back() == '_') {
        return false;
    }
    for (std::size_t i = 0; i < digits.size(); ++i) {
        const char c = digits[i];
        if (c == '_') {
            if (digits[i - 1] == '_') {
                return false;
            }
        } else if (!is_valid_digit(c)) {
            return false;
        }
    }
    return true;
}

// Returns an explanation when `text` is not a well-formed numeric literal.
const char* validate_number(std::string_view text) {
    if (text.starts_with("0x")) {
        return valid_digit_run(text.substr(2), is_hex_digit) ? nullptr
                                                             : "malformed hexadecimal literal";
    }
    if (text.starts_with("0b")) {
        const auto is_binary = [](char c) { return c == '0' || c == '1'; };
        return valid_digit_run(text.substr(2), is_binary) ? nullptr : "malformed binary literal";
    }

    const std::size_t exponent_at = text.find_first_of("eE");
    const std::string_view mantissa = text.substr(0, exponent_at);
    const std::size_t dot_at = mantissa.find('.');
    const std::string_view integer_part = mantissa.substr(0, dot_at);

    if (!valid_digit_run(integer_part, is_digit)) {
        return "malformed numeric literal";
    }
    if (integer_part.size() > 1 && integer_part.front() == '0') {
        return "leading zeros are not allowed in decimal literals";
    }
    if (dot_at != std::string_view::npos &&
        !valid_digit_run(mantissa.substr(dot_at + 1), is_digit)) {
        return "malformed fractional part";
    }
    if (exponent_at != std::string_view::npos) {
        std::string_view exponent = text.substr(exponent_at + 1);
        if (exponent.starts_with('+') || exponent.starts_with('-')) {
            exponent.remove_prefix(1);
        }
        if (!valid_digit_run(exponent, is_digit)) {
            return "malformed exponent";
        }
    }
    return nullptr;
}

class Lexer {
public:
    Lexer(const SourceManager& sources, FileId file, DiagnosticSink& sink)
        : file_(file), text_(sources.contents(file)), sink_(sink) {}

    LexResult run() {
        if (text_.starts_with("\xEF\xBB\xBF")) {
            pos_ = 3; // byte-order mark
        }
        while (pos_ < text_.size()) {
            lex_one();
        }
        flush_invalid_run();
        push(TokenKind::Eof, pos_);
        return std::move(result_);
    }

private:
    char peek(std::size_t ahead = 0) const {
        return pos_ + ahead < text_.size() ? text_[pos_ + ahead] : '\0';
    }

    SourceSpan span_from(std::size_t begin) const {
        return {file_, static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(pos_)};
    }

    void push(TokenKind kind, std::size_t begin) {
        result_.tokens.push_back({kind, span_from(begin)});
    }

    Diagnostic& error(const char* code, SourceSpan span, std::string message) {
        Diagnostic diagnostic;
        diagnostic.code = code;
        diagnostic.message = std::move(message);
        diagnostic.primary.span = span;
        pending_.push_back(std::move(diagnostic));
        return pending_.back();
    }

    void flush_diagnostics() {
        for (Diagnostic& diagnostic : pending_) {
            sink_.report(std::move(diagnostic));
        }
        pending_.clear();
    }

    void lex_one() {
        const char c = peek();
        if (is_whitespace(c)) {
            flush_invalid_run();
            ++pos_;
            return;
        }
        if (c == '/' && (peek(1) == '/' || peek(1) == '*')) {
            flush_invalid_run();
            peek(1) == '/' ? line_comment() : block_comment();
        } else if (is_identifier_start(c)) {
            flush_invalid_run();
            word();
        } else if (is_digit(c)) {
            flush_invalid_run();
            number();
        } else if (c == '"') {
            flush_invalid_run();
            string();
        } else if (!punctuation()) {
            invalid_character();
        }
        flush_diagnostics();
    }

    // Reports invalid UTF-8 inside comments and strings, advancing one code point.
    void advance_code_point() {
        const DecodedCodePoint decoded = decode_utf8(text_, pos_);
        const std::size_t begin = pos_;
        pos_ += decoded.length;
        if (!decoded.valid) {
            error(codes::invalid_character, span_from(begin), "invalid UTF-8 byte sequence")
                .notes.emplace_back("source files must be UTF-8");
        }
    }

    void line_comment() {
        const std::size_t begin = pos_;
        const bool is_doc = peek(2) == '/' && peek(3) != '/';
        while (pos_ < text_.size() && peek() != '\n') {
            advance_code_point();
        }
        std::size_t end = pos_;
        if (end > begin && text_[end - 1] == '\r') {
            --end;
        }
        result_.comments.push_back(
            {is_doc ? CommentKind::DocLine : CommentKind::Line,
             {file_, static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(end)}});
    }

    void block_comment() {
        const std::size_t begin = pos_;
        const bool is_doc = peek(2) == '*' && peek(3) != '*' && peek(3) != '/';
        pos_ += 2;
        std::size_t depth = 1;
        while (pos_ < text_.size() && depth != 0) {
            if (peek() == '/' && peek(1) == '*') {
                pos_ += 2;
                ++depth;
            } else if (peek() == '*' && peek(1) == '/') {
                pos_ += 2;
                --depth;
            } else {
                advance_code_point();
            }
        }
        if (depth != 0) {
            const SourceSpan opener{
                file_, static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(begin + 2)};
            error(codes::unterminated_comment, opener, "unterminated block comment")
                .notes.emplace_back("block comments nest; every `/*` needs a matching `*/`");
        }
        result_.comments.push_back(
            {is_doc ? CommentKind::DocBlock : CommentKind::Block, span_from(begin)});
    }

    void word() {
        const std::size_t begin = pos_;
        while (is_identifier_continue(peek())) {
            ++pos_;
        }
        push(classify_word(text_.substr(begin, pos_ - begin)), begin);
    }

    void consume_alphanumeric_run() {
        while (is_identifier_continue(peek())) {
            ++pos_;
        }
    }

    void consume_exponent_sign(std::size_t begin) {
        const char last = text_[pos_ - 1];
        const bool is_prefixed =
            text_.substr(begin).starts_with("0x") || text_.substr(begin).starts_with("0b");
        if (!is_prefixed && (last == 'e' || last == 'E') && (peek() == '+' || peek() == '-') &&
            is_digit(peek(1))) {
            ++pos_;
            consume_alphanumeric_run();
        }
    }

    void number() {
        const std::size_t begin = pos_;
        consume_alphanumeric_run();
        consume_exponent_sign(begin);

        // A fraction needs a digit after the dot, so `0..N` and `0::2` stay intact.
        const std::string_view integer_part = text_.substr(begin, pos_ - begin);
        const bool is_plain_decimal =
            integer_part.find_first_not_of("0123456789_") == std::string_view::npos;
        if (is_plain_decimal && peek() == '.' && is_digit(peek(1))) {
            ++pos_;
            consume_alphanumeric_run();
            consume_exponent_sign(begin);
        }

        const std::string_view literal = text_.substr(begin, pos_ - begin);
        const bool is_prefixed = literal.starts_with("0x") || literal.starts_with("0b");
        const bool is_float =
            !is_prefixed && literal.find_first_of(".eE") != std::string_view::npos;
        if (const char* problem = validate_number(literal)) {
            error(codes::invalid_number, span_from(begin), problem);
        }
        push(is_float ? TokenKind::Float : TokenKind::Integer, begin);
    }

    void escape_sequence() {
        const std::size_t begin = pos_;
        ++pos_; // backslash
        const char c = peek();
        if (c == 'n' || c == 'r' || c == 't' || c == '0' || c == '\\' || c == '"') {
            ++pos_;
            return;
        }
        if (c == 'u' && peek(1) == '{') {
            pos_ += 2;
            std::uint32_t value = 0;
            std::size_t digits = 0;
            while (is_hex_digit(peek())) {
                const char h = peek();
                const int digit = is_digit(h) ? h - '0' : (h | 0x20) - 'a' + 10;
                value = digits < 7 ? value * 16 + static_cast<std::uint32_t>(digit) : value;
                ++digits;
                ++pos_;
            }
            const bool is_closed = peek() == '}';
            if (is_closed) {
                ++pos_;
            }
            const bool is_scalar = value <= 0x10FFFF && (value < 0xD800 || value > 0xDFFF);
            if (!is_closed || digits == 0 || digits > 6 || !is_scalar) {
                error(codes::invalid_string, span_from(begin), "invalid Unicode escape")
                    .help.emplace_back(
                        "write `\\u{...}` with 1 to 6 hex digits naming a Unicode scalar value");
            }
            return;
        }
        if (pos_ < text_.size() && c != '\n' && c != '\r') {
            advance_code_point();
        }
        error(codes::invalid_string, span_from(begin), "unknown escape sequence")
            .help.emplace_back("supported escapes: \\n \\r \\t \\0 \\\\ \\\" \\u{...}");
    }

    void string() {
        const std::size_t begin = pos_;
        ++pos_;
        while (true) {
            const char c = peek();
            if (pos_ >= text_.size() || c == '\n' || (c == '\r' && peek(1) == '\n')) {
                error(codes::invalid_string, span_from(begin), "unterminated string literal")
                    .notes.emplace_back("string literals cannot span lines");
                break;
            }
            if (c == '"') {
                ++pos_;
                break;
            }
            if (c == '\\') {
                escape_sequence();
            } else {
                advance_code_point();
            }
        }
        push(TokenKind::String, begin);
    }

    bool punctuation() {
        struct Entry {
            std::string_view spelling;
            TokenKind kind;
        };
        // Longest spellings first so that maximal munch applies.
        static constexpr auto table = std::to_array<Entry>({
            {"...", TokenKind::Ellipsis},    {"::", TokenKind::ColonColon},
            {"..", TokenKind::DotDot},       {"->", TokenKind::Arrow},
            {"=>", TokenKind::FatArrow},     {"==", TokenKind::EqualEqual},
            {"!=", TokenKind::BangEqual},    {"<=", TokenKind::LessEqual},
            {">=", TokenKind::GreaterEqual}, {"&&", TokenKind::AmpAmp},
            {"||", TokenKind::PipePipe},     {"&", TokenKind::Amp},
            {"|", TokenKind::Pipe},          {"^", TokenKind::Caret},
            {"(", TokenKind::LParen},        {")", TokenKind::RParen},
            {"[", TokenKind::LBracket},      {"]", TokenKind::RBracket},
            {"{", TokenKind::LBrace},        {"}", TokenKind::RBrace},
            {",", TokenKind::Comma},         {":", TokenKind::Colon},
            {";", TokenKind::Semicolon},     {".", TokenKind::Dot},
            {"=", TokenKind::Equal},         {"<", TokenKind::Less},
            {">", TokenKind::Greater},       {"+", TokenKind::Plus},
            {"-", TokenKind::Minus},         {"*", TokenKind::Star},
            {"/", TokenKind::Slash},         {"%", TokenKind::Percent},
            {"!", TokenKind::Bang},          {"?", TokenKind::Question},
        });
        const std::string_view rest = text_.substr(pos_);
        for (const Entry& entry : table) {
            if (rest.starts_with(entry.spelling)) {
                flush_invalid_run();
                const std::size_t begin = pos_;
                pos_ += entry.spelling.size();
                push(entry.kind, begin);
                return true;
            }
        }
        return false;
    }

    // Adjacent characters that cannot start a token are reported together.
    void invalid_character() {
        if (!has_invalid_run_) {
            has_invalid_run_ = true;
            invalid_run_begin_ = pos_;
        }
        pos_ += decode_utf8(text_, pos_).length;
    }

    void flush_invalid_run() {
        if (!has_invalid_run_) {
            return;
        }
        has_invalid_run_ = false;
        const SourceSpan span{file_,
                              static_cast<std::uint32_t>(invalid_run_begin_),
                              static_cast<std::uint32_t>(pos_)};
        const std::string_view run = text_.substr(span.begin, span.size());

        bool is_valid_utf8 = true;
        bool is_printable = true;
        for (std::size_t offset = 0; offset < run.size();) {
            const DecodedCodePoint decoded = decode_utf8(run, offset);
            is_valid_utf8 = is_valid_utf8 && decoded.valid;
            is_printable = is_printable && decoded.value >= 0x20 && decoded.value != 0x7F;
            offset += decoded.length;
        }

        if (!is_valid_utf8) {
            error(codes::invalid_character, span, "invalid UTF-8 byte sequence")
                .notes.emplace_back("source files must be UTF-8");
        } else if (run == "&" || run == "|") {
            Diagnostic& diagnostic =
                error(codes::invalid_character, span, "unexpected `" + std::string(run) + "`");
            diagnostic.help.push_back("the logical operators are `&&` and `||`");
        } else if (!is_printable) {
            error(codes::invalid_character, span, "unexpected control character");
        } else {
            Diagnostic& diagnostic =
                error(codes::invalid_character, span, "unexpected `" + std::string(run) + "`");
            if (static_cast<unsigned char>(run.front()) >= 0x80) {
                diagnostic.notes.emplace_back(
                    "outside comments and strings, Linnet source is restricted to ASCII");
            }
        }
        flush_diagnostics();
    }

    FileId file_;
    std::string_view text_;
    DiagnosticSink& sink_;
    LexResult result_;
    std::vector<Diagnostic> pending_;
    std::size_t pos_ = 0;
    std::size_t invalid_run_begin_ = 0;
    bool has_invalid_run_ = false;
};

} // namespace

TokenKind classify_word(std::string_view word) {
    constexpr auto first = static_cast<std::size_t>(TokenKind::KwModule);
    constexpr auto last = static_cast<std::size_t>(TokenKind::KwExtern);
    for (std::size_t i = first; i <= last; ++i) {
        const auto kind = static_cast<TokenKind>(i);
        if (token_kind_name(kind) == word) {
            return kind;
        }
    }
    for (const std::string_view reserved : reserved_words) {
        if (reserved == word) {
            return TokenKind::ReservedWord;
        }
    }
    return TokenKind::Identifier;
}

LexResult lex(const SourceManager& sources, FileId file, DiagnosticSink& sink) {
    return Lexer(sources, file, sink).run();
}

} // namespace linnet
