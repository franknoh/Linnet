#include "linnet/syntax/lexer.hpp"

#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

struct Lexed {
    SourceManager sources;
    DiagnosticSink sink;
    LexResult result;
    FileId file = invalid_file_id;

    explicit Lexed(std::string text) {
        file = sources.add_file("test.linnet", std::move(text)).value();
        result = lex(sources, file, sink);
    }

    // Token texts separated by single spaces, without the final Eof.
    std::string texts() const {
        std::string out;
        for (const Token& token : result.tokens) {
            if (token.kind == TokenKind::Eof) {
                break;
            }
            if (!out.empty()) {
                out += ' ';
            }
            out += sources.text(token.span);
        }
        return out;
    }

    std::string codes() const {
        std::string out;
        for (const Diagnostic& diagnostic : sink.diagnostics()) {
            if (!out.empty()) {
                out += ' ';
            }
            out += diagnostic.code;
        }
        return out;
    }

    TokenKind kind(std::size_t index) const { return result.tokens.at(index).kind; }
};

} // namespace

TEST("lexer: empty input yields only Eof") {
    const Lexed lexed("");
    CHECK_EQ(lexed.result.tokens.size(), 1U);
    CHECK(lexed.kind(0) == TokenKind::Eof);
    CHECK(lexed.result.tokens[0].span == SourceSpan{lexed.file, 0, 0});
}

TEST("lexer: every keyword is recognized and round-trips its spelling") {
    const std::string keywords = "module use pub crate self super as const type struct enum fn op "
                                 "block entry param buffer state sub let var return if else match "
                                 "static for in while where true false none some extern";
    const Lexed lexed(keywords);
    CHECK(!lexed.sink.has_errors());
    CHECK_EQ(lexed.texts(), keywords);
    for (const Token& token : lexed.result.tokens) {
        if (token.kind != TokenKind::Eof) {
            CHECK(is_keyword(token.kind));
            CHECK_EQ(token_kind_name(token.kind), lexed.sources.text(token.span));
        }
    }
}

TEST("lexer: reserved words are distinguished from identifiers") {
    const Lexed lexed("async await effect unsafe macro trait impl derive rng mut ref yield "
                      "kernel device asyncs Tensor f32 sum _x x1 std");
    CHECK(!lexed.sink.has_errors());
    for (std::size_t i = 0; i < 14; ++i) {
        CHECK(lexed.kind(i) == TokenKind::ReservedWord);
    }
    for (std::size_t i = 14; i < 21; ++i) {
        CHECK(lexed.kind(i) == TokenKind::Identifier);
    }
}

TEST("lexer: punctuation uses maximal munch") {
    const Lexed lexed("...:: .. -> => == != <= >= && || ( ) [ ] { } , : ; . = < > + - * / % ! ?");
    CHECK(!lexed.sink.has_errors());
    CHECK(lexed.kind(0) == TokenKind::Ellipsis);
    CHECK(lexed.kind(1) == TokenKind::ColonColon);
    CHECK(lexed.kind(2) == TokenKind::DotDot);
    CHECK_EQ(lexed.result.tokens.size(), 32U);
    for (const Token& token : lexed.result.tokens) {
        if (token.kind != TokenKind::Eof) {
            CHECK_EQ(token_kind_name(token.kind), lexed.sources.text(token.span));
        }
    }
}

TEST("lexer: integer and float literals") {
    const Lexed lexed("0 42 1_000_000 0xff 0xFF_ff 0b1010 0.0 1.5 1e-5 3.141_592 2E+10 1e5 7.0e3");
    CHECK_EQ(lexed.codes(), "");
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(lexed.kind(i) == TokenKind::Integer);
    }
    for (std::size_t i = 6; i < 13; ++i) {
        CHECK(lexed.kind(i) == TokenKind::Float);
    }
}

TEST("lexer: numbers next to dots and colons") {
    const Lexed range("0..Layers");
    CHECK_EQ(range.texts(), "0 .. Layers");
    const Lexed slice("x[..., 0::2]");
    CHECK_EQ(slice.texts(), "x [ ... , 0 :: 2 ]");
    const Lexed hex_minus("0x1e-5");
    CHECK_EQ(hex_minus.texts(), "0x1e - 5");
    const Lexed member("1.x");
    CHECK_EQ(member.texts(), "1 . x");
    const Lexed underscore("1._5");
    CHECK_EQ(underscore.texts(), "1 . _5");
}

TEST("lexer: malformed numbers are single tokens with one diagnostic") {
    const char* cases[] = {"1_",
                           "1__0",
                           "0x",
                           "0xZZ",
                           "0b102",
                           "12abc",
                           "1e",
                           "1e+",
                           "1.5_",
                           "007",
                           "0X1F",
                           "1.5e",
                           "00.5"};
    for (const char* text : cases) {
        const Lexed lexed(text);
        CHECK_EQ(lexed.codes(), "E1005");
    }
    const Lexed lexed("12abc");
    CHECK_EQ(lexed.result.tokens.size(), 2U);
    CHECK(lexed.kind(0) == TokenKind::Integer);
}

TEST("lexer: strings and escapes") {
    const Lexed lexed(R"("model.layers.0.weight" "a\n\t\\\"\0\u{1F600}" ")"
                      "\xC3\xA9"
                      R"(" "")");
    CHECK_EQ(lexed.codes(), "");
    CHECK_EQ(lexed.result.tokens.size(), 5U);
    CHECK(lexed.kind(1) == TokenKind::String);
}

TEST("lexer: malformed strings") {
    CHECK_EQ(Lexed("\"abc").codes(), "E1003");
    CHECK_EQ(Lexed("\"abc\nlet").texts(), "\"abc let");
    CHECK_EQ(Lexed(R"("\q")").codes(), "E1003");
    CHECK_EQ(Lexed(R"("\u{}")").codes(), "E1003");
    CHECK_EQ(Lexed(R"("\u{D800}")").codes(), "E1003");
    CHECK_EQ(Lexed(R"("\u{1234567}")").codes(), "E1003");
    CHECK_EQ(Lexed("\"\\").codes(), "E1003 E1003");
    CHECK_EQ(Lexed("\"\xFF\"").codes(), "E1001");
}

TEST("lexer: comments are collected separately") {
    const Lexed lexed("a // line\n/// doc\n//// plain\n/* block */ b /** doc */ /**/ /*** x */");
    CHECK_EQ(lexed.codes(), "");
    CHECK_EQ(lexed.texts(), "a b");
    const auto& comments = lexed.result.comments;
    CHECK_EQ(comments.size(), 7U);
    CHECK(comments[0].kind == CommentKind::Line);
    CHECK_EQ(lexed.sources.text(comments[0].span), "// line");
    CHECK(comments[1].kind == CommentKind::DocLine);
    CHECK(comments[2].kind == CommentKind::Line);
    CHECK(comments[3].kind == CommentKind::Block);
    CHECK(comments[4].kind == CommentKind::DocBlock);
    CHECK(comments[5].kind == CommentKind::Block);
    CHECK(comments[6].kind == CommentKind::Block);
}

TEST("lexer: line comment excludes carriage return") {
    const Lexed lexed("// note\r\nx");
    CHECK_EQ(lexed.sources.text(lexed.result.comments.at(0).span), "// note");
}

TEST("lexer: block comments nest") {
    const Lexed lexed("/* outer /* inner */ still outer */ x");
    CHECK_EQ(lexed.codes(), "");
    CHECK_EQ(lexed.texts(), "x");
}

TEST("lexer: unterminated block comment") {
    const Lexed lexed("x /* outer /* inner */ y");
    CHECK_EQ(lexed.codes(), "E1002");
    CHECK_EQ(lexed.texts(), "x");
    CHECK(lexed.sink.diagnostics()[0].primary.span == SourceSpan{lexed.file, 2, 4});
}

TEST("lexer: invalid characters are grouped and skipped") {
    const Lexed lexed("a @#$ b");
    CHECK_EQ(lexed.codes(), "E1001");
    CHECK_EQ(lexed.texts(), "a b");
    CHECK(lexed.sink.diagnostics()[0].primary.span == SourceSpan{lexed.file, 2, 5});

    CHECK_EQ(Lexed("a $ b @ c").codes(), "E1001 E1001");
    CHECK_EQ(Lexed("let caf\xC3\xA9 = 1").codes(), "E1001");
    CHECK_EQ(Lexed("\xFF\xFE").codes(), "E1001");
    CHECK_EQ(Lexed(std::string("a\0b", 3)).codes(), "E1001");
    CHECK_EQ(Lexed("a\fb").codes(), "E1001");
}

TEST("lexer: byte-order mark is skipped") {
    const Lexed lexed("\xEF\xBB\xBFmodule m");
    CHECK_EQ(lexed.codes(), "");
    CHECK_EQ(lexed.texts(), "module m");
}

TEST("lexer: spans are contiguous with the source") {
    const Lexed lexed("let y[i] = sum[j] a[i, j] * b[j]");
    CHECK_EQ(lexed.texts(), "let y [ i ] = sum [ j ] a [ i , j ] * b [ j ]");
    CHECK(lexed.result.tokens.back().kind == TokenKind::Eof);
    CHECK_EQ(lexed.result.tokens.back().span.begin, 32U);
}
