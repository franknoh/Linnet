#pragma once

#include "linnet/source/source_manager.hpp"

#include <cstdint>
#include <string_view>

namespace linnet {

// X-macro over every token kind: X(EnumName, "display spelling").
// Keywords must stay contiguous between KwModule and KwExtern.
#define LINNET_TOKEN_KINDS(X)                                                                      \
    X(Eof, "end of file")                                                                          \
    X(Identifier, "identifier")                                                                    \
    X(ReservedWord, "reserved word")                                                               \
    X(Integer, "integer literal")                                                                  \
    X(Float, "float literal")                                                                      \
    X(String, "string literal")                                                                    \
    X(KwModule, "module")                                                                          \
    X(KwUse, "use")                                                                                \
    X(KwPub, "pub")                                                                                \
    X(KwCrate, "crate")                                                                            \
    X(KwSelf, "self")                                                                              \
    X(KwSuper, "super")                                                                            \
    X(KwAs, "as")                                                                                  \
    X(KwConst, "const")                                                                            \
    X(KwType, "type")                                                                              \
    X(KwStruct, "struct")                                                                          \
    X(KwEnum, "enum")                                                                              \
    X(KwFn, "fn")                                                                                  \
    X(KwOp, "op")                                                                                  \
    X(KwBlock, "block")                                                                            \
    X(KwEntry, "entry")                                                                            \
    X(KwParam, "param")                                                                            \
    X(KwBuffer, "buffer")                                                                          \
    X(KwState, "state")                                                                            \
    X(KwSub, "sub")                                                                                \
    X(KwLet, "let")                                                                                \
    X(KwVar, "var")                                                                                \
    X(KwReturn, "return")                                                                          \
    X(KwIf, "if")                                                                                  \
    X(KwElse, "else")                                                                              \
    X(KwMatch, "match")                                                                            \
    X(KwStatic, "static")                                                                          \
    X(KwFor, "for")                                                                                \
    X(KwIn, "in")                                                                                  \
    X(KwWhile, "while")                                                                            \
    X(KwWhere, "where")                                                                            \
    X(KwTrue, "true")                                                                              \
    X(KwFalse, "false")                                                                            \
    X(KwNone, "none")                                                                              \
    X(KwSome, "some")                                                                              \
    X(KwExtern, "extern")                                                                          \
    X(LParen, "(")                                                                                 \
    X(RParen, ")")                                                                                 \
    X(LBracket, "[")                                                                               \
    X(RBracket, "]")                                                                               \
    X(LBrace, "{")                                                                                 \
    X(RBrace, "}")                                                                                 \
    X(Comma, ",")                                                                                  \
    X(Colon, ":")                                                                                  \
    X(ColonColon, "::")                                                                            \
    X(Semicolon, ";")                                                                              \
    X(Dot, ".")                                                                                    \
    X(DotDot, "..")                                                                                \
    X(Ellipsis, "...")                                                                             \
    X(Arrow, "->")                                                                                 \
    X(FatArrow, "=>")                                                                              \
    X(Equal, "=")                                                                                  \
    X(EqualEqual, "==")                                                                            \
    X(BangEqual, "!=")                                                                             \
    X(Less, "<")                                                                                   \
    X(LessEqual, "<=")                                                                             \
    X(Greater, ">")                                                                                \
    X(GreaterEqual, ">=")                                                                          \
    X(Plus, "+")                                                                                   \
    X(Minus, "-")                                                                                  \
    X(Star, "*")                                                                                   \
    X(Slash, "/")                                                                                  \
    X(Percent, "%")                                                                                \
    X(Bang, "!")                                                                                   \
    X(AmpAmp, "&&")                                                                                \
    X(PipePipe, "||")                                                                              \
    X(Question, "?")

enum class TokenKind : std::uint8_t {
#define LINNET_TOKEN_ENUM(name, spelling) name,
    LINNET_TOKEN_KINDS(LINNET_TOKEN_ENUM)
#undef LINNET_TOKEN_ENUM
};

// Spelling for keywords and punctuation, a description for other kinds.
std::string_view token_kind_name(TokenKind kind);

constexpr bool is_keyword(TokenKind kind) {
    return kind >= TokenKind::KwModule && kind <= TokenKind::KwExtern;
}

// Token text is recovered from the source span on demand.
struct Token {
    TokenKind kind = TokenKind::Eof;
    SourceSpan span;
};

enum class CommentKind : std::uint8_t { Line, Block, DocLine, DocBlock };

struct Comment {
    CommentKind kind = CommentKind::Line;
    SourceSpan span;
};

} // namespace linnet
