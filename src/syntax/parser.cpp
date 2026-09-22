#include "linnet/syntax/parser.hpp"

#include "linnet/diagnostic/codes.hpp"
#include "linnet/syntax/lexer.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>

namespace linnet {

namespace {

using namespace ast;
using K = TokenKind;

constexpr std::uint32_t max_nesting_depth = 256;

struct BinaryOperator {
    K token;
    BinaryOp op;
    int precedence;
};

constexpr auto binary_operators = std::to_array<BinaryOperator>({
    {K::PipePipe, BinaryOp::Or, 1},
    {K::AmpAmp, BinaryOp::And, 2},
    {K::EqualEqual, BinaryOp::Equal, 3},
    {K::BangEqual, BinaryOp::NotEqual, 3},
    {K::Less, BinaryOp::Less, 4},
    {K::LessEqual, BinaryOp::LessEqual, 4},
    {K::Greater, BinaryOp::Greater, 4},
    {K::GreaterEqual, BinaryOp::GreaterEqual, 4},
    {K::Plus, BinaryOp::Add, 5},
    {K::Minus, BinaryOp::Subtract, 5},
    {K::Star, BinaryOp::Multiply, 6},
    {K::Slash, BinaryOp::Divide, 6},
    {K::Percent, BinaryOp::Remainder, 6},
});

constexpr int additive_precedence = 5;

constexpr bool is_comparison(BinaryOp op) {
    return op >= BinaryOp::Equal && op <= BinaryOp::GreaterEqual;
}

std::optional<ReductionKind> reduction_kind(std::string_view word) {
    constexpr auto kinds = std::to_array<ReductionKind>({ReductionKind::Sum,
                                                         ReductionKind::Prod,
                                                         ReductionKind::Max,
                                                         ReductionKind::Min,
                                                         ReductionKind::Any,
                                                         ReductionKind::All});
    for (const ReductionKind kind : kinds) {
        if (reduction_kind_spelling(kind) == word) {
            return kind;
        }
    }
    return std::nullopt;
}

ConstraintKind constraint_kind(std::string_view word) {
    constexpr auto kinds = std::to_array<std::pair<std::string_view, ConstraintKind>>({
        {"Dim", ConstraintKind::Dim},
        {"Shape", ConstraintKind::Shape},
        {"DType", ConstraintKind::DType},
        {"Numeric", ConstraintKind::Numeric},
        {"Integer", ConstraintKind::Integer},
        {"Float", ConstraintKind::Float},
    });
    for (const auto& [name, kind] : kinds) {
        if (name == word) {
            return kind;
        }
    }
    return ConstraintKind::Unknown;
}

constexpr bool starts_item(K kind) {
    switch (kind) {
    case K::KwPub:
    case K::KwConst:
    case K::KwType:
    case K::KwStruct:
    case K::KwEnum:
    case K::KwFn:
    case K::KwOp:
    case K::KwBlock:
    case K::KwEntry:
    case K::KwParam:
    case K::KwBuffer:
    case K::KwSub:
    case K::KwState:
    case K::KwExtern:
    case K::KwUse:
    case K::KwModule:
        return true;
    default:
        return false;
    }
}

constexpr bool starts_expression(K kind) {
    switch (kind) {
    case K::Identifier:
    case K::ReservedWord:
    case K::Integer:
    case K::Float:
    case K::String:
    case K::KwTrue:
    case K::KwFalse:
    case K::KwNone:
    case K::KwSome:
    case K::KwIf:
    case K::KwMatch:
    case K::LParen:
    case K::LBracket:
    case K::Bang:
    case K::Minus:
    case K::Plus:
        return true;
    default:
        return false;
    }
}

class Parser {
public:
    Parser(const SourceManager& sources, FileId file, DiagnosticSink& sink)
        : sources_(sources), sink_(sink) {
        LexResult lexed = lex(sources, file, sink);
        tokens_ = std::move(lexed.tokens);
        failed_generic_call_.assign(tokens_.size(), false);
        ast_.file = file;
        ast_.comments = std::move(lexed.comments);
    }

    Ast run() {
        parse_module_decl();
        while (!at(K::Eof)) {
            const std::size_t before = pos_;
            if (at(K::KwUse)) {
                if (!ast_.items.empty()) {
                    error_here("`use` declarations must come before all items");
                }
                parse_use();
            } else if (at(K::RBrace)) {
                error_here("unmatched `}`");
                advance();
            } else {
                ast_.items.push_back(parse_item(false));
            }
            if (pos_ == before) {
                advance();
            }
        }
        return std::move(ast_);
    }

private:
    // ------------------------------------------------------------ token access

    const Token& peek(std::size_t ahead = 0) const {
        return tokens_[std::min(pos_ + ahead, tokens_.size() - 1)];
    }
    bool at(K kind) const { return peek().kind == kind; }
    std::string_view text(const Token& token) const { return sources_.text(token.span); }

    void advance() {
        if (!at(K::Eof)) {
            last_end_ = peek().span.end;
            ++pos_;
        }
    }

    bool accept(K kind) {
        if (at(kind)) {
            advance();
            return true;
        }
        return false;
    }

    std::uint32_t here() const { return peek().span.begin; }
    SourceSpan span_from(std::uint32_t begin) const {
        return {ast_.file, begin, std::max(begin, last_end_)};
    }

    static std::string describe(const Token& token, std::string_view spelling) {
        switch (token.kind) {
        case K::Eof:
            return "end of file";
        case K::Identifier:
        case K::ReservedWord:
        case K::Integer:
        case K::Float:
            return "`" + std::string(spelling) + "`";
        case K::String:
            return "string literal";
        default:
            return (is_keyword(token.kind) ? "keyword `" : "`") +
                   std::string(token_kind_name(token.kind)) + "`";
        }
    }

    // ------------------------------------------------------------- diagnostics

    // At most one syntax error is reported per token; later complaints about
    // the same position are consequences of the first.
    Diagnostic* error_at(SourceSpan span, std::string message, const char* code) {
        if (speculation_depth_ != 0) {
            speculation_failed_ = true;
            return nullptr;
        }
        ++error_count_;
        if (has_error_position_ && error_position_ == pos_) {
            return nullptr;
        }
        has_error_position_ = true;
        error_position_ = pos_;

        Diagnostic diagnostic;
        diagnostic.code = code;
        diagnostic.message = std::move(message);
        diagnostic.primary.span = span;
        scratch_diagnostic_ = std::move(diagnostic);
        return &scratch_diagnostic_;
    }

    // Reports `diagnostic` once the caller has finished decorating it.
    void emit(Diagnostic* diagnostic) {
        if (diagnostic != nullptr) {
            sink_.report(std::move(*diagnostic));
        }
    }

    void error_here(std::string message, const char* code = codes::unexpected_token) {
        emit(error_at(peek().span, std::move(message), code));
    }

    void error_expected(std::string_view what) {
        error_here("expected " + std::string(what) + ", found " + describe(peek(), text(peek())));
    }

    bool expect(K kind) {
        if (accept(kind)) {
            return true;
        }
        error_expected("`" + std::string(token_kind_name(kind)) + "`");
        return false;
    }

    Name expect_identifier(std::string_view what) {
        const Token token = peek();
        if (token.kind == K::Identifier) {
            advance();
            return {text(token), token.span};
        }
        if (token.kind == K::ReservedWord) {
            report_reserved(token);
            advance();
            return {text(token), token.span};
        }
        error_expected(what);
        return {{}, {ast_.file, token.span.begin, token.span.begin}};
    }

    void report_reserved(const Token& token) {
        Diagnostic* diagnostic =
            error_at(token.span,
                     "`" + std::string(text(token)) + "` is reserved and cannot be used as a name",
                     codes::reserved_identifier);
        if (diagnostic != nullptr) {
            diagnostic->notes.emplace_back("this word is reserved for future language features");
        }
        emit(diagnostic);
    }

    void unsupported(SourceSpan span, std::string message) {
        emit(error_at(span, std::move(message), codes::unsupported_syntax));
    }

    // ------------------------------------------------------------- speculation

    // Runs `body` without reporting errors. When it fails, the token position
    // and every node it created are rolled back.
    template <typename Body>
    bool attempt(Body&& body) {
        const std::size_t saved_pos = pos_;
        const std::uint32_t saved_end = last_end_;
        const Ast::Mark mark = ast_.mark();
        const bool saved_failed = speculation_failed_;
        const bool saved_bailing = is_bailing_;

        speculation_failed_ = false;
        ++speculation_depth_;
        body();
        --speculation_depth_;
        const bool succeeded = !speculation_failed_;
        speculation_failed_ = saved_failed;

        if (!succeeded) {
            is_bailing_ = saved_bailing;
            pos_ = saved_pos;
            last_end_ = saved_end;
            ast_.rewind(mark);
        }
        return succeeded;
    }

    void fail_speculation() { speculation_failed_ = true; }
    // True while list and chain loops should wind down: either a speculative
    // parse has already failed, or the nesting limit was hit and the current
    // statement or item is being abandoned.
    bool should_stop() const {
        return is_bailing_ || (speculation_depth_ != 0 && speculation_failed_);
    }

    // Guards recursive productions against stack exhaustion on hostile input.
    class DepthGuard {
    public:
        explicit DepthGuard(Parser& parser) : parser_(parser) { ++parser_.depth_; }
        ~DepthGuard() { --parser_.depth_; }
        DepthGuard(const DepthGuard&) = delete;
        DepthGuard& operator=(const DepthGuard&) = delete;

        bool exceeded() const {
            if (parser_.depth_ <= max_nesting_depth) {
                return false;
            }
            parser_.error_here("nesting is too deep");
            parser_.is_bailing_ = true;
            return true;
        }

    private:
        Parser& parser_;
    };

    // Operator and suffix chains build left-deep trees without recursing, so
    // each link counts toward the nesting limit to bound tree depth for every
    // later recursive consumer.
    class ChainGuard {
    public:
        explicit ChainGuard(Parser& parser) : parser_(parser) {}
        ~ChainGuard() { parser_.depth_ -= links_; }
        ChainGuard(const ChainGuard&) = delete;
        ChainGuard& operator=(const ChainGuard&) = delete;

        // Returns false when another link would exceed the limit.
        bool extend() {
            if (parser_.depth_ >= max_nesting_depth) {
                parser_.error_here("expression is too long or too deeply nested");
                parser_.is_bailing_ = true;
                return false;
            }
            ++parser_.depth_;
            ++links_;
            return true;
        }

    private:
        Parser& parser_;
        std::uint32_t links_ = 0;
    };

    // ---------------------------------------------------------------- recovery

    // Skips tokens until `stop` accepts the current one outside any brackets
    // opened while skipping.
    template <typename Stop>
    void skip_until(Stop stop) {
        std::uint32_t depth = 0;
        while (!at(K::Eof)) {
            const K kind = peek().kind;
            if (depth == 0 && stop(kind)) {
                return;
            }
            if (kind == K::LParen || kind == K::LBracket || kind == K::LBrace) {
                ++depth;
            } else if (kind == K::RParen || kind == K::RBracket || kind == K::RBrace) {
                if (depth == 0) {
                    return;
                }
                --depth;
            }
            advance();
        }
    }

    void skip_to_item_start() {
        while (!starts_item(peek().kind) && !at(K::RBrace) && !at(K::Eof)) {
            skip_until([](K kind) { return starts_item(kind); });
            if (at(K::RParen) || at(K::RBracket)) {
                advance(); // stray closer left behind by the damaged item
            }
        }
    }

    bool at_statement_start() const {
        switch (peek().kind) {
        case K::KwLet:
        case K::KwVar:
        case K::KwReturn:
        case K::KwStatic:
        case K::RBrace:
        case K::Eof:
            return true;
        case K::Identifier:
            return peek(1).kind == K::Equal;
        default:
            return starts_item(peek().kind);
        }
    }

    void skip_to_statement_start() {
        while (!at_statement_start()) {
            skip_until([this](K) { return at_statement_start(); });
            if (at(K::RParen) || at(K::RBracket)) {
                advance(); // stray closer left behind by the damaged statement
            }
        }
    }

    // ------------------------------------------------------------------ module

    std::vector<Name> parse_dotted_path(std::string_view what) {
        std::vector<Name> path;
        path.push_back(expect_identifier(what));
        while (at(K::Dot)) {
            advance();
            path.push_back(expect_identifier(what));
        }
        return path;
    }

    void parse_module_decl() {
        if (!at(K::KwModule)) {
            Diagnostic* diagnostic = error_at({ast_.file, here(), here()},
                                              "missing `module` declaration",
                                              codes::missing_module_decl);
            if (diagnostic != nullptr) {
                diagnostic->help.emplace_back(
                    "every source file starts with a declaration such as `module my.module`");
            }
            emit(diagnostic);
            // Allow the first real token to report its own error if needed.
            has_error_position_ = false;
            return;
        }
        const std::uint32_t begin = here();
        advance();
        ast_.module_path = parse_dotted_path("module name");
        ast_.module_span = span_from(begin);
    }

    void parse_use() {
        const std::uint32_t begin = here();
        advance();
        UseDecl use{};

        if (at(K::KwSelf) || at(K::KwSuper)) {
            unsupported(peek().span,
                        "`" + std::string(text(peek())) + "` imports are not supported yet");
            use.path.push_back({text(peek()), peek().span});
            advance();
        } else if (at(K::KwCrate)) {
            use.path.push_back({text(peek()), peek().span});
            advance();
        } else {
            use.path.push_back(expect_identifier("import path"));
        }
        while (accept(K::Dot)) {
            use.path.push_back(expect_identifier("module name"));
        }

        if (accept(K::ColonColon)) {
            if (accept(K::LBrace)) {
                use.is_braced = true;
                while (!at(K::RBrace) && !at(K::Eof)) {
                    use.names.push_back(parse_import_name());
                    if (!accept(K::Comma)) {
                        break;
                    }
                }
                expect(K::RBrace);
            } else {
                use.names.push_back({expect_identifier("imported name"), {}});
            }
        }
        use.span = span_from(begin);
        ast_.uses.push_back(std::move(use));
    }

    ImportName parse_import_name() {
        ImportName import{expect_identifier("imported name"), {}};
        if (accept(K::KwAs)) {
            import.alias = expect_identifier("alias");
        }
        return import;
    }

    // ------------------------------------------------------------------- items

    ItemId error_item(std::uint32_t begin, bool is_pub) {
        return ast_.add(Item{span_from(begin), is_pub, ErrorItem{}});
    }

    ItemId parse_item(bool in_block) {
        const DepthGuard guard(*this);
        const std::uint32_t begin = here();
        const std::size_t errors_before = error_count_;
        if (guard.exceeded()) {
            skip_to_item_start();
            return error_item(begin, false);
        }

        const bool is_pub = accept(K::KwPub);
        ItemData data = ErrorItem{};
        switch (peek().kind) {
        case K::KwConst:
            data = parse_const();
            break;
        case K::KwType:
            data = parse_type_alias();
            break;
        case K::KwStruct:
            data = parse_struct();
            break;
        case K::KwEnum:
            data = parse_enum();
            break;
        case K::KwFn:
            data = parse_function(FunctionKind::Fn);
            break;
        case K::KwOp:
            data = parse_function(FunctionKind::Op);
            break;
        case K::KwEntry:
            data = parse_function(FunctionKind::Entry);
            break;
        case K::KwBlock:
            data = parse_block();
            break;
        case K::KwParam:
        case K::KwBuffer:
        case K::KwSub:
            if (!in_block) {
                error_here("`" + std::string(text(peek())) +
                           "` declarations are only allowed inside a `block`");
            } else if (is_pub) {
                error_here("`" + std::string(text(peek())) + "` declarations cannot be `pub`");
            }
            data = parse_member();
            break;
        case K::KwState:
            unsupported(peek().span, "`state` declarations are reserved for a future version");
            advance();
            break;
        case K::KwExtern:
            unsupported(peek().span, "`extern` declarations are reserved for a future version");
            advance();
            if (at(K::KwFn) || at(K::KwOp)) {
                advance(); // skip the whole declaration, not just the keyword
            }
            break;
        default:
            error_expected(in_block ? "a block member" : "an item");
            break;
        }

        if (error_count_ != errors_before && !starts_item(peek().kind)) {
            skip_to_item_start();
        }
        is_bailing_ = false;
        return ast_.add(Item{span_from(begin), is_pub, std::move(data)});
    }

    ConstDecl parse_const() {
        advance();
        ConstDecl decl{expect_identifier("constant name"), no_id, no_id};
        if (accept(K::Colon)) {
            decl.type = parse_type();
        }
        expect(K::Equal);
        decl.value = parse_expr();
        return decl;
    }

    TypeAliasDecl parse_type_alias() {
        advance();
        TypeAliasDecl decl{expect_identifier("type name"), {}, no_id};
        decl.generics = parse_generic_params(decl.generics_span);
        expect(K::Equal);
        decl.type = parse_type();
        return decl;
    }

    StructDecl parse_struct() {
        advance();
        StructDecl decl{expect_identifier("struct name"), {}, {}};
        decl.generics = parse_generic_params(decl.generics_span);
        if (!expect(K::LBrace)) {
            return decl;
        }
        while (!at(K::RBrace) && !at(K::Eof) && !starts_item(peek().kind)) {
            const std::size_t before = pos_;
            if (at(K::Comma)) {
                error_here("struct fields are separated by newlines, not commas");
                advance();
                continue;
            }
            const std::uint32_t begin = here();
            FieldDecl field{{}, expect_identifier("field name"), no_id};
            expect(K::Colon);
            field.type = parse_type();
            field.span = span_from(begin);
            decl.fields.push_back(field);
            if (pos_ == before) {
                advance();
            }
        }
        expect(K::RBrace);
        return decl;
    }

    EnumDecl parse_enum() {
        advance();
        EnumDecl decl{expect_identifier("enum name"), {}, {}};
        decl.generics = parse_generic_params(decl.generics_span);
        if (!expect(K::LBrace)) {
            return decl;
        }
        while (!at(K::RBrace) && !at(K::Eof)) {
            if (at(K::Identifier) && peek(1).kind == K::LParen) {
                unsupported(peek(1).span, "payload-carrying enum variants are not supported yet");
            }
            decl.variants.push_back(expect_identifier("variant name"));
            if (!accept(K::Comma)) {
                break;
            }
        }
        expect(K::RBrace);
        return decl;
    }

    FunctionDecl parse_function(FunctionKind kind) {
        const std::string_view keyword = text(peek());
        advance();
        FunctionDecl decl{kind, expect_identifier("function name"), {}, {}, no_id, {}, {}};
        decl.generics = parse_generic_params(decl.generics_span);
        decl.parameters = parse_parameters(decl.parameters_span);
        if (accept(K::Arrow)) {
            decl.return_type = parse_type();
        } else if (kind == FunctionKind::Op) {
            error_expected("`->` and a result type; an `" + std::string(keyword) +
                           "` must declare its result");
        }
        if (at(K::KwWhere)) {
            decl.constraints = parse_where();
        }
        decl.body = parse_body();
        return decl;
    }

    BlockDecl parse_block() {
        advance();
        BlockDecl decl{expect_identifier("block name"), {}, {}};
        decl.generics = parse_generic_params(decl.generics_span);
        if (!expect(K::LBrace)) {
            return decl;
        }
        while (!at(K::RBrace) && !at(K::Eof)) {
            const std::size_t before = pos_;
            if (at(K::KwUse) || at(K::KwModule)) {
                break; // a missing `}`; let the file level continue from here
            }
            decl.members.push_back(parse_item(true));
            if (pos_ == before) {
                advance();
            }
        }
        expect(K::RBrace);
        return decl;
    }

    MemberDecl parse_member() {
        const K keyword = peek().kind;
        const MemberKind kind = keyword == K::KwParam    ? MemberKind::Param
                                : keyword == K::KwBuffer ? MemberKind::Buffer
                                                         : MemberKind::Sub;
        advance();
        MemberDecl decl{kind, expect_identifier("member name"), no_id, no_id};
        expect(K::Colon);
        decl.type = parse_type();
        if (at(K::Equal)) {
            if (kind != MemberKind::Param) {
                error_here("only `param` declarations may have a default");
            }
            advance();
            decl.default_value = parse_expr();
        }
        return decl;
    }

    std::vector<GenericParam> parse_generic_params(SourceSpan& list_span) {
        std::vector<GenericParam> params;
        const std::uint32_t list_begin = here();
        if (!accept(K::Less)) {
            return params;
        }
        while (!at(K::Greater) && !at(K::Eof) && !should_stop()) {
            const std::uint32_t begin = here();
            GenericParam param{};
            param.is_pack = accept(K::Star);
            param.name = expect_identifier("generic parameter name");
            if (expect(K::Colon)) {
                param.constraint_name = expect_identifier("generic constraint");
                param.constraint = constraint_kind(param.constraint_name.text);
                if (param.constraint == ConstraintKind::Unknown &&
                    !param.constraint_name.text.empty()) {
                    Diagnostic* diagnostic =
                        error_at(param.constraint_name.span,
                                 "unknown generic constraint `" +
                                     std::string(param.constraint_name.text) + "`",
                                 codes::unexpected_token);
                    if (diagnostic != nullptr) {
                        diagnostic->help.emplace_back(
                            "expected one of: Dim, Shape, DType, Numeric, Integer, Float");
                    }
                    emit(diagnostic);
                }
            }
            if (accept(K::Equal)) {
                param.default_value = parse_type_or_expr();
            }
            param.span = span_from(begin);
            params.push_back(param);
            if (!accept(K::Comma)) {
                break;
            }
        }
        expect(K::Greater);
        list_span = span_from(list_begin);
        return params;
    }

    std::vector<Parameter> parse_parameters(SourceSpan& list_span) {
        std::vector<Parameter> params;
        const std::uint32_t list_begin = here();
        if (!expect(K::LParen)) {
            return params;
        }
        while (!at(K::RParen) && !at(K::Eof)) {
            const std::uint32_t begin = here();
            Parameter param{{}, expect_identifier("parameter name"), no_id, no_id};
            if (expect(K::Colon)) {
                param.type = parse_type();
            }
            if (accept(K::Equal)) {
                param.default_value = parse_expr();
            }
            param.span = span_from(begin);
            params.push_back(param);
            if (!accept(K::Comma)) {
                break;
            }
        }
        expect(K::RParen);
        list_span = span_from(list_begin);
        return params;
    }

    std::vector<ExprId> parse_where() {
        advance();
        std::vector<ExprId> constraints;
        while (!at(K::LBrace) && !at(K::Eof)) {
            const ExprId constraint = parse_expr();
            const Expr& node = ast_.expr(constraint);
            const auto* binary = std::get_if<BinaryExpr>(&node.data);
            if ((binary == nullptr || !is_comparison(binary->op)) &&
                !std::holds_alternative<ErrorExpr>(node.data)) {
                emit(error_at(node.span,
                              "a `where` constraint must be a comparison such as `H % N == 0`",
                              codes::unexpected_token));
            }
            constraints.push_back(constraint);
            if (!accept(K::Comma)) {
                break;
            }
        }
        return constraints;
    }

    // -------------------------------------------------------------- statements

    std::vector<StmtId> parse_body() {
        std::vector<StmtId> body;
        if (!expect(K::LBrace)) {
            return body;
        }
        while (!at(K::RBrace) && !at(K::Eof)) {
            if (starts_item(peek().kind)) {
                break; // a missing `}`; the enclosing scope continues from here
            }
            const std::size_t before = pos_;
            body.push_back(parse_statement());
            if (pos_ == before) {
                advance();
            }
        }
        expect(K::RBrace);
        return body;
    }

    StmtId parse_statement() {
        const DepthGuard guard(*this);
        const std::uint32_t begin = here();
        const std::size_t errors_before = error_count_;
        StmtData data = ErrorStmt{};

        if (guard.exceeded()) {
            // handled by the recovery below
        } else if (at(K::KwLet)) {
            data = parse_let();
        } else if (at(K::KwVar)) {
            advance();
            VarStmt var{expect_identifier("variable name"), no_id, no_id};
            if (accept(K::Colon)) {
                var.type = parse_type();
            }
            expect(K::Equal);
            var.value = parse_expr();
            data = var;
        } else if (at(K::KwReturn)) {
            advance();
            ReturnStmt ret{no_id};
            const bool is_next_assignment = at(K::Identifier) && peek(1).kind == K::Equal;
            if (starts_expression(peek().kind) && !is_next_assignment) {
                ret.value = parse_expr();
            }
            data = ret;
        } else if (at(K::KwStatic)) {
            data = parse_static_for();
        } else if (at(K::Identifier) && peek(1).kind == K::Equal) {
            AssignStmt assign{{text(peek()), peek().span}, no_id};
            advance();
            advance();
            assign.value = parse_expr();
            data = assign;
        } else if (at(K::KwWhile) || at(K::KwFor)) {
            unsupported(peek().span,
                        "runtime loops are reserved for a future version; use `static for` for "
                        "structural iteration");
            advance();
        } else {
            Diagnostic* diagnostic =
                error_at(peek().span,
                         "expected a statement, found " + describe(peek(), text(peek())),
                         codes::unexpected_token);
            if (diagnostic != nullptr && starts_expression(peek().kind)) {
                diagnostic->notes.emplace_back(
                    "Linnet has no expression statements; bind the value with `let` or "
                    "`return` it");
            }
            emit(diagnostic);
        }

        if (error_count_ != errors_before) {
            skip_to_statement_start();
        }
        is_bailing_ = false;
        return ast_.add(Stmt{span_from(begin), std::move(data)});
    }

    StmtData parse_let() {
        advance();
        if (at(K::Identifier) && peek(1).kind == K::LBracket) {
            ComprehensionStmt stmt{{text(peek()), peek().span}, {}, no_id};
            advance();
            advance();
            while (!at(K::RBracket) && !at(K::Eof)) {
                const bool is_pack = accept(K::Star);
                stmt.outputs.push_back({expect_identifier("output index"), is_pack});
                if (!accept(K::Comma)) {
                    break;
                }
            }
            expect(K::RBracket);
            expect(K::Equal);
            stmt.value = parse_expr();
            return stmt;
        }

        LetStmt stmt{parse_pattern(), no_id, no_id};
        if (accept(K::Colon)) {
            stmt.type = parse_type();
        }
        expect(K::Equal);
        stmt.value = parse_expr();
        return stmt;
    }

    StmtData parse_static_for() {
        advance();
        if (!expect(K::KwFor)) {
            return ErrorStmt{};
        }
        StaticForStmt stmt{parse_pattern(), no_id, {}};
        expect(K::KwIn);
        stmt.iterable = parse_expr();
        if (at(K::DotDot)) {
            unsupported(peek().span, "static integer ranges are not supported yet");
            advance();
            parse_expr();
        }
        stmt.body = parse_body();
        return stmt;
    }

    // ---------------------------------------------------------------- patterns

    // Match arms have no separator, so an arm pattern cannot begin with `(`:
    // it would read as a call on the previous arm's value.
    PatternId parse_pattern(bool is_arm = false) {
        const DepthGuard guard(*this);
        const std::uint32_t begin = here();
        PatternData data = ErrorPattern{};

        if (guard.exceeded()) {
            // leave the error pattern
        } else if (is_arm && at(K::LParen)) {
            Diagnostic* diagnostic = error_at(peek().span,
                                              "a match arm cannot start with a tuple pattern",
                                              codes::unexpected_token);
            if (diagnostic != nullptr) {
                diagnostic->notes.emplace_back(
                    "`match` works on optional and enum values; arms are `some(...)`, "
                    "`none`, a variant, or a name");
            }
            emit(diagnostic);
        } else if (at(K::Identifier) || at(K::ReservedWord)) {
            data = BindingPattern{expect_identifier("pattern")};
        } else if (accept(K::KwNone)) {
            data = NonePattern{};
        } else if (accept(K::KwSome)) {
            expect(K::LParen);
            const PatternId inner = parse_pattern();
            expect(K::RParen);
            data = SomePattern{inner};
        } else if (accept(K::LParen)) {
            TuplePattern tuple;
            while (!at(K::RParen) && !at(K::Eof) && !should_stop()) {
                tuple.elements.push_back(parse_pattern());
                if (!accept(K::Comma)) {
                    break;
                }
            }
            expect(K::RParen);
            data = std::move(tuple);
        } else {
            error_expected("a pattern");
        }
        return ast_.add(Pattern{span_from(begin), std::move(data)});
    }

    // ------------------------------------------------------------------- types

    TypeId parse_type() {
        const DepthGuard guard(*this);
        const std::uint32_t begin = here();
        if (guard.exceeded()) {
            return ast_.add(Type{span_from(begin), ErrorType{}});
        }
        const TypeId primary = parse_type_primary();
        if (accept(K::Question)) {
            return ast_.add(Type{span_from(begin), OptionalType{primary}});
        }
        return primary;
    }

    TypeId parse_type_primary() {
        const std::uint32_t begin = here();
        TypeData data = ErrorType{};

        if (at(K::Identifier) && text(peek()) == "Tensor" && peek(1).kind == K::LBracket) {
            advance();
            advance();
            TensorType tensor{{}, no_id};
            while (!at(K::Semicolon) && !at(K::RBracket) && !at(K::Eof) && !should_stop()) {
                ShapeElement element;
                if (accept(K::Star)) {
                    element.pack = expect_identifier("shape pack name");
                } else {
                    element.dim = parse_expr();
                }
                tensor.shape.push_back(element);
                if (!accept(K::Comma)) {
                    break;
                }
            }
            if (expect(K::Semicolon)) {
                tensor.dtype = parse_named_type();
            }
            expect(K::RBracket);
            if (tensor.shape.empty()) {
                unsupported(span_from(begin), "rank-zero tensor types are reserved; use a scalar");
            }
            data = std::move(tensor);
        } else if (at(K::Identifier) || at(K::ReservedWord)) {
            return parse_named_type();
        } else if (accept(K::LParen)) {
            TupleType tuple;
            tuple.elements.push_back(parse_type());
            if (!at(K::Comma)) {
                error_expected("`,`; a tuple type has at least two elements");
            }
            while (!should_stop() && accept(K::Comma)) {
                if (at(K::RParen)) {
                    break;
                }
                tuple.elements.push_back(parse_type());
            }
            expect(K::RParen);
            data = std::move(tuple);
        } else if (accept(K::LBracket)) {
            ArrayType array{parse_type(), no_id};
            if (expect(K::Semicolon)) {
                array.length = parse_expr();
            }
            expect(K::RBracket);
            data = array;
        } else {
            error_expected("a type");
        }
        return ast_.add(Type{span_from(begin), std::move(data)});
    }

    TypeId parse_named_type() {
        const std::uint32_t begin = here();
        NamedType named{parse_dotted_path("a type name"), {}};
        if (at(K::Less)) {
            named.args = parse_generic_args();
        }
        return ast_.add(Type{span_from(begin), std::move(named)});
    }

    std::vector<GenericArg> parse_generic_args() {
        std::vector<GenericArg> args;
        expect(K::Less);
        while (!at(K::Greater) && !at(K::Eof) && !should_stop()) {
            args.push_back(parse_type_or_expr());
            if (!accept(K::Comma)) {
                break;
            }
        }
        expect(K::Greater);
        return args;
    }

    // A generic argument is a type when it parses as one and is followed by
    // `,` or `>`. Otherwise it is an arithmetic expression such as `H / N` or
    // `4096`; comparisons and logical operators need parentheses here, which
    // keeps `a < b && c > (d)` an ordinary comparison chain.
    GenericArg parse_type_or_expr() {
        GenericArg arg;
        const bool is_type = attempt([&] {
            arg.type = parse_type();
            if (!at(K::Comma) && !at(K::Greater)) {
                fail_speculation();
            }
        });
        if (!is_type) {
            arg.type = no_id;
            arg.expr = parse_binary(additive_precedence);
        }
        return arg;
    }

    // ------------------------------------------------------------- expressions

    ExprId add_expr(std::uint32_t begin, ExprData data) {
        return ast_.add(Expr{span_from(begin), std::move(data)});
    }

    ExprId parse_expr() {
        const DepthGuard guard(*this);
        const std::uint32_t begin = here();
        if (guard.exceeded()) {
            return add_expr(begin, ErrorExpr{});
        }
        if (at(K::KwIf)) {
            return parse_if();
        }
        if (at(K::KwMatch)) {
            return parse_match();
        }
        return parse_binary(1);
    }

    ExprId parse_braced_expr() {
        expect(K::LBrace);
        const ExprId value = parse_expr();
        expect(K::RBrace);
        return value;
    }

    ExprId parse_if() {
        const std::uint32_t begin = here();
        advance();
        IfExpr node{parse_expr(), no_id, no_id};
        node.then_value = parse_braced_expr();
        if (expect(K::KwElse)) {
            if (at(K::KwIf)) {
                error_here("`else if` is not supported; nest the `if` inside `else { ... }`");
            }
            node.else_value = parse_braced_expr();
        } else {
            node.else_value = add_expr(here(), ErrorExpr{});
        }
        return add_expr(begin, node);
    }

    ExprId parse_match() {
        const std::uint32_t begin = here();
        advance();
        MatchExpr node{parse_expr(), {}};
        if (expect(K::LBrace)) {
            while (!at(K::RBrace) && !at(K::Eof) && !should_stop()) {
                const std::size_t before = pos_;
                const std::size_t errors_before = error_count_;
                MatchArm arm{parse_pattern(true), no_id};
                arm.value = expect(K::FatArrow) ? parse_expr() : add_expr(here(), ErrorExpr{});
                node.arms.push_back(arm);
                if (error_count_ != errors_before || pos_ == before) {
                    break;
                }
            }
            if (node.arms.empty()) {
                error_expected("a match arm");
            }
            expect(K::RBrace);
        }
        return add_expr(begin, std::move(node));
    }

    const BinaryOperator* binary_operator_here() const {
        for (const BinaryOperator& candidate : binary_operators) {
            if (at(candidate.token)) {
                return &candidate;
            }
        }
        return nullptr;
    }

    ExprId parse_binary(int min_precedence) {
        const std::uint32_t begin = here();
        ExprId lhs = parse_unary();
        ChainGuard chain(*this);
        while (!should_stop()) {
            const BinaryOperator* op = binary_operator_here();
            if (op == nullptr || op->precedence < min_precedence) {
                break;
            }
            if (!chain.extend()) {
                break;
            }
            advance();
            const ExprId rhs = parse_binary(op->precedence + 1);
            lhs = add_expr(begin, BinaryExpr{op->op, lhs, rhs});
        }
        return lhs;
    }

    ExprId parse_unary() {
        const std::uint32_t begin = here();
        std::optional<UnaryOp> op;
        if (at(K::Bang)) {
            op = UnaryOp::Not;
        } else if (at(K::Minus)) {
            op = UnaryOp::Negate;
        } else if (at(K::Plus)) {
            op = UnaryOp::Plus;
        }
        if (!op) {
            return parse_postfix();
        }
        advance();
        const ExprId operand = parse_postfix();
        return add_expr(begin, UnaryExpr{*op, operand});
    }

    ExprId parse_postfix() {
        const std::uint32_t begin = here();
        ExprId expr = parse_primary();
        ChainGuard chain(*this);
        while (!should_stop()) {
            const bool is_suffix = at(K::LParen) || at(K::LBracket) || at(K::Dot) || at(K::Less);
            if (is_suffix && !chain.extend()) {
                break;
            }
            if (at(K::LParen)) {
                expr = add_expr(begin, CallExpr{expr, {}, parse_arguments()});
            } else if (at(K::LBracket)) {
                expr = add_expr(begin, IndexExpr{expr, parse_index_components()});
            } else if (at(K::Dot)) {
                advance();
                expr = add_expr(begin, MemberExpr{expr, expect_identifier("a member name")});
            } else if (at(K::Less) && is_path(expr) && !failed_generic_call_[pos_]) {
                // `name<...>(` is a generic call; anything else is a comparison.
                std::vector<GenericArg> generic_args;
                const bool is_generic_call = attempt([&] {
                    generic_args = parse_generic_args();
                    if (!at(K::LParen)) {
                        fail_speculation();
                    }
                });
                if (!is_generic_call) {
                    failed_generic_call_[pos_] = true;
                    break;
                }
                expr = add_expr(begin, CallExpr{expr, std::move(generic_args), parse_arguments()});
            } else {
                break;
            }
        }
        return expr;
    }

    bool is_path(ExprId id) const {
        const ExprData& data = ast_.expr(id).data;
        return std::holds_alternative<NameExpr>(data) || std::holds_alternative<MemberExpr>(data);
    }

    std::vector<Argument> parse_arguments() {
        std::vector<Argument> args;
        expect(K::LParen);
        while (!at(K::RParen) && !at(K::Eof) && !should_stop()) {
            Argument arg{{}, no_id};
            if (at(K::Identifier) && peek(1).kind == K::Equal) {
                arg.keyword = {text(peek()), peek().span};
                advance();
                advance();
            }
            arg.value = parse_expr();
            args.push_back(arg);
            if (!accept(K::Comma)) {
                break;
            }
        }
        expect(K::RParen);
        return args;
    }

    bool at_slice_boundary() const {
        return at(K::Comma) || at(K::RBracket) || at(K::Colon) || at(K::ColonColon) || at(K::Eof);
    }

    std::vector<IndexComponent> parse_index_components() {
        std::vector<IndexComponent> components;
        expect(K::LBracket);
        while (!at(K::RBracket) && !at(K::Eof) && !should_stop()) {
            const std::uint32_t begin = here();
            IndexComponent component;
            if (accept(K::Ellipsis)) {
                component.kind = IndexKind::Ellipsis;
            } else if (accept(K::Star)) {
                component.kind = IndexKind::Pack;
                component.pack = expect_identifier("an index pack name");
            } else {
                if (!at(K::Colon) && !at(K::ColonColon)) {
                    component.value = parse_expr();
                }
                if (at(K::Colon) || at(K::ColonColon)) {
                    component.kind = IndexKind::Slice;
                    component.start = std::exchange(component.value, no_id);
                    // `::` is a slice with an empty stop.
                    const bool has_stop_slot = at(K::Colon);
                    advance();
                    if (has_stop_slot) {
                        if (!at_slice_boundary()) {
                            component.stop = parse_expr();
                        }
                        if (accept(K::Colon) && !at_slice_boundary()) {
                            component.step = parse_expr();
                        }
                    } else if (!at_slice_boundary()) {
                        component.step = parse_expr();
                    }
                }
            }
            component.span = span_from(begin);
            components.push_back(component);
            if (!accept(K::Comma)) {
                break;
            }
        }
        expect(K::RBracket);
        return components;
    }

    ExprId parse_reduction(ReductionKind kind, TypeId accumulator, std::uint32_t begin) {
        ReductionExpr node{kind, accumulator, {}, no_id};
        expect(K::LBracket);
        while (!at(K::RBracket) && !at(K::Eof)) {
            node.indices.push_back(expect_identifier("a reduction index"));
            if (!accept(K::Comma)) {
                break;
            }
        }
        if (node.indices.empty()) {
            error_expected("a reduction index");
        }
        expect(K::RBracket);
        node.body = parse_expr();
        return add_expr(begin, std::move(node));
    }

    ExprId parse_primary() {
        const std::uint32_t begin = here();
        const Token token = peek();
        switch (token.kind) {
        case K::Integer:
            advance();
            return add_expr(begin, LiteralExpr{LiteralKind::Integer});
        case K::Float:
            advance();
            return add_expr(begin, LiteralExpr{LiteralKind::Float});
        case K::String:
            advance();
            return add_expr(begin, LiteralExpr{LiteralKind::String});
        case K::KwTrue:
            advance();
            return add_expr(begin, LiteralExpr{LiteralKind::True});
        case K::KwFalse:
            advance();
            return add_expr(begin, LiteralExpr{LiteralKind::False});
        case K::KwNone:
            advance();
            return add_expr(begin, NoneExpr{});
        case K::KwSome: {
            advance();
            expect(K::LParen);
            const ExprId value = parse_expr();
            expect(K::RParen);
            return add_expr(begin, SomeExpr{value});
        }
        case K::ReservedWord:
            report_reserved(token);
            advance();
            return add_expr(begin, NameExpr{{text(token), token.span}});
        case K::Identifier:
            return parse_name_or_reduction();
        case K::LParen:
            return parse_paren_or_tuple();
        case K::LBracket: {
            advance();
            ShapeExpr shape;
            while (!at(K::RBracket) && !at(K::Eof) && !should_stop()) {
                shape.dims.push_back(parse_expr());
                if (!accept(K::Comma)) {
                    break;
                }
            }
            expect(K::RBracket);
            return add_expr(begin, std::move(shape));
        }
        default:
            error_expected("an expression");
            return add_expr(begin, ErrorExpr{});
        }
    }

    ExprId parse_name_or_reduction() {
        const std::uint32_t begin = here();
        const Token token = peek();
        if (const std::optional<ReductionKind> kind = reduction_kind(text(token))) {
            if (peek(1).kind == K::LBracket) {
                advance();
                return parse_reduction(*kind, no_id, begin);
            }
            if (peek(1).kind == K::Less) {
                TypeId accumulator = no_id;
                const bool has_accumulator = attempt([&] {
                    advance();
                    advance();
                    accumulator = parse_type();
                    expect(K::Greater);
                    if (!at(K::LBracket)) {
                        fail_speculation();
                    }
                });
                if (has_accumulator) {
                    return parse_reduction(*kind, accumulator, begin);
                }
            }
        }
        advance();
        return add_expr(begin, NameExpr{{text(token), token.span}});
    }

    ExprId parse_paren_or_tuple() {
        const std::uint32_t begin = here();
        advance();
        const ExprId first = parse_expr();
        if (!at(K::Comma)) {
            expect(K::RParen);
            return add_expr(begin, ParenExpr{first});
        }
        TupleExpr tuple{{first}};
        while (!should_stop() && accept(K::Comma)) {
            if (at(K::RParen)) {
                break;
            }
            tuple.elements.push_back(parse_expr());
        }
        expect(K::RParen);
        return add_expr(begin, std::move(tuple));
    }

    const SourceManager& sources_;
    DiagnosticSink& sink_;
    std::vector<Token> tokens_;
    Ast ast_;
    Diagnostic scratch_diagnostic_;

    std::size_t pos_ = 0;
    std::uint32_t last_end_ = 0;
    std::uint32_t depth_ = 0;

    std::size_t error_count_ = 0;
    std::size_t error_position_ = 0;
    bool has_error_position_ = false;

    // Positions of `<` already shown not to start a generic call; keeps
    // speculation linear on inputs such as `a<b<c<d...`.
    std::vector<bool> failed_generic_call_;
    bool is_bailing_ = false;

    std::uint32_t speculation_depth_ = 0;
    bool speculation_failed_ = false;
};

} // namespace

ast::Ast parse(const SourceManager& sources, FileId file, DiagnosticSink& sink) {
    return Parser(sources, file, sink).run();
}

} // namespace linnet
