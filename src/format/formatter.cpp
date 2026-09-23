#include "linnet/format/formatter.hpp"

#include <algorithm>
#include <functional>
#include <string_view>
#include <utility>
#include <vector>

namespace linnet::format {

namespace {

using namespace ast;

template <typename... Visitors>
struct Overloaded : Visitors... {
    using Visitors::operator()...;
};
template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

int precedence(BinaryOp op) {
    switch (op) {
    case BinaryOp::Or:
        return 1;
    case BinaryOp::And:
        return 2;
    case BinaryOp::Equal:
    case BinaryOp::NotEqual:
        return 3;
    case BinaryOp::Less:
    case BinaryOp::LessEqual:
    case BinaryOp::Greater:
    case BinaryOp::GreaterEqual:
        return 4;
    case BinaryOp::Add:
    case BinaryOp::Subtract:
        return 5;
    case BinaryOp::Multiply:
    case BinaryOp::Divide:
    case BinaryOp::Remainder:
        return 6;
    }
    return 0;
}

// One entry of a line sequence or bracketed list. `doc` runs lazily and in
// source order so that comments are claimed by the right element.
struct Element {
    std::uint32_t begin;
    std::uint32_t end;
    std::function<DocId()> doc;
    bool blank_before = false; // force a blank line between this and the previous element
};

class Formatter {
public:
    Formatter(const Ast& ast, const SourceManager& sources)
        : ast_(ast), sources_(sources), text_(sources.contents(ast.file)) {}

    std::string run(const LayoutOptions& options) {
        const DocId file = sequence(file_elements(), static_cast<std::uint32_t>(text_.size()) + 1);
        std::string out = print(b_, file, options);
        if (!out.empty()) {
            out += '\n';
        }
        return out;
    }

private:
    // ---------------------------------------------------------------- comments

    bool has_comment_before(std::uint32_t position) const {
        return next_comment_ < ast_.comments.size() &&
               ast_.comments[next_comment_].span.begin < position;
    }

    const Comment& take_comment() { return ast_.comments[next_comment_++]; }

    std::size_t newlines_between(std::uint32_t begin, std::uint32_t end) const {
        if (begin >= end || end > text_.size()) {
            return 0;
        }
        const std::string_view gap = text_.substr(begin, end - begin);
        return static_cast<std::size_t>(std::count(gap.begin(), gap.end(), '\n'));
    }

    // Continuation lines of a block comment are reproduced verbatim so that
    // their internal layout survives and formatting stays idempotent.
    DocId comment_doc(const Comment& comment) {
        std::string_view rest = sources_.text(comment.span);
        std::vector<DocId> parts;
        while (true) {
            const std::size_t newline = rest.find('\n');
            std::string_view line = rest.substr(0, newline);
            while (!line.empty() &&
                   (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
                line.remove_suffix(1);
            }
            parts.push_back(b_.text(line));
            if (newline == std::string_view::npos) {
                break;
            }
            parts.push_back(b_.literal_line());
            rest.remove_prefix(newline + 1);
        }
        return b_.concat(parts);
    }

    // Same-line comments that follow position `end`, up to `limit`.
    void
    append_trailing_comments(std::vector<DocId>& parts, std::uint32_t& end, std::uint32_t limit) {
        while (has_comment_before(limit)) {
            const Comment& comment = ast_.comments[next_comment_];
            if (comment.span.begin < end || newlines_between(end, comment.span.begin) != 0) {
                return;
            }
            ++next_comment_;
            parts.push_back(b_.text(" "));
            parts.push_back(comment_doc(comment));
            parts.push_back(b_.break_parent());
            end = comment.span.end;
        }
    }

    // --------------------------------------------------------------- sequences

    // Elements on separate lines, keeping comments and at most one blank line
    // wherever the source had any. `close` is the position that ends the
    // region; comments before it are kept inside.
    DocId sequence(const std::vector<Element>& elements, std::uint32_t close) {
        std::vector<DocId> parts;
        std::uint32_t previous_end = 0;
        const auto begin_line = [&](std::uint32_t begin, bool force_blank) {
            if (!parts.empty()) {
                parts.push_back(b_.hard_line());
                if (force_blank || newlines_between(previous_end, begin) >= 2) {
                    parts.push_back(b_.hard_line());
                }
            }
        };
        const auto own_line_comments = [&](std::uint32_t limit, bool force_blank) {
            while (has_comment_before(limit)) {
                const Comment& comment = take_comment();
                begin_line(comment.span.begin, force_blank);
                force_blank = false;
                parts.push_back(comment_doc(comment));
                previous_end = std::max(previous_end, comment.span.end);
            }
            return force_blank;
        };

        for (std::size_t i = 0; i < elements.size(); ++i) {
            const Element& element = elements[i];
            const bool force_blank = own_line_comments(element.begin, element.blank_before);
            begin_line(element.begin, force_blank);
            parts.push_back(element.doc());
            previous_end = element.end;

            // Comments written inside the element at untracked positions.
            const bool had_leftovers = has_comment_before(element.end);
            own_line_comments(element.end, false);
            if (!had_leftovers) {
                const std::uint32_t limit = i + 1 < elements.size() ? elements[i + 1].begin : close;
                append_trailing_comments(parts, previous_end, limit);
            }
        }
        own_line_comments(close, false);
        return b_.concat(parts);
    }

    DocId braced(const std::vector<Element>& elements, std::uint32_t close) {
        const DocId body = sequence(elements, close);
        if (body == b_.nil()) {
            return b_.text("{}");
        }
        return b_.concat({b_.text("{"),
                          b_.indent(b_.concat({b_.hard_line(), body})),
                          b_.hard_line(),
                          b_.text("}")});
    }

    // Comma-separated elements between brackets: one line when it fits,
    // otherwise one element per line with a trailing comma.
    DocId list(std::string_view open,
               std::string_view close_text,
               const std::vector<Element>& elements,
               std::uint32_t close,
               bool force_trailing_comma = false) {
        std::vector<DocId> inner;
        for (std::size_t i = 0; i < elements.size(); ++i) {
            const Element& element = elements[i];
            while (has_comment_before(element.begin)) {
                inner.push_back(comment_doc(take_comment()));
                inner.push_back(b_.hard_line());
            }
            inner.push_back(element.doc());
            const bool is_last = i + 1 == elements.size();
            if (!is_last || force_trailing_comma) {
                inner.push_back(b_.text(","));
            } else {
                inner.push_back(b_.if_break(b_.text(","), b_.nil()));
            }
            std::uint32_t end = element.end;
            const bool had_leftovers = has_comment_before(end);
            while (has_comment_before(end)) {
                inner.push_back(b_.hard_line());
                inner.push_back(comment_doc(take_comment()));
            }
            if (!had_leftovers) {
                append_trailing_comments(inner, end, is_last ? close : elements[i + 1].begin);
            }
            if (!is_last) {
                inner.push_back(b_.line());
            }
        }
        while (has_comment_before(close)) {
            if (!inner.empty()) {
                inner.push_back(b_.hard_line());
            }
            inner.push_back(comment_doc(take_comment()));
            inner.push_back(b_.break_parent());
        }
        if (inner.empty()) {
            return b_.text(std::string(open) + std::string(close_text));
        }
        // A trailing comma in the source is the author asking for one element
        // per line; the expanded form ends in a comma, so this is stable.
        if (!force_trailing_comma && has_trailing_comma(close)) {
            inner.push_back(b_.break_parent());
        }
        return b_.group(b_.concat({b_.text(open),
                                   b_.indent(b_.concat({b_.soft_line(), b_.concat(inner)})),
                                   b_.soft_line(),
                                   b_.text(close_text)}));
    }

    bool has_trailing_comma(std::uint32_t close) const {
        std::size_t position = std::min<std::size_t>(close, text_.size());
        while (position > 0) {
            const char c = text_[position - 1];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                return c == ',';
            }
            --position;
        }
        return false;
    }

    // `prefix value`, moving the value to an indented line of its own when it
    // does not fit and has no brackets of its own to break at.
    DocId assigned(DocId prefix, ExprId value) {
        const ExprData& data = ast_.expr(value).data;
        const bool breaks_inside =
            std::holds_alternative<CallExpr>(data) || std::holds_alternative<MatchExpr>(data) ||
            std::holds_alternative<TupleExpr>(data) || std::holds_alternative<ShapeExpr>(data) ||
            std::holds_alternative<SomeExpr>(data);
        if (breaks_inside) {
            return b_.concat({prefix, b_.text(" "), expr(value)});
        }
        return b_.group(b_.concat({prefix, b_.indent(b_.concat({b_.line(), block_expr(value)}))}));
    }

    // An expression that starts its own indented line: continuation lines of
    // an operator chain align with its first operand instead of indenting.
    DocId block_expr(ExprId id) {
        if (const auto* top = std::get_if<BinaryExpr>(&ast_.expr(id).data)) {
            return binary(*top, false);
        }
        return expr(id);
    }

    // ------------------------------------------------------------------- names

    DocId name(const Name& name) { return b_.text(name.text); }

    DocId path(const std::vector<Name>& names) {
        std::vector<DocId> parts;
        parts.reserve(names.size());
        for (const Name& segment : names) {
            if (!parts.empty()) {
                parts.push_back(b_.text("."));
            }
            parts.push_back(name(segment));
        }
        return b_.concat(parts);
    }

    DocId comma_separated(const std::vector<DocId>& docs) { return b_.join(b_.text(", "), docs); }

    // ------------------------------------------------------------------- types

    DocId generic_arg(const GenericArg& arg) {
        return arg.type != no_id ? type(arg.type) : expr(arg.expr);
    }

    DocId generic_args(const std::vector<GenericArg>& args) {
        if (args.empty()) {
            return b_.nil();
        }
        std::vector<DocId> docs;
        docs.reserve(args.size());
        for (const GenericArg& arg : args) {
            docs.push_back(generic_arg(arg));
        }
        return b_.concat({b_.text("<"), comma_separated(docs), b_.text(">")});
    }

    DocId type(TypeId id) {
        const Type& node = ast_.type(id);
        return std::visit(
            Overloaded{
                [&](const ErrorType&) { return b_.text(sources_.text(node.span)); },
                [&](const NamedType& named) {
                    return b_.concat({path(named.path), generic_args(named.args)});
                },
                [&](const TensorType& tensor) {
                    std::vector<DocId> dims;
                    dims.reserve(tensor.shape.size());
                    for (const ShapeElement& element : tensor.shape) {
                        dims.push_back(element.dim == no_id
                                           ? b_.concat({b_.text("*"), name(element.pack)})
                                           : expr(element.dim));
                    }
                    return b_.concat({b_.text("Tensor["),
                                      comma_separated(dims),
                                      b_.text("; "),
                                      tensor.dtype == no_id ? b_.nil() : type(tensor.dtype),
                                      b_.text("]")});
                },
                [&](const TupleType& tuple) {
                    std::vector<DocId> docs;
                    docs.reserve(tuple.elements.size());
                    for (const TypeId element : tuple.elements) {
                        docs.push_back(type(element));
                    }
                    return b_.concat({b_.text("("),
                                      comma_separated(docs),
                                      b_.text(tuple.elements.size() == 1 ? ",)" : ")")});
                },
                [&](const ArrayType& array) {
                    return b_.concat({b_.text("["),
                                      type(array.element),
                                      b_.text("; "),
                                      expr(array.length),
                                      b_.text("]")});
                },
                [&](const OptionalType& optional) {
                    return b_.concat({type(optional.inner), b_.text("?")});
                },
            },
            node.data);
    }

    // ---------------------------------------------------------------- patterns

    DocId pattern(PatternId id) {
        const Pattern& node = ast_.pattern(id);
        return std::visit(
            Overloaded{
                [&](const ErrorPattern&) { return b_.text(sources_.text(node.span)); },
                [&](const BindingPattern& binding) { return name(binding.name); },
                [&](const TuplePattern& tuple) {
                    std::vector<DocId> docs;
                    docs.reserve(tuple.elements.size());
                    for (const PatternId element : tuple.elements) {
                        docs.push_back(pattern(element));
                    }
                    return b_.concat({b_.text("("),
                                      comma_separated(docs),
                                      b_.text(tuple.elements.size() == 1 ? ",)" : ")")});
                },
                [&](const SomePattern& some) {
                    return b_.concat({b_.text("some("), pattern(some.inner), b_.text(")")});
                },
                [&](const NonePattern&) { return b_.text("none"); },
            },
            node.data);
    }

    // ------------------------------------------------------------- expressions

    std::vector<Element> expr_elements(const std::vector<ExprId>& ids) {
        std::vector<Element> elements;
        elements.reserve(ids.size());
        for (const ExprId id : ids) {
            const SourceSpan span = ast_.expr(id).span;
            elements.push_back({span.begin, span.end, [this, id] { return expr(id); }});
        }
        return elements;
    }

    DocId binary(const BinaryExpr& top, bool indent_continuation = true) {
        // Flatten a left-associative chain of equal precedence into one group
        // so that it breaks uniformly, with operators trailing each line.
        std::vector<const BinaryExpr*> chain{&top};
        ExprId first = top.lhs;
        while (const auto* lhs = std::get_if<BinaryExpr>(&ast_.expr(first).data)) {
            if (precedence(lhs->op) != precedence(top.op)) {
                break;
            }
            chain.push_back(lhs);
            first = lhs->lhs;
        }
        const DocId head = expr(first);
        std::vector<DocId> rest;
        for (auto link = chain.rbegin(); link != chain.rend(); ++link) {
            rest.push_back(b_.text(" " + std::string(binary_op_spelling((*link)->op))));
            rest.push_back(b_.line());
            rest.push_back(expr((*link)->rhs));
        }
        const DocId tail = b_.concat(rest);
        return b_.group(b_.concat({head, indent_continuation ? b_.indent(tail) : tail}));
    }

    DocId index_component(const IndexComponent& component) {
        switch (component.kind) {
        case IndexKind::Expr:
            return expr(component.value);
        case IndexKind::Ellipsis:
            return b_.text("...");
        case IndexKind::Pack:
            return b_.concat({b_.text("*"), name(component.pack)});
        case IndexKind::Slice:
            break;
        }
        std::vector<DocId> parts;
        if (component.start != no_id) {
            parts.push_back(expr(component.start));
        }
        parts.push_back(b_.text(":"));
        if (component.stop != no_id) {
            parts.push_back(expr(component.stop));
        }
        if (component.step != no_id) {
            parts.push_back(b_.text(":"));
            parts.push_back(expr(component.step));
        }
        return b_.concat(parts);
    }

    DocId expr(ExprId id) {
        if (id == no_id) {
            return b_.nil();
        }
        const Expr& node = ast_.expr(id);
        return std::visit(
            Overloaded{
                [&](const ErrorExpr&) { return b_.text(sources_.text(node.span)); },
                [&](const LiteralExpr&) { return b_.text(sources_.text(node.span)); },
                [&](const NameExpr& named) { return name(named.name); },
                [&](const NoneExpr&) { return b_.text("none"); },
                [&](const SomeExpr& some) {
                    return b_.concat({b_.text("some("), expr(some.value), b_.text(")")});
                },
                [&](const ParenExpr& paren) {
                    return b_.concat({b_.text("("), expr(paren.inner), b_.text(")")});
                },
                [&](const TupleExpr& tuple) {
                    return list("(",
                                ")",
                                expr_elements(tuple.elements),
                                node.span.end - 1,
                                tuple.elements.size() == 1);
                },
                [&](const ShapeExpr& shape) {
                    return list("[", "]", expr_elements(shape.dims), node.span.end - 1);
                },
                [&](const UnaryExpr& unary) {
                    return b_.concat({b_.text(unary_op_spelling(unary.op)), expr(unary.operand)});
                },
                [&](const BinaryExpr& top) { return binary(top); },
                [&](const CallExpr& call) {
                    const DocId callee = expr(call.callee);
                    const DocId generics = generic_args(call.generic_args);
                    std::vector<Element> args;
                    args.reserve(call.args.size());
                    for (const Argument& arg : call.args) {
                        const SourceSpan value_span = ast_.expr(arg.value).span;
                        const std::uint32_t begin =
                            arg.keyword.text.empty() ? value_span.begin : arg.keyword.span.begin;
                        args.push_back({begin, value_span.end, [this, &arg] {
                                            if (arg.keyword.text.empty()) {
                                                return expr(arg.value);
                                            }
                                            return b_.concat({name(arg.keyword),
                                                              b_.text(" = "),
                                                              expr(arg.value)});
                                        }});
                    }
                    return b_.concat({callee, generics, list("(", ")", args, node.span.end - 1)});
                },
                [&](const IndexExpr& index) {
                    const DocId base = expr(index.base);
                    std::vector<DocId> components;
                    components.reserve(index.components.size());
                    for (const IndexComponent& component : index.components) {
                        components.push_back(index_component(component));
                    }
                    return b_.concat(
                        {base, b_.text("["), comma_separated(components), b_.text("]")});
                },
                [&](const MemberExpr& member) {
                    return b_.concat({expr(member.base), b_.text("."), name(member.member)});
                },
                [&](const ReductionExpr& reduction) {
                    std::vector<DocId> head{b_.text(reduction_kind_spelling(reduction.kind))};
                    if (reduction.accumulator != no_id) {
                        head.push_back(b_.text("<"));
                        head.push_back(type(reduction.accumulator));
                        head.push_back(b_.text(">"));
                    }
                    std::vector<DocId> indices;
                    indices.reserve(reduction.indices.size());
                    for (const Name& index : reduction.indices) {
                        indices.push_back(name(index));
                    }
                    head.push_back(b_.text("["));
                    head.push_back(comma_separated(indices));
                    head.push_back(b_.text("]"));
                    const DocId head_doc = b_.concat(head);
                    return b_.group(b_.concat(
                        {head_doc, b_.indent(b_.concat({b_.line(), block_expr(reduction.body)}))}));
                },
                [&](const IfExpr& conditional) {
                    const DocId condition = expr(conditional.condition);
                    const DocId then_value = expr(conditional.then_value);
                    const DocId else_value = expr(conditional.else_value);
                    return b_.group(b_.concat({
                        b_.text("if "),
                        condition,
                        b_.text(" {"),
                        b_.indent(b_.concat({b_.line(), then_value})),
                        b_.line(),
                        b_.text("} else {"),
                        b_.indent(b_.concat({b_.line(), else_value})),
                        b_.line(),
                        b_.text("}"),
                    }));
                },
                [&](const MatchExpr& match) {
                    const DocId scrutinee = expr(match.scrutinee);
                    std::vector<Element> arms;
                    arms.reserve(match.arms.size());
                    for (const MatchArm& arm : match.arms) {
                        arms.push_back({ast_.pattern(arm.pattern).span.begin,
                                        ast_.expr(arm.value).span.end,
                                        [this, &arm] {
                                            const DocId head =
                                                b_.concat({pattern(arm.pattern), b_.text(" =>")});
                                            return assigned(head, arm.value);
                                        }});
                    }
                    return b_.concat({b_.text("match "),
                                      scrutinee,
                                      b_.text(" "),
                                      braced(arms, node.span.end - 1)});
                },
            },
            node.data);
    }

    // -------------------------------------------------------------- statements

    std::vector<Element> statement_elements(const std::vector<StmtId>& ids) {
        std::vector<Element> elements;
        elements.reserve(ids.size());
        for (const StmtId id : ids) {
            const SourceSpan span = ast_.stmt(id).span;
            elements.push_back({span.begin, span.end, [this, id] { return stmt(id); }});
        }
        return elements;
    }

    DocId annotated(DocId head, TypeId annotation) {
        if (annotation == no_id) {
            return head;
        }
        return b_.concat({head, b_.text(": "), type(annotation)});
    }

    DocId stmt(StmtId id) {
        const Stmt& node = ast_.stmt(id);
        return std::visit(
            Overloaded{
                [&](const ErrorStmt&) { return b_.text(sources_.text(node.span)); },
                [&](const LetStmt& let) {
                    const DocId head =
                        annotated(b_.concat({b_.text("let "), pattern(let.pattern)}), let.type);
                    return assigned(b_.concat({head, b_.text(" =")}), let.value);
                },
                [&](const ComprehensionStmt& comprehension) {
                    std::vector<DocId> outputs;
                    outputs.reserve(comprehension.outputs.size());
                    for (const IndexOutput& output : comprehension.outputs) {
                        outputs.push_back(
                            b_.concat({b_.text(output.is_pack ? "*" : ""), name(output.name)}));
                    }
                    const DocId head = b_.concat({b_.text("let "),
                                                  name(comprehension.target),
                                                  b_.text("["),
                                                  comma_separated(outputs),
                                                  b_.text("] =")});
                    return assigned(head, comprehension.value);
                },
                [&](const VarStmt& var) {
                    const DocId head =
                        annotated(b_.concat({b_.text("var "), name(var.name)}), var.type);
                    return assigned(b_.concat({head, b_.text(" =")}), var.value);
                },
                [&](const AssignStmt& assign) {
                    return assigned(b_.concat({name(assign.target), b_.text(" =")}), assign.value);
                },
                [&](const ReturnStmt& ret) {
                    if (ret.value == no_id) {
                        return b_.text("return");
                    }
                    // The value stays on the `return` line so that the
                    // statement can never be read as a bare `return`.
                    return b_.concat({b_.text("return "), expr(ret.value)});
                },
                [&](const StaticForStmt& loop) {
                    DocId iterable = expr(loop.iterable);
                    if (loop.range_end != no_id) {
                        iterable = b_.concat({iterable, b_.text(".."), expr(loop.range_end)});
                    }
                    const DocId head = b_.concat({b_.text("static for "),
                                                  pattern(loop.pattern),
                                                  b_.text(" in "),
                                                  iterable,
                                                  b_.text(" ")});
                    return b_.concat(
                        {head, braced(statement_elements(loop.body), node.span.end - 1)});
                },
            },
            node.data);
    }

    // ------------------------------------------------------------------- items

    DocId generic_params(const std::vector<GenericParam>& params, SourceSpan list_span) {
        if (list_span.empty()) {
            return b_.nil();
        }
        std::vector<Element> elements;
        elements.reserve(params.size());
        for (const GenericParam& param : params) {
            elements.push_back({param.span.begin, param.span.end, [this, &param] {
                                    std::vector<DocId> parts{b_.text(param.is_pack ? "*" : ""),
                                                             name(param.name),
                                                             b_.text(": "),
                                                             name(param.constraint_name)};
                                    if (param.default_value.type != no_id ||
                                        param.default_value.expr != no_id) {
                                        parts.push_back(b_.text(" = "));
                                        parts.push_back(generic_arg(param.default_value));
                                    }
                                    return b_.concat(parts);
                                }});
        }
        return list("<", ">", elements, list_span.end - 1);
    }

    DocId function(const FunctionDecl& decl, SourceSpan item_span) {
        const std::string_view keyword = decl.kind == FunctionKind::Fn   ? "fn "
                                         : decl.kind == FunctionKind::Op ? "op "
                                                                         : "entry ";
        std::vector<DocId> parts{b_.text(keyword), name(decl.name)};
        parts.push_back(generic_params(decl.generics, decl.generics_span));

        std::vector<Element> params;
        params.reserve(decl.parameters.size());
        for (const Parameter& param : decl.parameters) {
            params.push_back({param.span.begin, param.span.end, [this, &param] {
                                  std::vector<DocId> docs{
                                      name(param.name), b_.text(": "), type(param.type)};
                                  if (param.default_value != no_id) {
                                      docs.push_back(b_.text(" = "));
                                      docs.push_back(expr(param.default_value));
                                  }
                                  return b_.concat(docs);
                              }});
        }
        parts.push_back(list("(", ")", params, decl.parameters_span.end - 1));

        if (decl.return_type != no_id) {
            parts.push_back(b_.text(" -> "));
            parts.push_back(type(decl.return_type));
        }

        where_clause(parts, decl.constraints);
        parts.push_back(braced(statement_elements(decl.body), item_span.end - 1));
        return b_.concat(parts);
    }

    // Appends the `where` clause and the space or line before the body's `{`.
    // One constraint stays on the `where` line; several go one per line with
    // the `{` on its own line.
    void where_clause(std::vector<DocId>& parts, const std::vector<ExprId>& constraints) {
        if (constraints.size() == 1) {
            parts.push_back(b_.hard_line());
            parts.push_back(b_.text("where "));
            parts.push_back(expr(constraints.front()));
            parts.push_back(b_.text(" "));
        } else if (!constraints.empty()) {
            std::vector<DocId> lines;
            for (std::size_t i = 0; i < constraints.size(); ++i) {
                lines.push_back(b_.hard_line());
                lines.push_back(expr(constraints[i]));
                if (i + 1 != constraints.size()) {
                    lines.push_back(b_.text(","));
                }
            }
            parts.push_back(b_.hard_line());
            parts.push_back(b_.text("where"));
            parts.push_back(b_.indent(b_.concat(lines)));
            parts.push_back(b_.hard_line());
        } else {
            parts.push_back(b_.text(" "));
        }
    }

    static bool is_compact(const Item& item) {
        return std::holds_alternative<ConstDecl>(item.data) ||
               std::holds_alternative<MemberDecl>(item.data) ||
               std::holds_alternative<TypeAliasDecl>(item.data);
    }

    // Declarations with bodies are always set apart by blank lines; runs of
    // one-line declarations keep the grouping the author chose.
    std::vector<Element> item_elements(const std::vector<ItemId>& ids) {
        std::vector<Element> elements;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const ItemId id = ids[i];
            const Item& node = ast_.item(id);
            const bool blank_before =
                i != 0 && !(is_compact(node) && is_compact(ast_.item(ids[i - 1])));
            elements.push_back(
                {node.span.begin, node.span.end, [this, id] { return item(id); }, blank_before});
        }
        return elements;
    }

    DocId item(ItemId id) {
        const Item& node = ast_.item(id);
        const DocId visibility = b_.text(node.is_pub ? "pub " : "");
        const std::uint32_t close = node.span.end - 1;
        const DocId body = std::visit(
            Overloaded{
                [&](const ErrorItem&) { return b_.text(sources_.text(node.span)); },
                [&](const ConstDecl& decl) {
                    const DocId head =
                        annotated(b_.concat({b_.text("const "), name(decl.name)}), decl.type);
                    return assigned(b_.concat({head, b_.text(" =")}), decl.value);
                },
                [&](const TypeAliasDecl& decl) {
                    const DocId head = b_.concat({b_.text("type "),
                                                  name(decl.name),
                                                  generic_params(decl.generics, decl.generics_span),
                                                  b_.text(" =")});
                    return b_.group(
                        b_.concat({head, b_.indent(b_.concat({b_.line(), type(decl.type)}))}));
                },
                [&](const StructDecl& decl) {
                    const DocId head = b_.concat({b_.text("struct "),
                                                  name(decl.name),
                                                  generic_params(decl.generics, decl.generics_span),
                                                  b_.text(" ")});
                    std::vector<Element> fields;
                    fields.reserve(decl.fields.size());
                    for (const FieldDecl& field : decl.fields) {
                        fields.push_back({field.span.begin, field.span.end, [this, &field] {
                                              return b_.concat({name(field.name),
                                                                b_.text(": "),
                                                                type(field.type)});
                                          }});
                    }
                    return b_.concat({head, braced(fields, close)});
                },
                [&](const EnumDecl& decl) {
                    const DocId head = b_.concat({b_.text("enum "),
                                                  name(decl.name),
                                                  generic_params(decl.generics, decl.generics_span),
                                                  b_.text(" ")});
                    std::vector<Element> variants;
                    variants.reserve(decl.variants.size());
                    for (const Name& variant : decl.variants) {
                        variants.push_back({variant.span.begin, variant.span.end, [this, &variant] {
                                                return b_.concat({name(variant), b_.text(",")});
                                            }});
                    }
                    return b_.concat({head, braced(variants, close)});
                },
                [&](const FunctionDecl& decl) { return function(decl, node.span); },
                [&](const BlockDecl& decl) {
                    std::vector<DocId> parts{b_.text("block "),
                                             name(decl.name),
                                             generic_params(decl.generics, decl.generics_span)};
                    where_clause(parts, decl.constraints);
                    parts.push_back(braced(item_elements(decl.members), close));
                    return b_.concat(parts);
                },
                [&](const MemberDecl& decl) {
                    const std::string keyword = std::string(member_keyword(decl.kind)) + " ";
                    const DocId head = b_.concat(
                        {b_.text(keyword), name(decl.name), b_.text(": "), type(decl.type)});
                    if (decl.default_value == no_id) {
                        return head;
                    }
                    return assigned(b_.concat({head, b_.text(" =")}), decl.default_value);
                },
            },
            node.data);
        return b_.concat({visibility, body});
    }

    DocId use(const UseDecl& decl) {
        std::vector<DocId> parts{b_.text("use "), path(decl.path)};
        if (decl.names.empty()) {
            return b_.concat(parts);
        }
        parts.push_back(b_.text("::"));
        const auto import_doc = [this](const ImportName& import) {
            if (import.alias.text.empty()) {
                return name(import.name);
            }
            return b_.concat({name(import.name), b_.text(" as "), name(import.alias)});
        };
        if (!decl.is_braced) {
            parts.push_back(import_doc(decl.names.front()));
            return b_.concat(parts);
        }
        std::vector<Element> names;
        names.reserve(decl.names.size());
        for (const ImportName& import : decl.names) {
            const std::uint32_t end =
                import.alias.text.empty() ? import.name.span.end : import.alias.span.end;
            names.push_back({import.name.span.begin, end, [&import, import_doc] {
                                 return import_doc(import);
                             }});
        }
        parts.push_back(list("{", "}", names, decl.span.end - 1));
        return b_.concat(parts);
    }

    std::vector<Element> file_elements() {
        std::vector<Element> elements;
        if (!ast_.module_path.empty()) {
            elements.push_back({ast_.module_span.begin, ast_.module_span.end, [this] {
                                    return b_.concat({b_.text("module "), path(ast_.module_path)});
                                }});
        }
        for (std::size_t i = 0; i < ast_.uses.size(); ++i) {
            const UseDecl& decl = ast_.uses[i];
            elements.push_back(
                {decl.span.begin, decl.span.end, [this, &decl] { return use(decl); }, i == 0});
        }
        std::vector<Element> items = item_elements(ast_.items);
        if (!items.empty()) {
            items.front().blank_before = true;
        }
        elements.insert(elements.end(),
                        std::make_move_iterator(items.begin()),
                        std::make_move_iterator(items.end()));
        if (!elements.empty()) {
            elements.front().blank_before = false;
        }
        // `use` declarations may legally only precede items, so source order
        // already matches this order.
        return elements;
    }

    const Ast& ast_;
    const SourceManager& sources_;
    std::string_view text_;
    DocBuilder b_;
    std::size_t next_comment_ = 0;
};

} // namespace

std::string
format(const ast::Ast& ast, const SourceManager& sources, const LayoutOptions& options) {
    return Formatter(ast, sources).run(options);
}

} // namespace linnet::format
