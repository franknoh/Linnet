#include "linnet/ast/ast.hpp"

#include <utility>

namespace linnet::ast {

std::string_view unary_op_spelling(UnaryOp op) {
    switch (op) {
    case UnaryOp::Not:
        return "!";
    case UnaryOp::Negate:
        return "-";
    case UnaryOp::Plus:
        return "+";
    }
    return "?";
}

std::string_view binary_op_spelling(BinaryOp op) {
    switch (op) {
    case BinaryOp::Or:
        return "||";
    case BinaryOp::And:
        return "&&";
    case BinaryOp::Equal:
        return "==";
    case BinaryOp::NotEqual:
        return "!=";
    case BinaryOp::Less:
        return "<";
    case BinaryOp::LessEqual:
        return "<=";
    case BinaryOp::Greater:
        return ">";
    case BinaryOp::GreaterEqual:
        return ">=";
    case BinaryOp::Add:
        return "+";
    case BinaryOp::Subtract:
        return "-";
    case BinaryOp::Multiply:
        return "*";
    case BinaryOp::Divide:
        return "/";
    case BinaryOp::Remainder:
        return "%";
    }
    return "?";
}

std::string_view reduction_kind_spelling(ReductionKind kind) {
    switch (kind) {
    case ReductionKind::Sum:
        return "sum";
    case ReductionKind::Prod:
        return "prod";
    case ReductionKind::Max:
        return "max";
    case ReductionKind::Min:
        return "min";
    case ReductionKind::Any:
        return "any";
    case ReductionKind::All:
        return "all";
    }
    return "?";
}

namespace {

template <typename Node>
std::uint32_t append(std::vector<Node>& arena, Node node) {
    arena.push_back(std::move(node));
    return static_cast<std::uint32_t>(arena.size() - 1);
}

} // namespace

ExprId Ast::add(Expr node) {
    return append(exprs_, std::move(node));
}
TypeId Ast::add(Type node) {
    return append(types_, std::move(node));
}
PatternId Ast::add(Pattern node) {
    return append(patterns_, std::move(node));
}
StmtId Ast::add(Stmt node) {
    return append(stmts_, std::move(node));
}
ItemId Ast::add(Item node) {
    return append(items_, std::move(node));
}

Ast::Mark Ast::mark() const {
    return {exprs_.size(), types_.size(), patterns_.size(), stmts_.size(), items_.size()};
}

void Ast::rewind(const Mark& mark) {
    exprs_.erase(exprs_.begin() + static_cast<std::ptrdiff_t>(mark.exprs), exprs_.end());
    types_.erase(types_.begin() + static_cast<std::ptrdiff_t>(mark.types), types_.end());
    patterns_.erase(patterns_.begin() + static_cast<std::ptrdiff_t>(mark.patterns),
                    patterns_.end());
    stmts_.erase(stmts_.begin() + static_cast<std::ptrdiff_t>(mark.stmts), stmts_.end());
    items_.erase(items_.begin() + static_cast<std::ptrdiff_t>(mark.items), items_.end());
}

} // namespace linnet::ast
