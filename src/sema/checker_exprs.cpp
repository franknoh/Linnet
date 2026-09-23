#include "linnet/diagnostic/codes.hpp"

#include "checker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

namespace linnet::sema {

namespace {

template <typename... Visitors>
struct Overloaded : Visitors... {
    using Visitors::operator()...;
};
template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

std::optional<std::int64_t> parse_integer(std::string_view text) {
    int base = 10;
    if (text.starts_with("0x")) {
        base = 16;
        text.remove_prefix(2);
    } else if (text.starts_with("0b")) {
        base = 2;
        text.remove_prefix(2);
    }
    constexpr std::uint64_t limit = std::numeric_limits<std::int64_t>::max();
    std::uint64_t value = 0;
    for (const char c : text) {
        if (c == '_') {
            continue;
        }
        const int digit = c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;
        if (value >
            (limit - static_cast<std::uint64_t>(digit)) / static_cast<std::uint64_t>(base)) {
            return std::nullopt;
        }
        value = value * static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(digit);
    }
    return static_cast<std::int64_t>(value);
}

struct IntRange {
    std::int64_t low;
    std::int64_t high;
};

// Range of an integer kind; u64 is limited to what a literal can spell.
IntRange int_range(ScalarKind kind) {
    constexpr std::int64_t max = std::numeric_limits<std::int64_t>::max();
    switch (kind) {
    case ScalarKind::I8:
        return {-128, 127};
    case ScalarKind::I16:
        return {-32768, 32767};
    case ScalarKind::I32:
        return {-2147483648LL, 2147483647LL};
    case ScalarKind::U8:
        return {0, 255};
    case ScalarKind::U16:
        return {0, 65535};
    case ScalarKind::U32:
        return {0, 4294967295LL};
    case ScalarKind::U64:
        return {0, max};
    default:
        return {std::numeric_limits<std::int64_t>::min(), max};
    }
}

double max_finite(ScalarKind kind) {
    switch (kind) {
    case ScalarKind::F16:
        return 65504.0;
    case ScalarKind::BF16:
        return 3.3895313892515355e38;
    case ScalarKind::F32:
        return 3.4028234663852886e38;
    default:
        return std::numeric_limits<double>::max();
    }
}

bool is_comparison(ast::BinaryOp op) {
    return op >= ast::BinaryOp::Equal && op <= ast::BinaryOp::GreaterEqual;
}

bool is_literal(TypeKind kind) {
    return kind == TypeKind::CompileInt || kind == TypeKind::FloatLiteral;
}

} // namespace

// -------------------------------------------------------------------- literals

TypeId Checker::check_literal(const ast::Expr& node, const ast::LiteralExpr& literal) {
    switch (literal.kind) {
    case ast::LiteralKind::Integer: {
        const auto value = parse_integer(text(node.span));
        if (!value) {
            error(codes::literal_out_of_range, node.span, "integer literal is too large")
                .note("integer literals are limited to the range of `i64`");
            return types_.error();
        }
        return types_.compile_int(shape::Poly(*value));
    }
    case ast::LiteralKind::Float: {
        std::string digits(text(node.span));
        std::erase(digits, '_');
        const double value = std::strtod(digits.c_str(), nullptr);
        if (!std::isfinite(value)) {
            error(codes::literal_out_of_range, node.span, "float literal is too large");
            return types_.error();
        }
        return types_.float_literal(value);
    }
    case ast::LiteralKind::True:
    case ast::LiteralKind::False:
        return types_.scalar(ScalarKind::Bool);
    case ast::LiteralKind::String:
        error(codes::invalid_operand, node.span, "string literals are not values")
            .note("Linnet has no string type; strings only name external resources");
        return types_.error();
    }
    return types_.error();
}

bool Checker::literal_fits(const TypeData& literal, DType dtype, SourceSpan span) {
    const DTypeClass target = types_.class_of(dtype);
    const bool is_bool = !dtype.is_var && dtype.scalar == ScalarKind::Bool;
    const std::string name = types_.to_string(dtype);
    if (literal.kind == TypeKind::CompileInt) {
        if (is_bool || (dtype.is_var && target == DTypeClass::Any)) {
            error(codes::invalid_operand, span, "an integer cannot be used as `" + name + "`")
                .note(dtype.is_var ? "`" + name + "` may be `bool`; constrain it with `Numeric`"
                                   : "there is no conversion between integers and `bool`");
            return false;
        }
        const auto value = literal.value.constant();
        if (value && !dtype.is_var && is_integer(dtype.scalar)) {
            const IntRange range = int_range(dtype.scalar);
            if (*value < range.low || *value > range.high) {
                error(codes::literal_out_of_range,
                      span,
                      "`" + std::to_string(*value) + "` does not fit in `" + name + "`");
                return false;
            }
        }
        return true;
    }
    if (target != DTypeClass::Float) {
        error(codes::invalid_operand, span, "a float literal cannot be used as `" + name + "`")
            .help(dtype.is_var ? "constrain `" + name + "` with `Float`"
                               : "write an integer literal, or convert explicitly with `cast`");
        return false;
    }
    if (literal.number && !dtype.is_var && std::fabs(*literal.number) > max_finite(dtype.scalar)) {
        error(codes::literal_out_of_range, span, "this literal does not fit in `" + name + "`");
        return false;
    }
    return true;
}

TypeId Checker::default_literals(TypeId type, SourceSpan span) {
    const TypeData& data = types_.get(type);
    switch (data.kind) {
    case TypeKind::CompileInt:
        // Symbolic dimensions stay compile-time values.
        return data.value.constant() ? types_.scalar(ScalarKind::I64) : type;
    case TypeKind::FloatLiteral:
        return types_.scalar(ScalarKind::F64);
    case TypeKind::NoneLiteral:
        error(codes::none_needs_context, span, "cannot tell which optional type `none` has")
            .help("add a type annotation");
        return types_.error();
    case TypeKind::Tuple:
    case TypeKind::Optional: {
        std::vector<TypeId> elements;
        bool has_changed = false;
        for (const TypeId element : data.elements) {
            elements.push_back(default_literals(element, span));
            has_changed = has_changed || elements.back() != element;
        }
        if (!has_changed) {
            return type;
        }
        return data.kind == TypeKind::Tuple ? types_.tuple(std::move(elements))
                                            : types_.optional(elements.front());
    }
    default:
        return type;
    }
}

// ----------------------------------------------------------------- conversions

bool Checker::assignable(TypeId actual, TypeId expected) {
    const TypeData& from = types_.get(actual);
    const TypeData& to = types_.get(expected);
    if (from.kind == TypeKind::Error || to.kind == TypeKind::Error) {
        return true;
    }
    if (is_literal(from.kind) && to.kind == TypeKind::Scalar) {
        const DTypeClass target = types_.class_of(to.dtype);
        const bool is_bool = !to.dtype.is_var && to.dtype.scalar == ScalarKind::Bool;
        if (from.kind == TypeKind::FloatLiteral) {
            return target == DTypeClass::Float;
        }
        return !is_bool && !(to.dtype.is_var && target == DTypeClass::Any);
    }
    if (from.kind == TypeKind::NoneLiteral) {
        return to.kind == TypeKind::Optional;
    }
    if ((from.kind == TypeKind::Tuple && to.kind == TypeKind::Tuple) ||
        (from.kind == TypeKind::Optional && to.kind == TypeKind::Optional)) {
        if (from.elements.size() != to.elements.size()) {
            return false;
        }
        for (std::size_t i = 0; i < from.elements.size(); ++i) {
            if (!assignable(from.elements[i], to.elements[i])) {
                return false;
            }
        }
        return true;
    }
    return types_.equal(actual, expected, env_->solver);
}

void Checker::report_mismatch(TypeId actual,
                              TypeId expected,
                              SourceSpan span,
                              std::string_view what) {
    const TypeData& from = types_.get(actual);
    const TypeData& to = types_.get(expected);
    const std::string subject(what);
    if (from.kind == TypeKind::Tensor && to.kind == TypeKind::Tensor) {
        if (from.dtype != to.dtype) {
            error(codes::dtype_mismatch, span, subject + " has the wrong dtype")
                .note("expected dtype " + types_.to_string(to.dtype))
                .note("found dtype " + types_.to_string(from.dtype))
                .help("Linnet does not implicitly convert tensor dtypes; use an explicit cast");
            return;
        }
        error(codes::shape_mismatch, span, subject + " has the wrong shape")
            .note("expected `" + str(expected) + "`")
            .note("found `" + str(actual) + "`")
            .help("if the shapes are equal, add a `where` constraint that proves it");
        return;
    }
    error(codes::type_mismatch,
          span,
          subject + " has type `" + str(actual) + "`, expected `" + str(expected) + "`");
}

TypeId Checker::coerce(TypeId actual, TypeId expected, SourceSpan span, std::string_view what) {
    if (types_.is_error(actual) || types_.is_error(expected)) {
        return expected;
    }
    if (!assignable(actual, expected)) {
        report_mismatch(actual, expected, span, what);
        return types_.error();
    }
    const TypeData& from = types_.get(actual);
    if (is_literal(from.kind) && !literal_fits(from, types_.get(expected).dtype, span)) {
        return types_.error();
    }
    return expected;
}

TypeId Checker::join_branches(TypeId a, TypeId b, SourceSpan span) {
    if (types_.is_error(a) || types_.is_error(b)) {
        return types_.is_error(a) ? b : a;
    }
    const TypeKind first = types_.kind(a);
    const TypeKind second = types_.kind(b);
    if (first == TypeKind::CompileInt && second == TypeKind::CompileInt) {
        return types_.scalar(ScalarKind::I64);
    }
    if (is_literal(first) && is_literal(second)) {
        return types_.float_literal(std::nullopt);
    }
    if (assignable(b, a)) {
        return coerce(b, a, span, "this branch");
    }
    if (assignable(a, b)) {
        return coerce(a, b, span, "this branch");
    }
    error(codes::branch_mismatch, span, "branches have different types")
        .note("one branch has type `" + str(a) + "`")
        .note("another has type `" + str(b) + "`");
    return types_.error();
}

// ----------------------------------------------------------------- elementwise

std::optional<Shape> Checker::broadcast(const Shape& a, const Shape& b, SourceSpan span) {
    Shape result(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < result.size(); ++i) {
        const ShapeElem* x = i < a.size() ? &a[a.size() - 1 - i] : nullptr;
        const ShapeElem* y = i < b.size() ? &b[b.size() - 1 - i] : nullptr;
        ShapeElem& out = result[result.size() - 1 - i];
        if (x == nullptr || y == nullptr) {
            // At least one side has this axis, since i < max(sizes).
            out = x != nullptr ? *x : y != nullptr ? *y : ShapeElem{};
            continue;
        }
        if (x->is_pack || y->is_pack) {
            if (x->is_pack && y->is_pack && x->pack == y->pack) {
                out = *x;
                continue;
            }
        } else if (env_->solver.prove_equal(x->dim, y->dim) ||
                   env_->solver.prove_equal(y->dim, shape::Poly(1))) {
            out = *x;
            continue;
        } else if (env_->solver.prove_equal(x->dim, shape::Poly(1))) {
            out = *y;
            continue;
        }
        const Shape left{*x};
        const Shape right{*y};
        error(codes::unproven_broadcast, span, "cannot prove that these shapes broadcast")
            .note("`" + types_.to_string(left) + "` and `" + types_.to_string(right) +
                  "` must be equal, or one of them must be 1")
            .note("left shape is [" + types_.to_string(a) + "]")
            .note("right shape is [" + types_.to_string(b) + "]")
            .help("use one shared dimension parameter or add a `where` constraint");
        return std::nullopt;
    }
    return result;
}

std::optional<DType> Checker::numeric_dtype(TypeId type) const {
    const TypeData& data = types_.get(type);
    if (data.kind == TypeKind::Scalar || data.kind == TypeKind::Tensor) {
        return data.dtype;
    }
    return std::nullopt;
}

TypeId Checker::elementwise(
    TypeId lhs, TypeId rhs, SourceSpan span, bool needs_numeric, bool for_comparison) {
    const TypeData& left = types_.get(lhs);
    const TypeData& right = types_.get(rhs);
    if (left.kind == TypeKind::Error || right.kind == TypeKind::Error) {
        return types_.error();
    }
    const auto left_dtype = numeric_dtype(lhs);
    const auto right_dtype = numeric_dtype(rhs);
    const bool is_valid = (left_dtype || is_literal(left.kind)) &&
                          (right_dtype || is_literal(right.kind)) && (left_dtype || right_dtype);
    if (!is_valid) {
        error(codes::invalid_operand,
              span,
              "operands of type `" + str(lhs) + "` and `" + str(rhs) + "` cannot be combined");
        return types_.error();
    }

    const DType dtype = left_dtype ? *left_dtype : *right_dtype;
    if (left_dtype && right_dtype && *left_dtype != *right_dtype) {
        const bool has_tensor = left.kind == TypeKind::Tensor || right.kind == TypeKind::Tensor;
        error(codes::dtype_mismatch,
              span,
              has_tensor ? "tensor dtypes do not match" : "operand dtypes do not match")
            .note("left operand has dtype " + types_.to_string(*left_dtype))
            .note("right operand has dtype " + types_.to_string(*right_dtype))
            .help("Linnet does not implicitly promote tensor dtypes; use an explicit cast");
        return types_.error();
    }
    if ((!left_dtype && !literal_fits(left, dtype, span)) ||
        (!right_dtype && !literal_fits(right, dtype, span))) {
        return types_.error();
    }
    const bool may_be_bool =
        dtype.is_var ? types_.class_of(dtype) == DTypeClass::Any : dtype.scalar == ScalarKind::Bool;
    if (needs_numeric && may_be_bool) {
        error(codes::invalid_operand,
              span,
              "this operator needs numeric operands, but the dtype is `" + types_.to_string(dtype) +
                  "`")
            .help(dtype.is_var ? "constrain `" + types_.to_string(dtype) + "` with `Numeric`"
                               : "use `&&`, `||`, `!`, or `select` for booleans");
        return types_.error();
    }

    const DType result = for_comparison ? DType::of(ScalarKind::Bool) : dtype;
    const bool left_tensor = left.kind == TypeKind::Tensor;
    const bool right_tensor = right.kind == TypeKind::Tensor;
    if (!left_tensor && !right_tensor) {
        return types_.scalar(result);
    }
    if (left_tensor && right_tensor) {
        auto shape = broadcast(left.shape, right.shape, span);
        return shape ? types_.tensor(std::move(*shape), result) : types_.error();
    }
    return types_.tensor(left_tensor ? left.shape : right.shape, result);
}

// ----------------------------------------------------------------- expressions

TypeId Checker::check_expr(ast::ExprId id, TypeId expected) {
    if (id == ast::no_id) {
        return types_.error();
    }
    const TypeId type = check_expr_inner(id, expected);
    facts(id).type = type;
    return type;
}

TypeId Checker::check_expr_inner(ast::ExprId id, TypeId expected) {
    const ast::Expr& node = ast().expr(id);
    current_expr_ = id;
    return std::visit(
        Overloaded{
            [&](const ast::ErrorExpr&) { return types_.error(); },
            [&](const ast::LiteralExpr& literal) { return check_literal(node, literal); },
            [&](const ast::NameExpr& name) { return check_name(node, name); },
            [&](const ast::NoneExpr&) {
                return expected != no_type && types_.kind(expected) == TypeKind::Optional
                           ? expected
                           : types_.none_literal();
            },
            [&](const ast::SomeExpr& some) {
                TypeId inner_expected = no_type;
                if (expected != no_type && types_.kind(expected) == TypeKind::Optional) {
                    inner_expected = types_.get(expected).elements.front();
                }
                const TypeId inner = check_expr(some.value, inner_expected);
                return types_.is_error(inner) ? inner : types_.optional(inner);
            },
            [&](const ast::ParenExpr& paren) { return check_expr(paren.inner, expected); },
            [&](const ast::TupleExpr& tuple) {
                std::vector<TypeId> elements;
                const bool has_context =
                    expected != no_type && types_.kind(expected) == TypeKind::Tuple &&
                    types_.get(expected).elements.size() == tuple.elements.size();
                elements.reserve(tuple.elements.size());
                for (std::size_t i = 0; i < tuple.elements.size(); ++i) {
                    elements.push_back(
                        check_expr(tuple.elements[i],
                                   has_context ? types_.get(expected).elements[i] : no_type));
                }
                return types_.tuple(std::move(elements));
            },
            [&](const ast::ShapeExpr& literal) {
                Shape shape;
                for (const ast::ExprId dim : literal.dims) {
                    shape::Poly poly = eval_dim(dim);
                    if (!poly.is_valid()) {
                        return types_.error();
                    }
                    check_dimension(poly, ast().expr(dim).span);
                    shape.push_back(ShapeElem::of(std::move(poly)));
                }
                return types_.shape_value(std::move(shape));
            },
            [&](const ast::UnaryExpr& unary) { return check_unary(node, unary); },
            [&](const ast::BinaryExpr& binary) { return check_binary(node, binary); },
            [&](const ast::CallExpr& call) { return check_call(node, call); },
            [&](const ast::IndexExpr& index) { return check_index(node, index); },
            [&](const ast::MemberExpr& member) { return check_member(node, member); },
            [&](const ast::ReductionExpr& reduction) { return check_reduction(node, reduction); },
            [&](const ast::IfExpr& conditional) { return check_if(node, conditional, expected); },
            [&](const ast::MatchExpr& match) { return check_match(node, match, expected); },
        },
        node.data);
}

TypeId Checker::value_of_entity(EntityId entity, SourceSpan span) {
    resolve(entity);
    const Entity& target = entities_[entity];
    switch (target.kind) {
    case EntityKind::Local:
    case EntityKind::Member:
    case EntityKind::Const:
        return target.type == no_type ? types_.error() : target.type;
    case EntityKind::GenericDim:
        return types_.compile_int(dims_.symbol(target.symbol));
    case EntityKind::Function:
        error(codes::wrong_symbol_kind,
              span,
              "`" + std::string(target.name) + "` is a function and must be called");
        return types_.error();
    default:
        error(codes::wrong_symbol_kind, span, "`" + std::string(target.name) + "` is not a value")
            .label(target.span, "declared here");
        return types_.error();
    }
}

TypeId Checker::check_name(const ast::Expr& node, const ast::NameExpr& name) {
    if (find_index(name.name.text) != nullptr) {
        error(codes::invalid_index_expression,
              node.span,
              "index `" + std::string(name.name.text) + "` can only be used inside `[...]`")
            .help("index a tensor with it, for example `iota(N)[" + std::string(name.name.text) +
                  "]` for its position");
        return types_.error();
    }
    const EntityId entity = lookup(name.name);
    facts(current_expr_).entity = entity;
    if (entity == no_entity) {
        auto report = error(codes::unknown_symbol,
                            node.span,
                            "cannot find `" + std::string(name.name.text) + "` in this scope");
        if (is_prelude_name(name.name.text)) {
            report.note("`" + std::string(name.name.text) + "` is a prelude function or type");
        }
        return types_.error();
    }
    return value_of_entity(entity, node.span);
}

TypeId Checker::check_unary(const ast::Expr& node, const ast::UnaryExpr& unary) {
    const TypeId operand = check_expr(unary.operand);
    const TypeData& data = types_.get(operand);
    if (data.kind == TypeKind::Error) {
        return operand;
    }
    if (unary.op == ast::UnaryOp::Not) {
        if (data.kind == TypeKind::Scalar && data.dtype == DType::of(ScalarKind::Bool)) {
            return operand;
        }
        error(codes::logical_not_bool,
              node.span,
              "`!` needs a scalar `bool`, found `" + str(operand) + "`");
        return types_.error();
    }
    if (data.kind == TypeKind::CompileInt) {
        return unary.op == ast::UnaryOp::Negate ? types_.compile_int(-data.value) : operand;
    }
    if (data.kind == TypeKind::FloatLiteral) {
        return unary.op == ast::UnaryOp::Negate && data.number ? types_.float_literal(-*data.number)
                                                               : operand;
    }
    const auto dtype = numeric_dtype(operand);
    const bool is_numeric = dtype && (dtype->is_var ? types_.class_of(*dtype) != DTypeClass::Any
                                                    : dtype->scalar != ScalarKind::Bool);
    if (!is_numeric) {
        error(codes::invalid_operand,
              node.span,
              "`" + std::string(ast::unary_op_spelling(unary.op)) +
                  "` needs a numeric operand, found `" + str(operand) + "`");
        return types_.error();
    }
    return operand;
}

TypeId Checker::check_binary(const ast::Expr& node, const ast::BinaryExpr& binary) {
    const TypeId lhs = check_expr(binary.lhs);
    const TypeId rhs = check_expr(binary.rhs);
    const TypeData& left = types_.get(lhs);
    const TypeData& right = types_.get(rhs);
    if (left.kind == TypeKind::Error || right.kind == TypeKind::Error) {
        return types_.error();
    }

    if (binary.op == ast::BinaryOp::And || binary.op == ast::BinaryOp::Or) {
        const auto is_bool = [](const TypeData& data) {
            return data.kind == TypeKind::Scalar && data.dtype == DType::of(ScalarKind::Bool);
        };
        if (is_bool(left) && is_bool(right)) {
            return lhs;
        }
        auto report = error(codes::logical_not_bool,
                            node.span,
                            "`" + std::string(ast::binary_op_spelling(binary.op)) +
                                "` needs scalar `bool` operands");
        report.note("left operand has type `" + str(lhs) + "`")
            .note("right operand has type `" + str(rhs) + "`");
        if (left.kind == TypeKind::Tensor || right.kind == TypeKind::Tensor) {
            report.help("combine boolean tensors elementwise with `select`");
        }
        return types_.error();
    }

    const bool compares = is_comparison(binary.op);
    if (left.kind == TypeKind::CompileInt && right.kind == TypeKind::CompileInt) {
        if (compares) {
            return types_.scalar(ScalarKind::Bool);
        }
        shape::Poly value;
        switch (binary.op) {
        case ast::BinaryOp::Add:
            value = left.value + right.value;
            break;
        case ast::BinaryOp::Subtract:
            value = left.value - right.value;
            break;
        case ast::BinaryOp::Multiply:
            value = left.value * right.value;
            break;
        default:
            if (!env_->solver.prove(shape::Relation::Greater, right.value, {})) {
                error(codes::divisor_not_positive,
                      ast().expr(binary.rhs).span,
                      "cannot prove that the divisor `" + str(right.value) + "` is positive")
                    .help("add a constraint such as `where " + str(right.value) + " > 0`");
                return types_.error();
            }
            value = binary.op == ast::BinaryOp::Divide ? dims_.floor_div(left.value, right.value)
                                                       : dims_.mod(left.value, right.value);
            break;
        }
        if (!value.is_valid()) {
            error(codes::not_compile_time, node.span, "integer arithmetic overflows");
            return types_.error();
        }
        return types_.compile_int(std::move(value));
    }
    if (is_literal(left.kind) && is_literal(right.kind)) {
        if (compares) {
            return types_.scalar(ScalarKind::Bool);
        }
        const auto number = [](const TypeData& data) -> std::optional<double> {
            if (data.kind == TypeKind::FloatLiteral) {
                return data.number;
            }
            const auto constant = data.value.constant();
            return constant ? std::optional<double>(static_cast<double>(*constant)) : std::nullopt;
        };
        const auto a = number(left);
        const auto b = number(right);
        if (!a || !b) {
            return types_.float_literal(std::nullopt);
        }
        switch (binary.op) {
        case ast::BinaryOp::Add:
            return types_.float_literal(*a + *b);
        case ast::BinaryOp::Subtract:
            return types_.float_literal(*a - *b);
        case ast::BinaryOp::Multiply:
            return types_.float_literal(*a * *b);
        case ast::BinaryOp::Divide:
            return types_.float_literal(*b == 0.0 ? std::nullopt : std::optional<double>(*a / *b));
        default:
            return types_.float_literal(std::nullopt);
        }
    }

    const bool is_equality =
        binary.op == ast::BinaryOp::Equal || binary.op == ast::BinaryOp::NotEqual;
    if (is_equality && left.kind == TypeKind::Enum && right.kind == TypeKind::Enum) {
        if (types_.equal(lhs, rhs, env_->solver)) {
            return types_.scalar(ScalarKind::Bool);
        }
    }
    return elementwise(lhs, rhs, node.span, !is_equality, compares);
}

Substitution Checker::substitution_of(const TypeData& nominal) {
    Substitution substitution;
    const DeclInfo& info = decls_[nominal.decl];
    for (std::size_t i = 0; i < info.generics.size() && i < nominal.args.size(); ++i) {
        const GenericInfo& param = info.generics[i];
        switch (param.kind) {
        case GenericKind::Dim:
            substitution.dims[param.symbol] = nominal.args[i].dim;
            break;
        case GenericKind::Pack:
            substitution.packs[param.symbol] = nominal.args[i].shape;
            break;
        case GenericKind::DType:
            substitution.dtypes[param.dtype_var] = nominal.args[i].dtype;
            break;
        }
    }
    return substitution;
}

TypeId Checker::check_member(const ast::Expr& node, const ast::MemberExpr& member) {
    const ast::ExprId self = current_expr_;
    // `module.item` and `Enum.Variant` are paths rather than value accesses.
    if (const auto* base = std::get_if<ast::NameExpr>(&ast().expr(member.base).data)) {
        const EntityId entity =
            find_index(base->name.text) == nullptr ? lookup(base->name) : no_entity;
        if (entity != no_entity && entities_[entity].kind == EntityKind::Module) {
            const EntityId item = lookup_in_module(entities_[entity].module_ref, member.member);
            facts(self).entity = item;
            return item == no_entity ? types_.error() : value_of_entity(item, node.span);
        }
        if (entity != no_entity && entities_[entity].kind == EntityKind::Enum) {
            resolve(entity);
            const DeclInfo& info = decls_[entity];
            if (std::find(info.variants.begin(), info.variants.end(), member.member.text) ==
                info.variants.end()) {
                error(codes::unknown_member,
                      member.member.span,
                      "enum `" + std::string(entities_[entity].name) + "` has no variant `" +
                          std::string(member.member.text) + "`");
                return types_.error();
            }
            std::vector<GenericValue> values;
            if (!bind_generic_args(info, {}, node.span, &values)) {
                return types_.error();
            }
            return types_.nominal(TypeKind::Enum, entity, std::move(values));
        }
    }

    const TypeId base = check_expr(member.base);
    const TypeData& data = types_.get(base);
    if (data.kind == TypeKind::Error || member.member.text.empty()) {
        return types_.error();
    }
    if (data.kind == TypeKind::Struct) {
        resolve(data.decl);
        const std::vector<FieldInfo>& fields = decls_[data.decl].fields;
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (fields[i].name == member.member.text) {
                facts(self).is_field = true;
                facts(self).field = static_cast<std::uint32_t>(i);
                return types_.substitute(fields[i].type, substitution_of(data));
            }
        }
    } else if (data.kind == TypeKind::Block) {
        const Scope& scope = decls_[data.decl].scope;
        const auto found = scope.find(member.member.text);
        if (found != scope.end() && entities_[found->second].kind == EntityKind::Member) {
            entities_[found->second].is_used = true;
            record_ref(member.member.span, found->second);
            facts(self).entity = found->second;
            resolve(found->second);
            return types_.substitute(entities_[found->second].type, substitution_of(data));
        }
        if (found != scope.end() && entities_[found->second].kind == EntityKind::Function) {
            error(codes::wrong_symbol_kind,
                  member.member.span,
                  "`" + std::string(member.member.text) + "` is a method and must be called");
            return types_.error();
        }
    }
    error(codes::unknown_member,
          member.member.span,
          "`" + str(base) + "` has no member `" + std::string(member.member.text) + "`");
    return types_.error();
}

TypeId Checker::check_if(const ast::Expr& node, const ast::IfExpr& conditional, TypeId expected) {
    const TypeId condition = check_expr(conditional.condition);
    const TypeData& data = types_.get(condition);
    const bool is_bool = data.kind == TypeKind::Scalar && data.dtype == DType::of(ScalarKind::Bool);
    if (data.kind != TypeKind::Error && !is_bool) {
        auto report =
            error(codes::condition_not_bool,
                  ast().expr(conditional.condition).span,
                  "an `if` condition must be a scalar `bool`, found `" + str(condition) + "`");
        if (data.kind == TypeKind::Tensor) {
            report.help("choose elementwise with `select(condition, a, b)`");
        }
    }
    const TypeId then_value = check_expr(conditional.then_value, expected);
    const TypeId else_value = check_expr(conditional.else_value, expected);
    return join_branches(then_value, else_value, node.span);
}

TypeId Checker::check_match(const ast::Expr& node, const ast::MatchExpr& match, TypeId expected) {
    const TypeId scrutinee = check_expr(match.scrutinee);
    const TypeData& subject = types_.get(scrutinee);
    const bool is_optional = subject.kind == TypeKind::Optional;
    const bool is_enum = subject.kind == TypeKind::Enum;
    if (subject.kind != TypeKind::Error && !is_optional && !is_enum) {
        error(codes::invalid_pattern,
              ast().expr(match.scrutinee).span,
              "`match` needs an optional or an enum, found `" + str(scrutinee) + "`");
    }
    if (is_enum) {
        resolve(subject.decl);
    }

    bool has_some = false;
    bool has_none = false;
    bool has_catch_all = false;
    std::vector<std::string_view> seen_variants;
    TypeId result = no_type;

    for (const ast::MatchArm& arm : match.arms) {
        const ast::Pattern& pattern = ast().pattern(arm.pattern);
        env_->scopes.emplace_back();
        if (has_catch_all) {
            error(codes::invalid_pattern, pattern.span, "this arm can never match")
                .note("an earlier arm already matches every value");
        }
        std::visit(Overloaded{
                       [&](const ast::ErrorPattern&) { has_catch_all = true; },
                       [&](const ast::BindingPattern& binding) {
                           if (is_enum) {
                               const auto& variants = decls_[subject.decl].variants;
                               if (std::find(variants.begin(), variants.end(), binding.name.text) !=
                                   variants.end()) {
                                   seen_variants.push_back(binding.name.text);
                                   return;
                               }
                           }
                           has_catch_all = true;
                           if (binding.name.text != "_") {
                               record_binding(binding.name, scrutinee);
                               model_->bound[env_->module][arm.pattern] =
                                   declare_local(binding.name, scrutinee, false);
                           }
                       },
                       [&](const ast::SomePattern& some) {
                           if (is_optional) {
                               has_some = true;
                               bind_pattern(some.inner, subject.elements.front(), false);
                           } else if (subject.kind != TypeKind::Error) {
                               error(codes::invalid_pattern,
                                     pattern.span,
                                     "`some(...)` only matches optional values");
                           }
                       },
                       [&](const ast::NonePattern&) {
                           if (is_optional) {
                               has_none = true;
                           } else if (subject.kind != TypeKind::Error) {
                               error(codes::invalid_pattern,
                                     pattern.span,
                                     "`none` only matches optional values");
                           }
                       },
                       [&](const ast::TuplePattern&) {
                           error(codes::invalid_pattern,
                                 pattern.span,
                                 "tuple patterns are not supported in `match` arms");
                       },
                   },
                   pattern.data);

        const TypeId value = check_expr(arm.value, expected);
        result =
            result == no_type ? value : join_branches(result, value, ast().expr(arm.value).span);
        env_->scopes.pop_back();
    }

    std::string missing;
    if (is_optional && !has_catch_all) {
        missing = !has_some ? "some(_)" : !has_none ? "none" : "";
    } else if (is_enum && !has_catch_all) {
        for (const std::string_view variant : decls_[subject.decl].variants) {
            if (std::find(seen_variants.begin(), seen_variants.end(), variant) ==
                seen_variants.end()) {
                missing += (missing.empty() ? "" : ", ") + std::string(variant);
            }
        }
    }
    if (!missing.empty()) {
        error(codes::non_exhaustive_match, node.span, "`match` does not cover every case")
            .note("not covered: " + missing);
    }
    return result == no_type ? types_.error() : result;
}

// -------------------------------------------------------------- index notation

void Checker::bind_index_domain(IndexVar& var,
                                const Shape& domain,
                                SourceSpan use_span,
                                std::string_view tensor_text) {
    if (!var.has_domain) {
        var.has_domain = true;
        var.domain = domain;
        var.bound_by = std::string(tensor_text);
        return;
    }
    if (types_.equal(var.domain, domain, env_->solver)) {
        return;
    }
    const std::string name = (var.is_pack ? "*" : "") + std::string(var.name);
    error(codes::contraction_mismatch, var.decl_span, "incompatible contraction dimensions")
        .primary_label(std::string(var.is_output ? "output" : "reduction") + " index `" + name +
                       "` has incompatible domains")
        .label(use_span, "used here with a different extent")
        .note("`" + var.bound_by + "` binds `" + name + "` to " + types_.to_string(var.domain))
        .note("`" + std::string(tensor_text) + "` binds `" + name + "` to " +
              types_.to_string(domain))
        .help("use one shared dimension parameter or add a constraint that proves " +
              types_.to_string(var.domain) + " == " + types_.to_string(domain));
}

TypeId Checker::check_reduction(const ast::Expr& node, const ast::ReductionExpr& reduction) {
    const ast::ExprId self = current_expr_;
    std::vector<IndexVar> scope;
    if (!reduction.indices.empty()) {
        const SourceSpan head{
            node.span.file, node.span.begin, reduction.indices.back().span.end + 1};
        for (const ast::Name& index : reduction.indices) {
            const bool is_duplicate =
                std::any_of(scope.begin(), scope.end(), [&](const IndexVar& v) {
                    return v.name == index.text;
                });
            if (is_duplicate) {
                error(codes::duplicate_index,
                      index.span,
                      "index `" + std::string(index.text) + "` is listed more than once");
                continue;
            }
            scope.push_back({index.text, head, false, false, false, {}, {}});
        }
    }
    env_->index_scopes.push_back(std::move(scope));
    const TypeId body = check_expr(reduction.body);
    const std::vector<IndexVar> vars = std::move(env_->index_scopes.back());
    env_->index_scopes.pop_back();
    for (const IndexVar& var : vars) {
        facts(self).index_domains.push_back(var.domain);
    }

    std::optional<DType> accumulator;
    if (reduction.accumulator != ast::no_id) {
        accumulator = eval_dtype(reduction.accumulator);
    }
    const TypeData& data = types_.get(body);
    if (data.kind == TypeKind::Error) {
        return body;
    }
    for (const IndexVar& var : vars) {
        if (!var.has_domain) {
            error(codes::unused_reduction_index,
                  var.decl_span,
                  "reduction index `" + std::string(var.name) + "` does not index any tensor")
                .note("an index gets its range from the tensor axes it indexes");
            return types_.error();
        }
    }
    if (data.kind != TypeKind::Scalar) {
        error(codes::invalid_index_expression,
              ast().expr(reduction.body).span,
              "the reduced expression must be a scalar, found `" + str(body) + "`")
            .help("index every tensor, for example `a[i, j]`");
        return types_.error();
    }

    const std::string kind(ast::reduction_kind_spelling(reduction.kind));
    const bool is_logical =
        reduction.kind == ast::ReductionKind::Any || reduction.kind == ast::ReductionKind::All;
    const bool is_bool = !data.dtype.is_var && data.dtype.scalar == ScalarKind::Bool;
    const bool may_be_bool =
        is_bool || (data.dtype.is_var && types_.class_of(data.dtype) == DTypeClass::Any);
    if (is_logical ? !is_bool : may_be_bool) {
        error(codes::invalid_operand,
              ast().expr(reduction.body).span,
              "`" + kind + "` reduces " + (is_logical ? "`bool`" : "numeric") + " values, found `" +
                  str(body) + "`");
        return types_.error();
    }
    if (accumulator && *accumulator != data.dtype) {
        error(codes::dtype_mismatch,
              ast().expr(reduction.body).span,
              "reduced expression does not have the accumulator dtype")
            .note("accumulator dtype is " + types_.to_string(*accumulator))
            .note("expression has dtype " + types_.to_string(data.dtype))
            .help("convert explicitly: `" + kind + "<" + types_.to_string(*accumulator) +
                  ">[...] cast<" + types_.to_string(*accumulator) + ">(...)`");
        return types_.error();
    }
    return body;
}

TypeId Checker::check_element_access(const ast::Expr& node,
                                     const ast::IndexExpr& index,
                                     const TypeData& tensor) {
    const std::string_view tensor_text = text(ast().expr(index.base).span);
    std::size_t pack_count = 0;
    for (const ast::IndexComponent& component : index.components) {
        pack_count += component.kind == ast::IndexKind::Pack ? 1 : 0;
    }
    const std::size_t plain_count = index.components.size() - pack_count;
    if (pack_count > 1) {
        error(codes::pack_index_misuse, node.span, "at most one pack index may appear here");
        return types_.error();
    }
    if (pack_count == 0 ? tensor.shape.size() != plain_count : tensor.shape.size() < plain_count) {
        error(codes::rank_mismatch,
              node.span,
              "`" + std::string(tensor_text) + "` has " + std::to_string(tensor.shape.size()) +
                  " axes but is indexed with " + std::to_string(index.components.size()) +
                  " indices")
            .note("its type is `Tensor[" + types_.to_string(tensor.shape) + "; " +
                  types_.to_string(tensor.dtype) + "]`");
        return types_.error();
    }

    bool is_valid = true;
    std::size_t axis = 0;
    for (const ast::IndexComponent& component : index.components) {
        if (component.kind == ast::IndexKind::Pack) {
            const std::size_t width = tensor.shape.size() - plain_count;
            const Shape domain(tensor.shape.begin() + static_cast<std::ptrdiff_t>(axis),
                               tensor.shape.begin() + static_cast<std::ptrdiff_t>(axis + width));
            axis += width;
            IndexVar* var = find_index(component.pack.text);
            if (var == nullptr || !var->is_pack) {
                error(var == nullptr ? codes::unbound_index : codes::pack_index_misuse,
                      component.span,
                      "`*" + std::string(component.pack.text) + "` is not a pack index here")
                    .help("declare it as an output: `let y[*" + std::string(component.pack.text) +
                          ", ...] = ...`");
                is_valid = false;
                continue;
            }
            bind_index_domain(*var, domain, component.span, tensor_text);
            continue;
        }

        const ShapeElem& unit = tensor.shape[axis++];
        const auto* name = std::get_if<ast::NameExpr>(&ast().expr(component.value).data);
        IndexVar* var = name != nullptr ? find_index(name->name.text) : nullptr;
        if (unit.is_pack) {
            error(codes::pack_index_misuse,
                  component.span,
                  "shape pack `*" + std::string(dims_.symbol_name(unit.pack)) +
                      "` must be indexed with a pack index such as `*s`");
            is_valid = false;
            continue;
        }
        if (var != nullptr) {
            if (var->is_pack) {
                error(codes::pack_index_misuse,
                      component.span,
                      "pack index `" + std::string(var->name) + "` must be written `*" +
                          std::string(var->name) + "`");
                is_valid = false;
                continue;
            }
            bind_index_domain(*var, {unit}, component.span, tensor_text);
            continue;
        }
        if (name != nullptr && lookup(name->name) == no_entity) {
            auto& reported = env_->unbound_reported;
            if (std::find(reported.begin(), reported.end(), name->name.text) == reported.end()) {
                reported.push_back(name->name.text);
                const std::string index_name(name->name.text);
                error(
                    codes::unbound_index, component.span, "index `" + index_name + "` is not bound")
                    .note("Linnet does not perform implicit Einstein summation")
                    .help("reduce over it explicitly, for example `sum[" + index_name +
                          "] ...`, or make it an output index");
            }
            is_valid = false;
            continue;
        }
        // Anything else selects one fixed position along the axis.
        const TypeId position = check_expr(component.value);
        const TypeData& data = types_.get(position);
        const bool is_integer_value =
            data.kind == TypeKind::CompileInt ||
            (data.kind == TypeKind::Scalar && types_.class_of(data.dtype) == DTypeClass::Integer);
        if (data.kind != TypeKind::Error && !is_integer_value) {
            error(codes::invalid_index_expression,
                  component.span,
                  "an index must be an index variable or an integer, found `" + str(position) +
                      "`");
            is_valid = false;
        }
    }
    return is_valid ? types_.scalar(tensor.dtype) : types_.error();
}

std::optional<std::int64_t> Checker::constant_index(ast::ExprId id, std::string_view what) {
    const TypeId type = check_expr(id);
    const TypeData& data = types_.get(type);
    if (data.kind == TypeKind::Error) {
        return std::nullopt;
    }
    const auto value = data.kind == TypeKind::CompileInt ? data.value.constant() : std::nullopt;
    if (!value) {
        error(codes::invalid_slice,
              ast().expr(id).span,
              std::string(what) + " must be an integer constant");
    }
    return value;
}

TypeId
Checker::check_slicing(const ast::Expr& node, const ast::IndexExpr& index, const TypeData& tensor) {
    std::size_t ellipsis_count = 0;
    for (const ast::IndexComponent& component : index.components) {
        if (component.kind == ast::IndexKind::Pack) {
            error(codes::pack_index_misuse,
                  component.span,
                  "pack indices are only meaningful in index notation")
                .help("use `...` to keep the leading axes");
            return types_.error();
        }
        ellipsis_count += component.kind == ast::IndexKind::Ellipsis ? 1 : 0;
    }
    const std::size_t explicit_count = index.components.size() - ellipsis_count;
    if (ellipsis_count > 1 || explicit_count > tensor.shape.size()) {
        error(codes::rank_mismatch,
              node.span,
              ellipsis_count > 1 ? "`...` may appear only once"
                                 : "too many indices for a tensor with " +
                                       std::to_string(tensor.shape.size()) + " axes");
        return types_.error();
    }

    Shape result;
    std::size_t axis = 0;
    bool is_valid = true;
    for (const ast::IndexComponent& component : index.components) {
        if (component.kind == ast::IndexKind::Ellipsis) {
            const std::size_t kept = tensor.shape.size() - explicit_count;
            result.insert(result.end(),
                          tensor.shape.begin() + static_cast<std::ptrdiff_t>(axis),
                          tensor.shape.begin() + static_cast<std::ptrdiff_t>(axis + kept));
            axis += kept;
            continue;
        }
        const ShapeElem& unit = tensor.shape[axis++];
        if (unit.is_pack) {
            error(codes::pack_index_misuse,
                  component.span,
                  "cannot index into shape pack `*" + std::string(dims_.symbol_name(unit.pack)) +
                      "`")
                .help("place `...` so that this index applies to a named dimension");
            is_valid = false;
            continue;
        }
        if (component.kind == ast::IndexKind::Expr) {
            const TypeId position = check_expr(component.value);
            const TypeData& data = types_.get(position);
            if (data.kind == TypeKind::CompileInt) {
                if (env_->solver.prove(shape::Relation::Less, data.value, {})) {
                    error(
                        codes::invalid_slice, component.span, "negative indices are not supported");
                    is_valid = false;
                } else if (env_->solver.prove(
                               shape::Relation::GreaterEqual, data.value, unit.dim)) {
                    error(codes::invalid_slice,
                          component.span,
                          "index `" + str(data.value) + "` is out of range for dimension `" +
                              str(unit.dim) + "`");
                    is_valid = false;
                }
            } else if (data.kind != TypeKind::Error &&
                       !(data.kind == TypeKind::Scalar &&
                         types_.class_of(data.dtype) == DTypeClass::Integer)) {
                error(codes::invalid_index_expression,
                      component.span,
                      "an index must be an integer, found `" + str(position) + "`");
                is_valid = false;
            }
            continue; // the axis is removed
        }

        // Slice `start:stop:step` with Python semantics for the extent.
        shape::Poly start(0);
        shape::Poly stop = unit.dim;
        std::int64_t step = 1;
        const auto bound = [&](ast::ExprId id, shape::Poly& out) {
            if (id == ast::no_id) {
                return;
            }
            const TypeId type = check_expr(id);
            const TypeData& data = types_.get(type);
            if (data.kind == TypeKind::CompileInt &&
                env_->solver.prove(shape::Relation::GreaterEqual, data.value, {})) {
                out = data.value;
            } else if (data.kind != TypeKind::Error) {
                error(codes::invalid_slice,
                      ast().expr(id).span,
                      "slice bounds must be non-negative compile-time integers")
                    .note("a bound that depends on runtime data would make the shape dynamic");
                is_valid = false;
            }
        };
        bound(component.start, start);
        bound(component.stop, stop);
        if (component.step != ast::no_id) {
            const auto value = constant_index(component.step, "a slice step");
            if (value && *value <= 0) {
                error(codes::invalid_slice,
                      ast().expr(component.step).span,
                      "a slice step must be positive");
            }
            is_valid = is_valid && value && *value > 0;
            step = value.value_or(1);
        }
        if (!is_valid) {
            continue;
        }
        const shape::Poly clamped = env_->solver.simplify(dims_.min(stop, unit.dim));
        const shape::Poly count = env_->solver.simplify(
            dims_.floor_div(clamped - start + shape::Poly(step - 1), shape::Poly(step)));
        const bool is_non_negative = env_->solver.prove(shape::Relation::GreaterEqual, count, {});
        result.push_back(ShapeElem::of(is_non_negative ? count : dims_.max(count, {})));
    }
    if (!is_valid) {
        return types_.error();
    }
    const std::size_t kept = tensor.shape.size() - axis;
    result.insert(
        result.end(), tensor.shape.end() - static_cast<std::ptrdiff_t>(kept), tensor.shape.end());
    return result.empty() ? types_.scalar(tensor.dtype)
                          : types_.tensor(std::move(result), tensor.dtype);
}

TypeId Checker::check_index(const ast::Expr& node, const ast::IndexExpr& index) {
    // The indexed value is a value even when an index variable shares its
    // name, as in `b[*b, k, n]`.
    TypeId base = no_type;
    if (const auto* name = std::get_if<ast::NameExpr>(&ast().expr(index.base).data)) {
        if (const EntityId entity = lookup(name->name); entity != no_entity) {
            base = value_of_entity(entity, ast().expr(index.base).span);
            facts(index.base).type = base;
            facts(index.base).entity = entity;
        }
    }
    if (base == no_type) {
        base = check_expr(index.base);
    }
    const TypeData& data = types_.get(base);
    if (data.kind == TypeKind::Error) {
        return base;
    }
    if (data.kind == TypeKind::Array) {
        if (index.components.size() != 1 || index.components.front().kind != ast::IndexKind::Expr) {
            error(codes::invalid_index_expression,
                  node.span,
                  "a structural array takes exactly one constant index");
            return types_.error();
        }
        const auto position = constant_index(index.components.front().value, "an array index");
        if (!position) {
            return types_.error();
        }
        if (*position < 0 ||
            env_->solver.prove(shape::Relation::GreaterEqual, shape::Poly(*position), data.value)) {
            error(codes::invalid_slice,
                  index.components.front().span,
                  "index " + std::to_string(*position) + " is out of range for `" + str(base) +
                      "`");
            return types_.error();
        }
        return data.elements.front();
    }
    if (data.kind != TypeKind::Tensor) {
        error(codes::invalid_operand, node.span, "`" + str(base) + "` cannot be indexed");
        return types_.error();
    }
    const bool only_positions =
        std::all_of(index.components.begin(), index.components.end(), [](const auto& component) {
            return component.kind == ast::IndexKind::Expr || component.kind == ast::IndexKind::Pack;
        });
    if (!env_->index_scopes.empty() && only_positions) {
        return check_element_access(node, index, data);
    }
    // Outside index notation an unknown name is just an unknown name.
    return check_slicing(node, index, data);
}

// -------------------------------------------------------------------- patterns

void Checker::record_binding(const ast::Name& name, TypeId type) {
    if (!name.text.empty()) {
        result_.bindings.push_back(
            {name.span, env_->owner, std::string(name.text), str(type), !binding_is_annotated_});
    }
}

void Checker::bind_pattern(ast::PatternId id, TypeId type, bool is_mutable) {
    const ast::Pattern& pattern = ast().pattern(id);
    std::visit(
        Overloaded{
            [&](const ast::ErrorPattern&) {},
            [&](const ast::BindingPattern& binding) {
                if (binding.name.text != "_") {
                    record_binding(binding.name, type);
                    model_->bound[env_->module][id] = declare_local(binding.name, type, is_mutable);
                }
            },
            [&](const ast::TuplePattern& tuple) {
                const TypeData& data = types_.get(type);
                if (data.kind == TypeKind::Error) {
                    for (const ast::PatternId element : tuple.elements) {
                        bind_pattern(element, type, is_mutable);
                    }
                    return;
                }
                if (data.kind != TypeKind::Tuple || data.elements.size() != tuple.elements.size()) {
                    error(codes::invalid_pattern,
                          pattern.span,
                          "this pattern has " + std::to_string(tuple.elements.size()) +
                              " elements but the value has type `" + str(type) + "`");
                    for (const ast::PatternId element : tuple.elements) {
                        bind_pattern(element, types_.error(), is_mutable);
                    }
                    return;
                }
                for (std::size_t i = 0; i < tuple.elements.size(); ++i) {
                    bind_pattern(tuple.elements[i], data.elements[i], is_mutable);
                }
            },
            [&](const ast::SomePattern&) {
                error(codes::invalid_pattern,
                      pattern.span,
                      "`some(...)` may not match; use `match` to handle `none`");
            },
            [&](const ast::NonePattern&) {
                error(codes::invalid_pattern, pattern.span, "`none` binds nothing; use `match`");
            },
        },
        pattern.data);
}

// ------------------------------------------------------------------ statements

void Checker::check_body(const std::vector<ast::StmtId>& body, bool is_function_body) {
    for (const ast::StmtId id : body) {
        env_->unbound_reported.clear();
        check_stmt(id, !is_function_body);
    }
    if (!is_function_body || env_->result == no_type) {
        return;
    }
    const TypeKind kind = types_.kind(env_->result);
    const bool ends_with_return =
        !body.empty() && std::holds_alternative<ast::ReturnStmt>(ast().stmt(body.back()).data);
    if (kind != TypeKind::Unit && kind != TypeKind::Error && !ends_with_return) {
        const Entity& function = entities_[env_->function];
        error(codes::missing_return,
              function.span,
              "`" + std::string(function.name) + "` must end with a `return`")
            .note("it is declared to return `" + str(env_->result) + "`");
    }
}

void Checker::check_stmt(ast::StmtId id, bool in_static_for) {
    const ast::Stmt& node = ast().stmt(id);
    const auto checked_value = [&](ast::ExprId value, ast::TypeId annotation) {
        if (annotation == ast::no_id) {
            const TypeId type = check_expr(value);
            if (types_.kind(type) == TypeKind::Unit) {
                error(codes::type_mismatch, ast().expr(value).span, "this call returns no value");
                return types_.error();
            }
            return value == ast::no_id ? type : default_literals(type, ast().expr(value).span);
        }
        const TypeId declared = eval_type(annotation);
        if (value == ast::no_id) {
            return declared;
        }
        return coerce(check_expr(value, declared), declared, ast().expr(value).span, "this value");
    };

    std::visit(
        Overloaded{
            [&](const ast::ErrorStmt&) {},
            [&](const ast::LetStmt& let) {
                const TypeId type = checked_value(let.value, let.type);
                binding_is_annotated_ = let.type != ast::no_id;
                bind_pattern(let.pattern, type, false);
                binding_is_annotated_ = false;
            },
            [&](const ast::VarStmt& var) {
                const TypeId type = checked_value(var.value, var.type);
                binding_is_annotated_ = var.type != ast::no_id;
                record_binding(var.name, type);
                binding_is_annotated_ = false;
                facts_of_stmt(id).type = type;
                facts_of_stmt(id).entity = declare_local(var.name, type, true);
            },
            [&](const ast::ComprehensionStmt& comprehension) {
                std::vector<IndexVar> outputs;
                for (const ast::IndexOutput& output : comprehension.outputs) {
                    const bool is_duplicate =
                        std::any_of(outputs.begin(), outputs.end(), [&](const IndexVar& other) {
                            return other.name == output.name.text;
                        });
                    if (is_duplicate) {
                        error(codes::duplicate_index,
                              output.name.span,
                              "output index `" + std::string(output.name.text) +
                                  "` is listed more than once");
                        continue;
                    }
                    outputs.push_back(
                        {output.name.text, output.name.span, output.is_pack, true, false, {}, {}});
                }
                env_->index_scopes.push_back(std::move(outputs));
                const TypeId body = check_expr(comprehension.value);
                const std::vector<IndexVar> vars = std::move(env_->index_scopes.back());
                env_->index_scopes.pop_back();

                TypeId type = types_.error();
                const TypeData& data = types_.get(body);
                if (vars.empty()) {
                    error(codes::invalid_index_expression,
                          comprehension.target.span,
                          "a tensor comprehension needs at least one output index");
                } else if (data.kind == TypeKind::Scalar) {
                    Shape shape;
                    bool is_complete = true;
                    for (const IndexVar& var : vars) {
                        if (!var.has_domain) {
                            error(codes::index_without_domain,
                                  var.decl_span,
                                  "output index `" + std::string(var.name) +
                                      "` does not index any tensor")
                                .note("an index gets its range from the tensor axes it indexes");
                            is_complete = false;
                        }
                        shape.insert(shape.end(), var.domain.begin(), var.domain.end());
                    }
                    if (is_complete) {
                        type = types_.tensor(std::move(shape), data.dtype);
                    }
                } else if (data.kind != TypeKind::Error) {
                    error(codes::invalid_index_expression,
                          ast().expr(comprehension.value).span,
                          "each element must be a typed scalar, found `" + str(body) + "`")
                        .help(is_literal(data.kind) ? "use `fill` to build a constant tensor"
                                                    : "index every tensor, for example `a[i, j]`");
                }
                record_binding(comprehension.target, type);
                StmtFacts& stmt_facts = facts_of_stmt(id);
                stmt_facts.type = type;
                for (const IndexVar& var : vars) {
                    stmt_facts.output_domains.push_back(var.domain);
                }
                stmt_facts.entity = declare_local(comprehension.target, type, false);
            },
            [&](const ast::AssignStmt& assign) {
                const EntityId entity = lookup(assign.target);
                if (entity == no_entity) {
                    error(codes::unknown_symbol,
                          assign.target.span,
                          "cannot find `" + std::string(assign.target.text) + "` in this scope");
                    check_expr(assign.value);
                    return;
                }
                const Entity& target = entities_[entity];
                const bool is_own_state = target.kind == EntityKind::Member && target.is_state &&
                                          target.parent == env_->block;
                if (target.kind == EntityKind::Member && target.is_state && !is_own_state) {
                    error(codes::assign_immutable,
                          assign.target.span,
                          "cannot assign to `" + std::string(target.name) +
                              "` from outside its block")
                        .label(target.span, "declared here")
                        .help("only the functions of a block assign its `state` members");
                    check_expr(assign.value);
                    return;
                }
                if (!is_own_state && (target.kind != EntityKind::Local || !target.is_mutable)) {
                    error(codes::assign_immutable,
                          assign.target.span,
                          "cannot assign to `" + std::string(target.name) + "`")
                        .label(target.span, "not declared with `var`")
                        .help("only locals declared with `var` and `state` members can be "
                              "assigned");
                    check_expr(assign.value);
                    return;
                }
                if (is_own_state) {
                    entities_[entity].is_used = true;
                    facts_of_stmt(id).entity = entity;
                }
                const TypeId value = check_expr(assign.value, target.type);
                if (!types_.is_error(value) && !assignable(value, target.type)) {
                    error(codes::assign_changes_type,
                          ast().expr(assign.value).span,
                          "assignment would change the type of `" + std::string(target.name) + "`")
                        .note("`" + std::string(target.name) + "` has type `" + str(target.type) +
                              "`")
                        .note("the new value has type `" + str(value) + "`");
                } else {
                    coerce(value, target.type, ast().expr(assign.value).span, "this value");
                }
            },
            [&](const ast::ReturnStmt& ret) {
                if (in_static_for) {
                    error(codes::missing_return,
                          node.span,
                          "`return` is not allowed inside `static for`");
                }
                const TypeId expected = env_->result == no_type ? types_.error() : env_->result;
                const bool returns_unit = types_.kind(expected) == TypeKind::Unit;
                if (ret.value == ast::no_id) {
                    if (!returns_unit && !types_.is_error(expected)) {
                        error(codes::type_mismatch,
                              node.span,
                              "`return` needs a value of type `" + str(expected) + "`");
                    }
                    return;
                }
                if (returns_unit) {
                    check_expr(ret.value);
                    error(codes::type_mismatch,
                          ast().expr(ret.value).span,
                          "this function does not declare a result type");
                    return;
                }
                coerce(check_expr(ret.value, expected),
                       expected,
                       ast().expr(ret.value).span,
                       "returned value");
            },
            [&](const ast::StaticForStmt& loop) {
                const TypeId iterable = check_expr(loop.iterable);
                const TypeData& data = types_.get(iterable);
                TypeId element = types_.error();
                if (loop.range_end != ast::no_id) {
                    // `start..stop` over compile-time integers; the loop
                    // variable is an `i64` scalar of each iteration.
                    const TypeId stop = check_expr(loop.range_end);
                    for (const auto& [type, span] :
                         {std::pair{iterable, ast().expr(loop.iterable).span},
                          std::pair{stop, ast().expr(loop.range_end).span}}) {
                        const TypeKind kind = types_.kind(type);
                        if (kind != TypeKind::CompileInt && kind != TypeKind::Error) {
                            error(codes::invalid_static_for,
                                  span,
                                  "a `static for` range bound must be a compile-time integer, "
                                  "found `" +
                                      str(type) + "`");
                        }
                    }
                    element = types_.scalar(ScalarKind::I64);
                } else if (data.kind == TypeKind::Array) {
                    element = data.elements.front();
                } else if (data.kind != TypeKind::Error) {
                    error(codes::invalid_static_for,
                          ast().expr(loop.iterable).span,
                          "`static for` iterates over a structural array, found `" + str(iterable) +
                              "`")
                        .note("the iteration count must be known at compile time");
                }
                env_->scopes.emplace_back();
                bind_pattern(loop.pattern, element, false);
                check_body(loop.body, false);
                env_->scopes.pop_back();
            },
        },
        node.data);
}

} // namespace linnet::sema
