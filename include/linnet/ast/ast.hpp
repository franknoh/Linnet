#pragma once

#include "linnet/source/source_manager.hpp"
#include "linnet/syntax/token.hpp"

#include <cstdint>
#include <limits>
#include <string_view>
#include <variant>
#include <vector>

// The syntax tree is purely syntactic: it records what was written, with a
// source span on every node, and carries no resolved names or inferred types.
// Nodes live in arenas owned by `Ast` and refer to each other by stable ids.
// Name text views point into SourceManager storage, which must outlive the Ast.

namespace linnet::ast {

using ExprId = std::uint32_t;
using TypeId = std::uint32_t;
using PatternId = std::uint32_t;
using StmtId = std::uint32_t;
using ItemId = std::uint32_t;

inline constexpr std::uint32_t no_id = std::numeric_limits<std::uint32_t>::max();

struct Name {
    std::string_view text;
    SourceSpan span;
};

// One generic argument or generic default: either a type or an expression.
// A bare path such as `H` is recorded as a named type; whether it denotes a
// type or a dimension is decided during semantic analysis.
struct GenericArg {
    TypeId type = no_id;
    ExprId expr = no_id;
};

// ---------------------------------------------------------------- expressions

enum class LiteralKind : std::uint8_t { Integer, Float, String, True, False };
enum class UnaryOp : std::uint8_t { Not, Negate, Plus };
enum class BinaryOp : std::uint8_t {
    Or,
    And,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder
};
enum class ReductionKind : std::uint8_t { Sum, Prod, Max, Min, Any, All };

std::string_view unary_op_spelling(UnaryOp op);
std::string_view binary_op_spelling(BinaryOp op);
std::string_view reduction_kind_spelling(ReductionKind kind);

struct ErrorExpr {};
struct LiteralExpr {
    LiteralKind kind;
};
struct NameExpr {
    Name name;
};
struct NoneExpr {};
struct SomeExpr {
    ExprId value;
};
struct ParenExpr {
    ExprId inner;
};
struct TupleExpr {
    std::vector<ExprId> elements;
};
struct ShapeExpr {
    std::vector<ExprId> dims;
};
struct UnaryExpr {
    UnaryOp op;
    ExprId operand;
};
struct BinaryExpr {
    BinaryOp op;
    ExprId lhs;
    ExprId rhs;
};
struct Argument {
    Name keyword; // empty text for positional arguments
    ExprId value;
};
struct CallExpr {
    ExprId callee;
    std::vector<GenericArg> generic_args;
    std::vector<Argument> args;
};

enum class IndexKind : std::uint8_t { Expr, Ellipsis, Slice, Pack };
struct IndexComponent {
    IndexKind kind = IndexKind::Expr;
    SourceSpan span;
    ExprId value = no_id; // Expr
    Name pack;            // Pack: `*name`
    ExprId start = no_id; // Slice parts; any may be absent
    ExprId stop = no_id;
    ExprId step = no_id;
};
struct IndexExpr {
    ExprId base;
    std::vector<IndexComponent> components;
};
struct MemberExpr {
    ExprId base;
    Name member;
};
struct ReductionExpr {
    ReductionKind kind;
    TypeId accumulator; // no_id unless written as `sum<T>[...]`
    std::vector<Name> indices;
    ExprId body;
};
struct IfExpr {
    ExprId condition;
    ExprId then_value;
    ExprId else_value;
};
struct MatchArm {
    PatternId pattern;
    ExprId value;
};
struct MatchExpr {
    ExprId scrutinee;
    std::vector<MatchArm> arms;
};

using ExprData = std::variant<ErrorExpr,
                              LiteralExpr,
                              NameExpr,
                              NoneExpr,
                              SomeExpr,
                              ParenExpr,
                              TupleExpr,
                              ShapeExpr,
                              UnaryExpr,
                              BinaryExpr,
                              CallExpr,
                              IndexExpr,
                              MemberExpr,
                              ReductionExpr,
                              IfExpr,
                              MatchExpr>;

struct Expr {
    SourceSpan span;
    ExprData data;
};

// ---------------------------------------------------------------------- types

struct ErrorType {};
struct NamedType {
    std::vector<Name> path; // scalar dtypes are single-segment named types
    std::vector<GenericArg> args;
};
struct ShapeElement {
    Name pack;          // non-empty text for `*S`
    ExprId dim = no_id; // otherwise a dimension expression
};
struct TensorType {
    std::vector<ShapeElement> shape;
    TypeId dtype;
};
struct TupleType {
    std::vector<TypeId> elements;
};
struct ArrayType {
    TypeId element;
    ExprId length;
};
struct OptionalType {
    TypeId inner;
};

using TypeData = std::variant<ErrorType, NamedType, TensorType, TupleType, ArrayType, OptionalType>;

struct Type {
    SourceSpan span;
    TypeData data;
};

// ------------------------------------------------------------------- patterns

struct ErrorPattern {};
struct BindingPattern {
    Name name;
};
struct TuplePattern {
    std::vector<PatternId> elements;
};
struct SomePattern {
    PatternId inner;
};
struct NonePattern {};

using PatternData =
    std::variant<ErrorPattern, BindingPattern, TuplePattern, SomePattern, NonePattern>;

struct Pattern {
    SourceSpan span;
    PatternData data;
};

// ----------------------------------------------------------------- statements

struct ErrorStmt {};
struct LetStmt {
    PatternId pattern;
    TypeId type; // optional annotation
    ExprId value;
};
struct IndexOutput {
    Name name;
    bool is_pack;
};
// `let c[m, n] = ...`
struct ComprehensionStmt {
    Name target;
    std::vector<IndexOutput> outputs;
    ExprId value;
};
struct VarStmt {
    Name name;
    TypeId type; // optional annotation
    ExprId value;
};
struct AssignStmt {
    Name target;
    ExprId value;
};
struct ReturnStmt {
    ExprId value; // optional
};
struct StaticForStmt {
    PatternId pattern;
    ExprId iterable;
    std::vector<StmtId> body;
};

using StmtData = std::
    variant<ErrorStmt, LetStmt, ComprehensionStmt, VarStmt, AssignStmt, ReturnStmt, StaticForStmt>;

struct Stmt {
    SourceSpan span;
    StmtData data;
};

// ---------------------------------------------------------------------- items

enum class ConstraintKind : std::uint8_t { Unknown, Dim, Shape, DType, Numeric, Integer, Float };

struct GenericParam {
    SourceSpan span;
    bool is_pack;
    Name name;
    Name constraint_name;
    ConstraintKind constraint;
    GenericArg default_value; // both ids are no_id when absent
};

struct Parameter {
    SourceSpan span;
    Name name;
    TypeId type;
    ExprId default_value; // optional
};

struct ErrorItem {};
struct ConstDecl {
    Name name;
    TypeId type; // optional
    ExprId value;
};
struct TypeAliasDecl {
    Name name;
    std::vector<GenericParam> generics;
    TypeId type;
};
struct FieldDecl {
    SourceSpan span;
    Name name;
    TypeId type;
};
struct StructDecl {
    Name name;
    std::vector<GenericParam> generics;
    std::vector<FieldDecl> fields;
};
struct EnumDecl {
    Name name;
    std::vector<GenericParam> generics;
    std::vector<Name> variants;
};

enum class FunctionKind : std::uint8_t { Fn, Op, Entry };
struct FunctionDecl {
    FunctionKind kind;
    Name name;
    std::vector<GenericParam> generics;
    std::vector<Parameter> parameters;
    TypeId return_type; // optional for fn and entry
    std::vector<ExprId> constraints;
    std::vector<StmtId> body;
};

struct BlockDecl {
    Name name;
    std::vector<GenericParam> generics;
    std::vector<ItemId> members;
};

// `param`, `buffer`, and `sub` declarations; valid only inside a block.
enum class MemberKind : std::uint8_t { Param, Buffer, Sub };
struct MemberDecl {
    MemberKind kind;
    Name name;
    TypeId type;
    ExprId default_value; // optional; only `param` may have one
};

using ItemData = std::variant<ErrorItem,
                              ConstDecl,
                              TypeAliasDecl,
                              StructDecl,
                              EnumDecl,
                              FunctionDecl,
                              BlockDecl,
                              MemberDecl>;

struct Item {
    SourceSpan span;
    bool is_pub;
    ItemData data;
};

struct ImportName {
    Name name;
    Name alias; // empty text when absent
};
struct UseDecl {
    SourceSpan span;
    std::vector<Name> path;
    std::vector<ImportName> names; // empty: the module itself is imported
    bool is_braced;
};

// ----------------------------------------------------------------------- file

class Ast {
public:
    FileId file = invalid_file_id;
    SourceSpan module_span; // the whole `module a.b` declaration; empty when missing
    std::vector<Name> module_path;
    std::vector<UseDecl> uses;
    std::vector<ItemId> items;
    std::vector<Comment> comments;

    const Expr& expr(ExprId id) const { return exprs_[id]; }
    const Type& type(TypeId id) const { return types_[id]; }
    const Pattern& pattern(PatternId id) const { return patterns_[id]; }
    const Stmt& stmt(StmtId id) const { return stmts_[id]; }
    const Item& item(ItemId id) const { return items_[id]; }

    ExprId add(Expr node);
    TypeId add(Type node);
    PatternId add(Pattern node);
    StmtId add(Stmt node);
    ItemId add(Item node);

    // Arena sizes, used by the parser to discard nodes of abandoned
    // speculative parses.
    struct Mark {
        std::size_t exprs, types, patterns, stmts, items;
    };
    Mark mark() const;
    void rewind(const Mark& mark);

private:
    std::vector<Expr> exprs_;
    std::vector<Type> types_;
    std::vector<Pattern> patterns_;
    std::vector<Stmt> stmts_;
    std::vector<Item> items_;
};

} // namespace linnet::ast
