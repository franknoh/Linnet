#pragma once

// Internal definition of the semantic checker, shared by its implementation
// files. Not part of the public interface.

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/sema/types.hpp"
#include "linnet/shape/solver.hpp"

#include <deque>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace linnet::sema {

inline constexpr EntityId no_entity = 0xFFFFFFFFU;

// Names of the implicit, non-shadowable language prelude.
bool is_prelude_name(std::string_view name);

enum class EntityKind : std::uint8_t {
    Module, // a module imported as a namespace
    Const,
    TypeAlias,
    Struct,
    Enum,
    Function, // fn, op, entry
    Block,
    Member, // param, buffer, sub
    GenericDim,
    GenericPack,
    GenericDType,
    Local, // parameters, let/var bindings, pattern bindings
};

enum class ResolveState : std::uint8_t { Pending, Running, Done };

struct Entity {
    EntityKind kind = EntityKind::Local;
    std::string_view name;
    SourceSpan span; // the declared name
    std::uint32_t module = 0;
    ast::ItemId item = ast::no_id;
    EntityId parent = no_entity; // enclosing block of members and methods
    bool is_pub = false;
    bool is_mutable = false;      // Local declared with `var`
    bool is_parameter = false;    // Local that is a function parameter
    bool is_used = false;         // referenced at least once
    shape::SymbolId symbol = 0;   // GenericDim, GenericPack
    DTypeVarId dtype_var = 0;     // GenericDType
    std::uint32_t module_ref = 0; // Module: index of the module it names
    TypeId type = no_type;        // Local, Member, Const (once resolved)
    ResolveState state = ResolveState::Pending;
};

using Scope = std::unordered_map<std::string_view, EntityId>;

enum class GenericKind : std::uint8_t { Dim, Pack, DType };

struct GenericInfo {
    std::string_view name;
    GenericKind kind = GenericKind::Dim;
    EntityId entity = no_entity;
    shape::SymbolId symbol = 0;
    DTypeVarId dtype_var = 0;
    DTypeClass constraint = DTypeClass::Any;
    std::optional<GenericValue> default_value;
};

struct ParamInfo {
    std::string_view name;
    SourceSpan span;
    TypeId type = no_type;
    bool has_default = false;
};

struct ConstraintInfo {
    shape::Relation relation;
    shape::Poly lhs;
    shape::Poly rhs;
    SourceSpan span;
};

struct FieldInfo {
    std::string_view name;
    TypeId type = no_type;
};

// Resolved facts about a declaration, keyed by its entity.
struct DeclInfo {
    std::vector<GenericInfo> generics;
    Scope scope; // generics; for blocks also members and methods
    // Functions
    std::vector<ParamInfo> params;
    std::vector<EntityId> param_entities;
    TypeId result = no_type;
    std::vector<ConstraintInfo> constraints;
    // Structs, enums, aliases
    std::vector<FieldInfo> fields;
    std::vector<std::string_view> variants;
    TypeId aliased = no_type;
};

// An index variable of tensor index notation. A plain index ranges over one
// dimension; a pack index over a run of shape units.
struct IndexVar {
    std::string_view name;
    SourceSpan decl_span;
    bool is_pack = false;
    bool is_output = false;
    bool has_domain = false;
    Shape domain;
    std::string bound_by; // text of the tensor that first fixed the domain
};

struct PendingEdge {
    EntityId caller;
    EntityId callee;
    SourceSpan span;
};

class Checker {
public:
    Checker(const SourceManager& sources,
            std::span<const ast::Ast* const> modules,
            DiagnosticSink& sink,
            const ImportTable* imports);

    AnalysisResult run();

private:
    // Reports on destruction, so call sites can chain notes and help.
    class Report {
    public:
        Report(Checker& checker,
               const char* code,
               SourceSpan span,
               std::string message,
               Severity severity = Severity::Error);
        Report(const Report&) = delete;
        Report& operator=(const Report&) = delete;
        ~Report();
        Report& note(std::string text);
        Report& help(std::string text);
        Report& label(SourceSpan span, std::string text);
        Report& primary_label(std::string text);

    private:
        Checker& checker_;
        Diagnostic diagnostic_;
    };
    Report error(const char* code, SourceSpan span, std::string message);
    Report warning(const char* code, SourceSpan span, std::string message);

    // Where names are currently being resolved.
    struct Env {
        std::uint32_t module = 0;
        EntityId function = no_entity;
        std::vector<Scope> scopes; // innermost last; below them: block, module
        EntityId block = no_entity;
        shape::Solver solver;
        TypeId result = no_type;
        std::vector<std::vector<IndexVar>> index_scopes;
        std::string owner;
        std::vector<std::string_view> unbound_reported; // per statement, to report once
    };
    class EnvGuard;

    const ast::Ast& ast() const { return *modules_[env_->module]; }
    std::string_view text(SourceSpan span) const { return sources_.text(span); }
    std::string str(TypeId type) const { return types_.to_string(type); }
    std::string str(const shape::Poly& poly) const { return dims_.to_string(poly); }

    // ------------------------------------------------------------ declarations
    void collect_module(std::uint32_t module);
    EntityId add_entity(Entity entity);
    void declare(Scope& scope, EntityId entity, const char* duplicate_code);
    bool check_not_prelude(std::string_view name, SourceSpan span);
    void resolve_imports(std::uint32_t module);
    void report_import_cycles();
    std::optional<std::uint32_t> find_module(const std::vector<ast::Name>& path) const;

    void resolve(EntityId entity);
    void resolve_generics(const std::vector<ast::GenericParam>& params, DeclInfo& info);
    void resolve_function(EntityId entity);
    void resolve_block(EntityId entity);
    void resolve_member(EntityId entity);
    void resolve_struct(EntityId entity);
    void resolve_enum(EntityId entity);
    void resolve_alias(EntityId entity);
    void resolve_const(EntityId entity);
    Env make_env(EntityId entity);
    void check_function_body(EntityId entity);
    void report_recursion();

    // ------------------------------------------------------------------- names
    EntityId lookup(std::string_view name);
    void report_unused();
    EntityId lookup_in_module(std::uint32_t module, const ast::Name& name);
    IndexVar* find_index(std::string_view name);
    EntityId declare_local(const ast::Name& name, TypeId type, bool is_mutable);

    // ------------------------------------------------------------------- types
    TypeId eval_type(ast::TypeId id);
    TypeId eval_named_type(const ast::NamedType& named, SourceSpan span);
    std::optional<DType> eval_dtype(ast::TypeId id);
    shape::Poly eval_dim(ast::ExprId id, bool check_divisors = true);
    shape::Poly dim_of_entity(EntityId entity, SourceSpan span);
    std::optional<GenericValue>
    eval_generic_arg(const ast::GenericArg& arg, const GenericInfo& param, SourceSpan span);
    std::optional<Substitution> bind_generic_args(const DeclInfo& info,
                                                  const std::vector<ast::GenericArg>& args,
                                                  SourceSpan span,
                                                  std::vector<GenericValue>* values);
    void check_dimension(const shape::Poly& dim, SourceSpan span);

    // ------------------------------------------------------------- expressions
    TypeId check_expr(ast::ExprId id, TypeId expected = no_type);
    TypeId check_expr_inner(ast::ExprId id, TypeId expected);
    TypeId check_literal(const ast::Expr& node, const ast::LiteralExpr& literal);
    TypeId check_name(const ast::Expr& node, const ast::NameExpr& name);
    TypeId value_of_entity(EntityId entity, SourceSpan span);
    TypeId check_unary(const ast::Expr& node, const ast::UnaryExpr& unary);
    TypeId check_binary(const ast::Expr& node, const ast::BinaryExpr& binary);
    TypeId check_member(const ast::Expr& node, const ast::MemberExpr& member);
    TypeId check_if(const ast::Expr& node, const ast::IfExpr& conditional, TypeId expected);
    TypeId check_match(const ast::Expr& node, const ast::MatchExpr& match, TypeId expected);
    TypeId check_reduction(const ast::Expr& node, const ast::ReductionExpr& reduction);
    TypeId check_index(const ast::Expr& node, const ast::IndexExpr& index);
    TypeId check_element_access(const ast::Expr& node,
                                const ast::IndexExpr& index,
                                const TypeData& tensor);
    TypeId
    check_slicing(const ast::Expr& node, const ast::IndexExpr& index, const TypeData& tensor);
    void bind_index_domain(IndexVar& var,
                           const Shape& domain,
                           SourceSpan use_span,
                           std::string_view tensor_text);

    TypeId check_call(const ast::Expr& node, const ast::CallExpr& call);
    TypeId check_user_call(const ast::Expr& node,
                           const ast::CallExpr& call,
                           EntityId callee,
                           const Substitution& receiver);
    TypeId
    check_builtin_call(const ast::Expr& node, const ast::CallExpr& call, std::string_view name);

    // Elementwise combination of two operand types; `for_comparison` yields bool.
    TypeId
    elementwise(TypeId lhs, TypeId rhs, SourceSpan span, bool needs_numeric, bool for_comparison);
    Substitution substitution_of(const TypeData& nominal);
    std::optional<std::int64_t> constant_index(ast::ExprId id, std::string_view what);
    std::optional<Shape> broadcast(const Shape& a, const Shape& b, SourceSpan span);
    std::optional<DType> numeric_dtype(TypeId type) const;

    // Whether a value of type `actual` may be used where `expected` is required.
    bool assignable(TypeId actual, TypeId expected);
    TypeId coerce(TypeId actual, TypeId expected, SourceSpan span, std::string_view what);
    void report_mismatch(TypeId actual, TypeId expected, SourceSpan span, std::string_view what);
    TypeId default_literals(TypeId type, SourceSpan span);
    TypeId join_branches(TypeId a, TypeId b, SourceSpan span);
    bool literal_fits(const TypeData& literal, DType dtype, SourceSpan span);

    // Generic inference.
    struct DeferredLiteral {
        TypeId literal;
        DType dtype;
        SourceSpan span;
    };
    struct Inference {
        const DeclInfo* callee = nullptr;
        Substitution bound;
        std::vector<std::pair<shape::Poly, shape::Poly>> deferred_dims;
        std::vector<DeferredLiteral> deferred_literals;
        SourceSpan span; // the argument being unified
        std::string failure;
    };
    bool mentions_unbound(const Inference& inference, const shape::Poly& poly) const;
    bool unify_dim(Inference& inference, const shape::Poly& param, const shape::Poly& arg);
    bool unify(Inference& inference, TypeId param, TypeId arg);
    bool unify_shape(Inference& inference, const Shape& param, const Shape& arg);
    bool unify_dtype(Inference& inference, DType param, DType arg);
    bool unify_value(Inference& inference, const GenericValue& param, const GenericValue& arg);

    // -------------------------------------------------------------- statements
    void check_body(const std::vector<ast::StmtId>& body, bool is_function_body);
    void check_stmt(ast::StmtId id, bool in_static_for);
    void bind_pattern(ast::PatternId id, TypeId type, bool is_mutable);
    void record_binding(const ast::Name& name, TypeId type);

    const SourceManager& sources_;
    std::vector<const ast::Ast*> modules_;
    DiagnosticSink& sink_;

    shape::DimContext dims_;
    TypeStore types_;
    std::deque<Entity> entities_;
    std::map<EntityId, DeclInfo> decls_;
    std::vector<Scope> module_scopes_;
    std::vector<EntityId> functions_;
    std::vector<PendingEdge> call_edges_;
    std::vector<PendingEdge> import_edges_; // between module indices
    const ImportTable* imports_ = nullptr;
    // Per module: names that a failed import would have introduced. Uses of
    // them are consequences of that failure and are not reported again.
    std::vector<std::vector<std::string_view>> failed_imports_;
    struct ImportedName {
        std::uint32_t module;
        std::string_view name;
        SourceSpan span;
        bool is_used = false;
    };
    std::vector<ImportedName> imported_names_;
    Env* env_ = nullptr;
    AnalysisResult result_;
};

} // namespace linnet::sema
