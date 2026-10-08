#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/sema/types.hpp"
#include "linnet/shape/dim.hpp"
#include "linnet/shape/solver.hpp"
#include "linnet/source/source_manager.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// The typed model that semantic analysis produces: every declaration as an
// entity with its resolved signature, and for every expression of every
// module its type together with what its names, calls, and indices resolved
// to. Together with the syntax trees this is the typed HIR that later stages
// consume; nothing has to be re-derived from source.

namespace linnet::sema {

inline constexpr EntityId no_entity = 0xFFFFFFFFU;

// Names of the implicit, non-shadowable language prelude.
bool is_prelude_name(std::string_view name);
// The prelude's types and functions (dtype names aside), as source spells them.
std::span<const std::string_view> prelude_names();

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
    bool is_buffer = false;       // Member declared with `buffer`
    bool is_state = false;        // Member declared with `state`
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

// Binds `generic` to `value` in `substitution`.
void bind(Substitution& substitution, const GenericInfo& generic, const GenericValue& value);
// What a nominal type's arguments (a block's `<...>`) bind its
// declaration's generics to.
Substitution substitution_of(const DeclInfo& decl, const TypeData& nominal);

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

// Facts about one expression, keyed by its id within its module.
struct ExprFacts {
    TypeId type = no_type;
    // NameExpr: the entity named. MemberExpr: the member or module item.
    // CallExpr: the callee function, or no_entity for a prelude function.
    EntityId entity = no_entity;
    std::string builtin;                       // CallExpr on a prelude function
    Substitution substitution;                 // CallExpr: generic bindings
    std::vector<std::uint32_t> argument_slots; // CallExpr: parameter index per argument
    std::uint32_t field = 0;                   // MemberExpr on a struct: field index
    bool is_field = false;
    std::vector<Shape> index_domains; // ReductionExpr: domain of each index, in order
};

// Facts about one statement.
struct StmtFacts {
    TypeId type = no_type;             // the bound value's type
    std::vector<Shape> output_domains; // ComprehensionStmt: domain of each output index
    EntityId entity = no_entity;       // VarStmt / ComprehensionStmt: the local created;
                                       // AssignStmt: the state member assigned
};

struct Model {
    shape::DimContext dims;
    TypeStore types{dims};
    std::vector<std::string> module_paths; // `a.b.c` per module index
    std::deque<std::string> names;         // owned names of entities without source
    std::deque<Entity> entities;
    std::map<EntityId, DeclInfo> decls;
    std::vector<std::unordered_map<ast::ExprId, ExprFacts>> exprs;   // per module
    std::vector<std::unordered_map<ast::StmtId, StmtFacts>> stmts;   // per module
    std::vector<std::unordered_map<ast::PatternId, EntityId>> bound; // BindingPattern -> local

    const ExprFacts& expr(std::uint32_t module, ast::ExprId id) const;
    const StmtFacts& stmt(std::uint32_t module, ast::StmtId id) const;
};

// The members of `block` in declaration order, with their names.
std::vector<std::pair<EntityId, std::string_view>> block_members(const Model& model,
                                                                 EntityId block);

// A tensor of a block's manifest, as `walk_manifest` finds it.
struct ManifestTensor {
    std::string path;                       // `layers[*].attention.q.weight`
    const Entity* member = nullptr;         // its `param`, `buffer` or `state`
    const TypeData* tensor = nullptr;       // its type, the block's generics bound
    const std::vector<shape::Poly>* repeat; // each `[*]`'s array length, outermost first
    bool is_optional = false;               // declared optional, or in an optional `sub`
};

// Visits every tensor member of `root` and of the blocks it holds, through
// arrays and optional subs, in declaration order. A block inside itself has
// no finite manifest and is not entered again (the cycle is an error).
void walk_manifest(const Model& model,
                   TypeStore& types,
                   EntityId root,
                   const std::function<void(const ManifestTensor&)>& visit);

} // namespace linnet::sema
