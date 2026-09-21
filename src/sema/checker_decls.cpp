#include "linnet/diagnostic/codes.hpp"

#include "checker.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <utility>

namespace linnet::sema {

namespace {

template <typename... Visitors>
struct Overloaded : Visitors... {
    using Visitors::operator()...;
};
template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

constexpr auto prelude_names = std::to_array<std::string_view>({
    "Tensor", "Dim",     "Shape",   "DType",        "Numeric", "Integer", "Float",
    "cast",   "reshape", "permute", "broadcast_to", "concat",  "pad",     "iota",
    "fill",   "gather",  "scatter", "exp",          "log",     "sqrt",    "rsqrt",
    "sin",    "cos",     "tanh",    "abs",          "select",  "min",     "max",
});

std::string join_path(const std::vector<ast::Name>& path, std::size_t from = 0) {
    std::string text;
    for (std::size_t i = from; i < path.size(); ++i) {
        text += text.empty() ? "" : ".";
        text += path[i].text;
    }
    return text;
}

SourceSpan path_span(const std::vector<ast::Name>& path) {
    return {path.front().span.file, path.front().span.begin, path.back().span.end};
}

std::optional<shape::Relation> relation_of(ast::BinaryOp op) {
    switch (op) {
    case ast::BinaryOp::Equal:
        return shape::Relation::Equal;
    case ast::BinaryOp::NotEqual:
        return shape::Relation::NotEqual;
    case ast::BinaryOp::Less:
        return shape::Relation::Less;
    case ast::BinaryOp::LessEqual:
        return shape::Relation::LessEqual;
    case ast::BinaryOp::Greater:
        return shape::Relation::Greater;
    case ast::BinaryOp::GreaterEqual:
        return shape::Relation::GreaterEqual;
    default:
        return std::nullopt;
    }
}

} // namespace

bool is_prelude_name(std::string_view name) {
    return scalar_from_name(name).has_value() ||
           std::find(prelude_names.begin(), prelude_names.end(), name) != prelude_names.end();
}

// --------------------------------------------------------------------- reports

Checker::Report::Report(Checker& checker, const char* code, SourceSpan span, std::string message)
    : checker_(checker) {
    diagnostic_.code = code;
    diagnostic_.message = std::move(message);
    diagnostic_.primary.span = span;
}

Checker::Report::~Report() {
    if (diagnostic_.code == codes::unknown_symbol && checker_.env_ != nullptr) {
        const auto& failed = checker_.failed_imports_[checker_.env_->module];
        const std::string_view name = checker_.text(diagnostic_.primary.span);
        if (std::find(failed.begin(), failed.end(), name) != failed.end()) {
            return;
        }
    }
    checker_.sink_.report(std::move(diagnostic_));
}

Checker::Report& Checker::Report::note(std::string text) {
    diagnostic_.notes.push_back(std::move(text));
    return *this;
}

Checker::Report& Checker::Report::help(std::string text) {
    diagnostic_.help.push_back(std::move(text));
    return *this;
}

Checker::Report& Checker::Report::label(SourceSpan span, std::string text) {
    diagnostic_.secondary.push_back({span, std::move(text)});
    return *this;
}

Checker::Report& Checker::Report::primary_label(std::string text) {
    diagnostic_.primary.message = std::move(text);
    return *this;
}

Checker::Report Checker::error(const char* code, SourceSpan span, std::string message) {
    return Report(*this, code, span, std::move(message));
}

class Checker::EnvGuard {
public:
    EnvGuard(Checker& checker, Env& env) : checker_(checker), saved_(checker.env_) {
        checker_.env_ = &env;
    }
    ~EnvGuard() { checker_.env_ = saved_; }
    EnvGuard(const EnvGuard&) = delete;
    EnvGuard& operator=(const EnvGuard&) = delete;

private:
    Checker& checker_;
    Env* saved_;
};

// ----------------------------------------------------------------------- setup

Checker::Checker(const SourceManager& sources,
                 std::span<const ast::Ast* const> modules,
                 DiagnosticSink& sink,
                 const ImportTable* imports)
    : sources_(sources), modules_(modules.begin(), modules.end()), sink_(sink), types_(dims_),
      imports_(imports) {
    types_.set_entity_namer([this](EntityId id) { return std::string(entities_[id].name); });
}

AnalysisResult Checker::run() {
    module_scopes_.resize(modules_.size());
    failed_imports_.resize(modules_.size());
    for (std::uint32_t module = 0; module < modules_.size(); ++module) {
        collect_module(module);
    }
    for (std::uint32_t module = 0; module < modules_.size(); ++module) {
        resolve_imports(module);
    }
    report_import_cycles();
    // Entities created while resolving (generics, parameters) need no pass of
    // their own, so only the declarations collected so far are visited.
    const std::size_t declared = entities_.size();
    for (EntityId entity = 0; entity < declared; ++entity) {
        resolve(entity);
    }
    for (const EntityId function : functions_) {
        check_function_body(function);
    }
    report_recursion();
    return std::move(result_);
}

EntityId Checker::add_entity(Entity entity) {
    entities_.push_back(entity);
    return static_cast<EntityId>(entities_.size() - 1);
}

bool Checker::check_not_prelude(std::string_view name, SourceSpan span) {
    if (!is_prelude_name(name)) {
        return true;
    }
    error(codes::shadows_prelude, span, "`" + std::string(name) + "` is a prelude name")
        .note("prelude names cannot be redefined or shadowed");
    return false;
}

void Checker::declare(Scope& scope, EntityId entity, const char* duplicate_code) {
    const Entity& declared = entities_[entity];
    if (declared.name.empty() || !check_not_prelude(declared.name, declared.span)) {
        return;
    }
    const auto [existing, inserted] = scope.emplace(declared.name, entity);
    if (!inserted) {
        error(duplicate_code,
              declared.span,
              "`" + std::string(declared.name) + "` is defined more than once")
            .primary_label("redefined here")
            .label(entities_[existing->second].span, "first defined here");
    }
}

void Checker::collect_module(std::uint32_t module) {
    const ast::Ast& tree = *modules_[module];
    const std::function<void(ast::ItemId, Scope&, EntityId)> collect =
        [&](ast::ItemId id, Scope& scope, EntityId parent) {
            const ast::Item& item = tree.item(id);
            Entity entity;
            entity.module = module;
            entity.item = id;
            entity.parent = parent;
            entity.is_pub = item.is_pub;
            const ast::Name* name = nullptr;
            std::visit(Overloaded{
                           [&](const ast::ErrorItem&) {},
                           [&](const ast::ConstDecl& decl) {
                               entity.kind = EntityKind::Const;
                               name = &decl.name;
                           },
                           [&](const ast::TypeAliasDecl& decl) {
                               entity.kind = EntityKind::TypeAlias;
                               name = &decl.name;
                           },
                           [&](const ast::StructDecl& decl) {
                               entity.kind = EntityKind::Struct;
                               name = &decl.name;
                           },
                           [&](const ast::EnumDecl& decl) {
                               entity.kind = EntityKind::Enum;
                               name = &decl.name;
                           },
                           [&](const ast::FunctionDecl& decl) {
                               entity.kind = EntityKind::Function;
                               name = &decl.name;
                           },
                           [&](const ast::BlockDecl& decl) {
                               entity.kind = EntityKind::Block;
                               name = &decl.name;
                           },
                           [&](const ast::MemberDecl& decl) {
                               if (parent != no_entity) {
                                   entity.kind = EntityKind::Member;
                                   name = &decl.name;
                               }
                           },
                       },
                       item.data);
            if (name == nullptr) {
                return;
            }
            entity.name = name->text;
            entity.span = name->span;
            const EntityId created = add_entity(entity);
            declare(scope,
                    created,
                    parent == no_entity ? codes::duplicate_item : codes::duplicate_name);
            if (entity.kind == EntityKind::Function) {
                functions_.push_back(created);
            }
            if (const auto* block = std::get_if<ast::BlockDecl>(&item.data)) {
                Scope& members = decls_[created].scope;
                for (const ast::ItemId member : block->members) {
                    collect(member, members, created);
                }
            }
        };
    for (const ast::ItemId id : tree.items) {
        collect(id, module_scopes_[module], no_entity);
    }
}

std::optional<std::uint32_t> Checker::find_module(const std::vector<ast::Name>& path) const {
    const std::string wanted = join_path(path);
    // `crate.a.b` names the module declared as `a.b` or `<package>.a.b`.
    const bool is_crate = path.front().text == "crate" && path.size() > 1;
    const std::string relative = is_crate ? join_path(path, 1) : std::string();
    for (std::uint32_t module = 0; module < modules_.size(); ++module) {
        const std::string declared = join_path(modules_[module]->module_path);
        if (declared == wanted) {
            return module;
        }
        if (is_crate && (declared == relative || declared.ends_with("." + relative))) {
            return module;
        }
    }
    return std::nullopt;
}

EntityId Checker::lookup_in_module(std::uint32_t module, const ast::Name& name) {
    const Scope& scope = module_scopes_[module];
    const auto found = scope.find(name.text);
    if (found == scope.end()) {
        error(codes::unknown_symbol,
              name.span,
              "module `" + join_path(modules_[module]->module_path) + "` has no item `" +
                  std::string(name.text) + "`");
        return no_entity;
    }
    const Entity& entity = entities_[found->second];
    if (entity.module == module && !entity.is_pub && env_ != nullptr && env_->module != module) {
        error(codes::private_item, name.span, "`" + std::string(name.text) + "` is private")
            .label(entity.span, "declared here without `pub`");
        return no_entity;
    }
    return found->second;
}

void Checker::resolve_imports(std::uint32_t module) {
    Env env{module, no_entity, {}, no_entity, shape::Solver(dims_), no_type, {}, {}, {}};
    const EnvGuard guard(*this, env);
    for (std::uint32_t index = 0; index < modules_[module]->uses.size(); ++index) {
        const ast::UseDecl& use = modules_[module]->uses[index];
        if (use.path.empty() || use.path.front().text.empty()) {
            continue;
        }
        std::optional<std::uint32_t> target;
        if (imports_ != nullptr) {
            const auto found = imports_->find({module, index});
            if (found != imports_->end()) {
                target = found->second;
            }
        } else {
            target = find_module(use.path);
        }
        if (target) {
            import_edges_.push_back({module, *target, path_span(use.path)});
        }
        if (!target) {
            error(codes::unknown_module,
                  path_span(use.path),
                  "cannot find module `" + join_path(use.path) + "`");
            if (use.names.empty()) {
                failed_imports_[module].push_back(use.path.back().text);
            }
            for (const ast::ImportName& import : use.names) {
                failed_imports_[module].push_back(import.alias.text.empty() ? import.name.text
                                                                            : import.alias.text);
            }
            continue;
        }
        if (use.names.empty()) {
            Entity entity;
            entity.kind = EntityKind::Module;
            entity.name = use.path.back().text;
            entity.span = use.path.back().span;
            entity.module = module;
            entity.module_ref = *target;
            entity.state = ResolveState::Done;
            declare(module_scopes_[module], add_entity(entity), codes::duplicate_item);
            continue;
        }
        for (const ast::ImportName& import : use.names) {
            const EntityId entity = lookup_in_module(*target, import.name);
            const ast::Name& local = import.alias.text.empty() ? import.name : import.alias;
            if (entity == no_entity) {
                failed_imports_[module].push_back(local.text);
                continue;
            }
            const auto [existing, inserted] = module_scopes_[module].emplace(local.text, entity);
            if (!inserted && existing->second != entity) {
                error(codes::duplicate_item,
                      local.span,
                      "`" + std::string(local.text) + "` is defined more than once")
                    .label(entities_[existing->second].span, "first defined here");
            }
        }
    }
}

// ------------------------------------------------------------------ resolution

Checker::Env Checker::make_env(EntityId entity) {
    const Entity& target = entities_[entity];
    Env env{target.module, no_entity, {}, no_entity, shape::Solver(dims_), no_type, {}, {}, {}};
    env.block = target.kind == EntityKind::Block ? entity : target.parent;
    if (env.block != no_entity && env.block != entity) {
        resolve(env.block);
    }
    return env;
}

void Checker::resolve(EntityId entity) {
    Entity& target = entities_[entity];
    if (target.state == ResolveState::Done) {
        return;
    }
    if (target.state == ResolveState::Running) {
        error(codes::cyclic_definition,
              target.span,
              "`" + std::string(target.name) + "` is defined in terms of itself");
        target.type = types_.error();
        return;
    }
    target.state = ResolveState::Running;
    switch (target.kind) {
    case EntityKind::Function:
        resolve_function(entity);
        break;
    case EntityKind::Block:
        resolve_block(entity);
        break;
    case EntityKind::Member:
        resolve_member(entity);
        break;
    case EntityKind::Struct:
        resolve_struct(entity);
        break;
    case EntityKind::Enum:
        resolve_enum(entity);
        break;
    case EntityKind::TypeAlias:
        resolve_alias(entity);
        break;
    case EntityKind::Const:
        resolve_const(entity);
        break;
    default:
        break;
    }
    entities_[entity].state = ResolveState::Done;
}

void Checker::resolve_generics(const std::vector<ast::GenericParam>& params, DeclInfo& info) {
    Scope& scope = env_->scopes.back();
    for (const ast::GenericParam& param : params) {
        if (param.name.text.empty() || param.constraint == ast::ConstraintKind::Unknown) {
            continue;
        }
        GenericInfo generic;
        generic.name = param.name.text;
        Entity entity;
        entity.name = param.name.text;
        entity.span = param.name.span;
        entity.module = env_->module;
        entity.state = ResolveState::Done;

        const bool is_shape = param.constraint == ast::ConstraintKind::Shape;
        if (is_shape != param.is_pack) {
            error(codes::bad_generic_arguments,
                  param.span,
                  is_shape ? "a `Shape` parameter is a pack and must be written `*" +
                                 std::string(param.name.text) + ": Shape`"
                           : "only `Shape` parameters can be packs");
            continue;
        }
        switch (param.constraint) {
        case ast::ConstraintKind::Dim:
            generic.kind = GenericKind::Dim;
            generic.symbol = dims_.add_symbol(std::string(param.name.text), shape::SymbolKind::Dim);
            entity.kind = EntityKind::GenericDim;
            entity.symbol = generic.symbol;
            break;
        case ast::ConstraintKind::Shape:
            generic.kind = GenericKind::Pack;
            generic.symbol =
                dims_.add_symbol(std::string(param.name.text), shape::SymbolKind::Pack);
            entity.kind = EntityKind::GenericPack;
            entity.symbol = generic.symbol;
            break;
        default:
            generic.kind = GenericKind::DType;
            generic.constraint =
                param.constraint == ast::ConstraintKind::Numeric   ? DTypeClass::Numeric
                : param.constraint == ast::ConstraintKind::Integer ? DTypeClass::Integer
                : param.constraint == ast::ConstraintKind::Float   ? DTypeClass::Float
                                                                   : DTypeClass::Any;
            generic.dtype_var =
                types_.add_dtype_var(std::string(param.name.text), generic.constraint);
            entity.kind = EntityKind::GenericDType;
            entity.dtype_var = generic.dtype_var;
            break;
        }
        generic.entity = add_entity(entity);
        declare(scope, generic.entity, codes::duplicate_name);
        if (param.default_value.type != ast::no_id || param.default_value.expr != ast::no_id) {
            generic.default_value = eval_generic_arg(param.default_value, generic, param.span);
        }
        info.generics.push_back(std::move(generic));
    }
}

void Checker::resolve_function(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::FunctionDecl>(modules_[target.module]->item(target.item).data);
    Env env = make_env(entity);
    env.function = entity;
    env.scopes.emplace_back();
    const EnvGuard guard(*this, env);
    DeclInfo& info = decls_[entity];
    resolve_generics(decl.generics, info);

    // Constraints come first: they justify divisions in the signature.
    for (const ast::ExprId id : decl.constraints) {
        const ast::Expr& node = ast().expr(id);
        const auto* comparison = std::get_if<ast::BinaryExpr>(&node.data);
        const auto relation = comparison ? relation_of(comparison->op) : std::nullopt;
        if (!relation) {
            continue;
        }
        ConstraintInfo constraint{*relation,
                                  eval_dim(comparison->lhs, false),
                                  eval_dim(comparison->rhs, false),
                                  node.span};
        if (constraint.lhs.is_valid() && constraint.rhs.is_valid()) {
            env.solver.assume(constraint.relation, constraint.lhs, constraint.rhs);
            info.constraints.push_back(std::move(constraint));
        }
    }

    Scope seen;
    for (const ast::Parameter& param : decl.parameters) {
        const ParamInfo parameter{param.name.text,
                                  param.name.span,
                                  param.type == ast::no_id ? types_.error() : eval_type(param.type),
                                  param.default_value != ast::no_id};
        Entity local;
        local.kind = EntityKind::Local;
        local.name = param.name.text;
        local.span = param.name.span;
        local.module = env.module;
        local.type = parameter.type;
        local.state = ResolveState::Done;
        const EntityId created = add_entity(local);
        declare(seen, created, codes::duplicate_name);
        info.params.push_back(parameter);
        info.param_entities.push_back(created);
    }
    info.result = decl.return_type == ast::no_id ? types_.unit() : eval_type(decl.return_type);
    info.scope = env.scopes.front();
}

void Checker::resolve_block(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::BlockDecl>(modules_[target.module]->item(target.item).data);
    Env env = make_env(entity);
    // Generics share the block scope with members so that both are visible
    // to every method.
    DeclInfo& info = decls_[entity];
    env.scopes.push_back(std::move(info.scope));
    const EnvGuard guard(*this, env);
    resolve_generics(decl.generics, info);
    info.scope = std::move(env.scopes.back());
}

void Checker::resolve_member(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::MemberDecl>(modules_[target.module]->item(target.item).data);
    Env env = make_env(entity);
    const EnvGuard guard(*this, env);
    const TypeId type = decl.type == ast::no_id ? types_.error() : eval_type(decl.type);
    entities_[entity].type = type;

    const auto is_kind = [&](TypeId id, TypeKind kind) { return types_.kind(id) == kind; };
    const TypeId inner =
        is_kind(type, TypeKind::Optional) ? types_.get(type).elements.front() : type;
    const SourceSpan type_span = decl.type == ast::no_id ? target.span : ast().type(decl.type).span;
    if (decl.kind == ast::MemberKind::Sub) {
        const TypeId element =
            is_kind(type, TypeKind::Array) ? types_.get(type).elements.front() : type;
        if (!types_.is_error(type) && !is_kind(element, TypeKind::Block)) {
            error(codes::invalid_member_type,
                  type_span,
                  "a `sub` must be a block or an array of blocks, not `" + str(type) + "`");
        }
    } else if (!types_.is_error(type) && !is_kind(inner, TypeKind::Tensor)) {
        error(codes::invalid_member_type,
              type_span,
              "a `" + std::string(decl.kind == ast::MemberKind::Param ? "param" : "buffer") +
                  "` must be a tensor, not `" + str(type) + "`");
    }

    if (decl.default_value != ast::no_id) {
        const ast::Expr& value = ast().expr(decl.default_value);
        if (!std::holds_alternative<ast::NoneExpr>(value.data)) {
            error(codes::param_payload, value.span, "a `param` cannot have a value in source")
                .note("parameter data is bound externally; the only default is `none`, which "
                      "marks an optional parameter as possibly absent");
        } else if (!types_.is_error(type) && !is_kind(type, TypeKind::Optional)) {
            error(codes::invalid_member_type,
                  value.span,
                  "only an optional `param` can default to `none`")
                .help("declare the type as `" + str(type) + "?`");
        }
    }
}

void Checker::resolve_struct(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::StructDecl>(modules_[target.module]->item(target.item).data);
    Env env = make_env(entity);
    env.scopes.emplace_back();
    const EnvGuard guard(*this, env);
    DeclInfo& info = decls_[entity];
    resolve_generics(decl.generics, info);
    info.scope = env.scopes.front();
    for (const ast::FieldDecl& field : decl.fields) {
        const bool is_duplicate =
            std::any_of(info.fields.begin(), info.fields.end(), [&](const FieldInfo& other) {
                return other.name == field.name.text;
            });
        if (is_duplicate) {
            error(codes::duplicate_name,
                  field.name.span,
                  "field `" + std::string(field.name.text) + "` is defined more than once");
            continue;
        }
        info.fields.push_back(
            {field.name.text, field.type == ast::no_id ? types_.error() : eval_type(field.type)});
    }
}

void Checker::resolve_enum(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::EnumDecl>(modules_[target.module]->item(target.item).data);
    Env env = make_env(entity);
    env.scopes.emplace_back();
    const EnvGuard guard(*this, env);
    DeclInfo& info = decls_[entity];
    resolve_generics(decl.generics, info);
    info.scope = env.scopes.front();
    for (const ast::Name& variant : decl.variants) {
        if (std::find(info.variants.begin(), info.variants.end(), variant.text) !=
            info.variants.end()) {
            error(codes::duplicate_name,
                  variant.span,
                  "variant `" + std::string(variant.text) + "` is defined more than once");
            continue;
        }
        info.variants.push_back(variant.text);
    }
}

void Checker::resolve_alias(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl =
        std::get<ast::TypeAliasDecl>(modules_[target.module]->item(target.item).data);
    Env env = make_env(entity);
    env.scopes.emplace_back();
    const EnvGuard guard(*this, env);
    DeclInfo& info = decls_[entity];
    resolve_generics(decl.generics, info);
    info.scope = env.scopes.front();
    info.aliased = decl.type == ast::no_id ? types_.error() : eval_type(decl.type);
}

void Checker::resolve_const(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::ConstDecl>(modules_[target.module]->item(target.item).data);
    Env env = make_env(entity);
    env.owner = std::string(target.name);
    const EnvGuard guard(*this, env);
    TypeId type = types_.error();
    if (decl.type != ast::no_id) {
        const TypeId declared = eval_type(decl.type);
        if (decl.value != ast::no_id) {
            type = coerce(check_expr(decl.value, declared),
                          declared,
                          ast().expr(decl.value).span,
                          "constant value");
        }
    } else if (decl.value != ast::no_id) {
        // An unannotated constant stays a compile-time value, so that it can
        // serve as a dimension and adopt a dtype where it is used.
        type = check_expr(decl.value);
        const TypeKind kind = types_.kind(type);
        if (kind != TypeKind::CompileInt && kind != TypeKind::FloatLiteral) {
            type = default_literals(type, ast().expr(decl.value).span);
        }
    }
    entities_[entity].type = type;
}

// --------------------------------------------------------------- function body

void Checker::check_function_body(EntityId entity) {
    resolve(entity);
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::FunctionDecl>(modules_[target.module]->item(target.item).data);
    const DeclInfo& info = decls_[entity];

    Env env = make_env(entity);
    env.function = entity;
    env.result = info.result;
    env.owner = (target.parent == no_entity ? std::string()
                                            : std::string(entities_[target.parent].name) + ".") +
                std::string(target.name);
    env.scopes.push_back(info.scope);
    env.scopes.emplace_back();
    for (const ConstraintInfo& constraint : info.constraints) {
        env.solver.assume(constraint.relation, constraint.lhs, constraint.rhs);
    }
    const EnvGuard guard(*this, env);

    for (std::size_t i = 0; i < info.param_entities.size(); ++i) {
        const EntityId param = info.param_entities[i];
        if (!entities_[param].name.empty() && !is_prelude_name(entities_[param].name)) {
            env.scopes.back().emplace(entities_[param].name, param);
        }
        const ast::ExprId default_value = decl.parameters[i].default_value;
        if (default_value != ast::no_id) {
            coerce(check_expr(default_value, info.params[i].type),
                   info.params[i].type,
                   ast().expr(default_value).span,
                   "default value");
        }
    }
    env.scopes.emplace_back();
    check_body(decl.body, true);
}

void Checker::report_import_cycles() {
    std::map<std::uint32_t, std::vector<const PendingEdge*>> graph;
    for (const PendingEdge& edge : import_edges_) {
        graph[edge.caller].push_back(&edge);
    }
    enum class Mark : std::uint8_t { Unseen, Active, Done };
    std::vector<Mark> marks(modules_.size(), Mark::Unseen);
    const std::function<void(std::uint32_t)> visit = [&](std::uint32_t module) {
        marks[module] = Mark::Active;
        for (const PendingEdge* edge : graph[module]) {
            if (marks[edge->callee] == Mark::Active) {
                error(codes::import_cycle,
                      edge->span,
                      "this import makes the modules depend on "
                      "each other")
                    .note("import cycles are not supported; move the shared declarations into "
                          "a module that both can import");
            } else if (marks[edge->callee] == Mark::Unseen) {
                visit(edge->callee);
            }
        }
        marks[module] = Mark::Done;
    };
    for (std::uint32_t module = 0; module < modules_.size(); ++module) {
        if (marks[module] == Mark::Unseen) {
            visit(module);
        }
    }
}

void Checker::report_recursion() {
    std::map<EntityId, std::vector<const PendingEdge*>> graph;
    for (const PendingEdge& edge : call_edges_) {
        graph[edge.caller].push_back(&edge);
    }
    enum class Mark : std::uint8_t { Unseen, Active, Done };
    std::map<EntityId, Mark> marks;
    const std::function<void(EntityId)> visit = [&](EntityId node) {
        marks[node] = Mark::Active;
        for (const PendingEdge* edge : graph[node]) {
            const Mark mark = marks[edge->callee];
            if (mark == Mark::Active) {
                const bool is_direct = edge->callee == edge->caller;
                error(codes::recursion,
                      edge->span,
                      is_direct ? "`" + std::string(entities_[edge->callee].name) + "` calls itself"
                                : "this call makes `" + std::string(entities_[edge->callee].name) +
                                      "` indirectly recursive")
                    .note("recursion is not supported; every call graph must be acyclic");
            } else if (mark == Mark::Unseen) {
                visit(edge->callee);
            }
        }
        marks[node] = Mark::Done;
    };
    for (const EntityId function : functions_) {
        if (marks[function] == Mark::Unseen) {
            visit(function);
        }
    }
}

// ----------------------------------------------------------------------- names

EntityId Checker::lookup(std::string_view name) const {
    for (auto scope = env_->scopes.rbegin(); scope != env_->scopes.rend(); ++scope) {
        if (const auto found = scope->find(name); found != scope->end()) {
            return found->second;
        }
    }
    for (EntityId block = env_->block; block != no_entity; block = entities_[block].parent) {
        const Scope& scope = decls_.at(block).scope;
        if (const auto found = scope.find(name); found != scope.end()) {
            return found->second;
        }
    }
    const Scope& module_scope = module_scopes_[env_->module];
    const auto found = module_scope.find(name);
    return found == module_scope.end() ? no_entity : found->second;
}

IndexVar* Checker::find_index(std::string_view name) {
    for (auto scope = env_->index_scopes.rbegin(); scope != env_->index_scopes.rend(); ++scope) {
        for (IndexVar& var : *scope) {
            if (var.name == name) {
                return &var;
            }
        }
    }
    return nullptr;
}

EntityId Checker::declare_local(const ast::Name& name, TypeId type, bool is_mutable) {
    if (name.text.empty()) {
        return no_entity;
    }
    // A rejected name is still bound, as an error, so that its uses do not
    // produce further diagnostics.
    if (!check_not_prelude(name.text, name.span)) {
        type = types_.error();
    }
    Entity local;
    local.kind = EntityKind::Local;
    local.name = name.text;
    local.span = name.span;
    local.module = env_->module;
    local.type = type;
    local.is_mutable = is_mutable;
    local.state = ResolveState::Done;
    const EntityId created = add_entity(local);
    env_->scopes.back()[name.text] = created; // later bindings shadow earlier ones
    return created;
}

// ----------------------------------------------------------------------- types

void Checker::check_dimension(const shape::Poly& dim, SourceSpan span) {
    if (!dim.is_valid()) {
        return;
    }
    const auto upper = env_->solver.upper_bound(dim);
    if (upper && *upper < 0) {
        error(codes::negative_dimension, span, "dimension `" + str(dim) + "` is negative");
    }
}

shape::Poly Checker::dim_of_entity(EntityId entity, SourceSpan span) {
    resolve(entity);
    const Entity& target = entities_[entity];
    if (target.kind == EntityKind::GenericDim) {
        return dims_.symbol(target.symbol);
    }
    if ((target.kind == EntityKind::Const || target.kind == EntityKind::Local) &&
        target.type != no_type) {
        if (types_.kind(target.type) == TypeKind::CompileInt) {
            return types_.get(target.type).value;
        }
        if (types_.is_error(target.type)) {
            return shape::Poly::invalid();
        }
    }
    error(codes::not_compile_time,
          span,
          "`" + std::string(target.name) + "` is not a compile-time integer")
        .note("dimensions are built from integer literals, `Dim` parameters, and integer "
              "constants");
    return shape::Poly::invalid();
}

shape::Poly Checker::eval_dim(ast::ExprId id, bool check_divisors) {
    if (id == ast::no_id) {
        return shape::Poly::invalid();
    }
    const ast::Expr& node = ast().expr(id);
    const auto fail = [&](std::string message) {
        error(codes::not_compile_time, node.span, std::move(message));
        return shape::Poly::invalid();
    };
    if (std::holds_alternative<ast::ErrorExpr>(node.data)) {
        return shape::Poly::invalid();
    }
    if (const auto* literal = std::get_if<ast::LiteralExpr>(&node.data)) {
        if (literal->kind != ast::LiteralKind::Integer) {
            return fail("a dimension must be an integer");
        }
        const TypeId type = check_literal(node, *literal);
        return types_.is_error(type) ? shape::Poly::invalid() : types_.get(type).value;
    }
    if (const auto* name = std::get_if<ast::NameExpr>(&node.data)) {
        const EntityId entity = lookup(name->name.text);
        if (entity == no_entity) {
            error(codes::unknown_symbol,
                  node.span,
                  "cannot find `" + std::string(name->name.text) + "` in this scope");
            return shape::Poly::invalid();
        }
        return dim_of_entity(entity, node.span);
    }
    if (const auto* paren = std::get_if<ast::ParenExpr>(&node.data)) {
        return eval_dim(paren->inner, check_divisors);
    }
    if (const auto* unary = std::get_if<ast::UnaryExpr>(&node.data)) {
        if (unary->op == ast::UnaryOp::Not) {
            return fail("`!` is not a dimension operator");
        }
        const shape::Poly operand = eval_dim(unary->operand, check_divisors);
        return unary->op == ast::UnaryOp::Negate ? -operand : operand;
    }
    if (const auto* binary = std::get_if<ast::BinaryExpr>(&node.data)) {
        const shape::Poly lhs = eval_dim(binary->lhs, check_divisors);
        const shape::Poly rhs = eval_dim(binary->rhs, check_divisors);
        if (!lhs.is_valid() || !rhs.is_valid()) {
            return shape::Poly::invalid();
        }
        switch (binary->op) {
        case ast::BinaryOp::Add:
            return lhs + rhs;
        case ast::BinaryOp::Subtract:
            return lhs - rhs;
        case ast::BinaryOp::Multiply:
            return lhs * rhs;
        case ast::BinaryOp::Divide:
        case ast::BinaryOp::Remainder: {
            if (check_divisors && !env_->solver.prove(shape::Relation::Greater, rhs, {})) {
                error(codes::divisor_not_positive,
                      ast().expr(binary->rhs).span,
                      "cannot prove that the divisor `" + str(rhs) + "` is positive")
                    .help("add a constraint such as `where " + str(rhs) + " > 0`");
                return shape::Poly::invalid();
            }
            return binary->op == ast::BinaryOp::Divide ? dims_.floor_div(lhs, rhs)
                                                       : dims_.mod(lhs, rhs);
        }
        default:
            return fail("comparison and logical operators cannot appear in a dimension");
        }
    }
    if (const auto* call = std::get_if<ast::CallExpr>(&node.data)) {
        const auto* callee = std::get_if<ast::NameExpr>(&ast().expr(call->callee).data);
        const bool is_min = callee != nullptr && callee->name.text == "min";
        const bool is_max = callee != nullptr && callee->name.text == "max";
        if ((is_min || is_max) && call->args.size() == 2 && call->generic_args.empty()) {
            const shape::Poly a = eval_dim(call->args[0].value, check_divisors);
            const shape::Poly b = eval_dim(call->args[1].value, check_divisors);
            return is_min ? dims_.min(a, b) : dims_.max(a, b);
        }
    }
    return fail("this expression is not a compile-time dimension");
}

std::optional<DType> Checker::eval_dtype(ast::TypeId id) {
    const TypeId type = eval_type(id);
    if (types_.is_error(type)) {
        return std::nullopt;
    }
    if (types_.kind(type) != TypeKind::Scalar) {
        error(codes::wrong_symbol_kind,
              ast().type(id).span,
              "expected an element type such as `f32`, found `" + str(type) + "`");
        return std::nullopt;
    }
    return types_.get(type).dtype;
}

std::optional<GenericValue>
Checker::eval_generic_arg(const ast::GenericArg& arg, const GenericInfo& param, SourceSpan span) {
    GenericValue value;
    // A bare name parses as a type; for `Dim` and `Shape` parameters it names
    // a dimension or a pack instead.
    const ast::NamedType* bare_name = nullptr;
    if (arg.type != ast::no_id) {
        const auto* named = std::get_if<ast::NamedType>(&ast().type(arg.type).data);
        if (named != nullptr && named->path.size() == 1 && named->args.empty()) {
            bare_name = named;
        }
    }
    const auto unknown = [&](const ast::Name& name) {
        error(codes::unknown_symbol,
              name.span,
              "cannot find `" + std::string(name.text) + "` in this scope");
        return std::nullopt;
    };

    switch (param.kind) {
    case GenericKind::Dim:
        value.kind = GenericValue::Kind::Dim;
        if (bare_name != nullptr && !scalar_from_name(bare_name->path.front().text)) {
            const ast::Name& name = bare_name->path.front();
            const EntityId entity = lookup(name.text);
            if (entity == no_entity) {
                return unknown(name);
            }
            value.dim = dim_of_entity(entity, name.span);
        } else if (arg.expr != ast::no_id) {
            value.dim = eval_dim(arg.expr);
        } else {
            error(codes::bad_generic_arguments,
                  span,
                  "`" + std::string(param.name) + "` expects a dimension, not a type");
            return std::nullopt;
        }
        if (!value.dim.is_valid()) {
            return std::nullopt;
        }
        return value;
    case GenericKind::Pack:
        value.kind = GenericValue::Kind::Pack;
        if (bare_name != nullptr) {
            const ast::Name& name = bare_name->path.front();
            const EntityId entity = lookup(name.text);
            if (entity == no_entity) {
                return unknown(name);
            }
            if (entities_[entity].kind == EntityKind::GenericPack) {
                value.shape.push_back(ShapeElem::of_pack(entities_[entity].symbol));
                return value;
            }
        } else if (arg.expr != ast::no_id) {
            if (const auto* literal = std::get_if<ast::ShapeExpr>(&ast().expr(arg.expr).data)) {
                for (const ast::ExprId dim : literal->dims) {
                    shape::Poly poly = eval_dim(dim);
                    if (!poly.is_valid()) {
                        return std::nullopt;
                    }
                    value.shape.push_back(ShapeElem::of(std::move(poly)));
                }
                return value;
            }
        }
        error(codes::bad_generic_arguments,
              span,
              "`*" + std::string(param.name) +
                  "` expects a shape such as `[B, S]` or a shape pack");
        return std::nullopt;
    case GenericKind::DType: {
        value.kind = GenericValue::Kind::DType;
        if (arg.type == ast::no_id) {
            error(codes::bad_generic_arguments,
                  span,
                  "`" + std::string(param.name) + "` expects an element type");
            return std::nullopt;
        }
        const auto dtype = eval_dtype(arg.type);
        if (!dtype) {
            return std::nullopt;
        }
        if (!class_contains(param.constraint, types_.class_of(*dtype)) &&
            !(!dtype->is_var && class_contains(param.constraint, dtype->scalar))) {
            error(codes::dtype_constraint,
                  ast().type(arg.type).span,
                  "`" + types_.to_string(*dtype) + "` does not satisfy the constraint on `" +
                      std::string(param.name) + "`");
            return std::nullopt;
        }
        value.dtype = *dtype;
        return value;
    }
    }
    return std::nullopt;
}

std::optional<Substitution> Checker::bind_generic_args(const DeclInfo& info,
                                                       const std::vector<ast::GenericArg>& args,
                                                       SourceSpan span,
                                                       std::vector<GenericValue>* values) {
    if (args.size() > info.generics.size()) {
        error(codes::bad_generic_arguments,
              span,
              "expected at most " + std::to_string(info.generics.size()) +
                  " generic arguments, found " + std::to_string(args.size()));
        return std::nullopt;
    }
    Substitution substitution;
    for (std::size_t i = 0; i < info.generics.size(); ++i) {
        const GenericInfo& param = info.generics[i];
        std::optional<GenericValue> value;
        if (i < args.size()) {
            value = eval_generic_arg(args[i], param, span);
            if (!value) {
                return std::nullopt;
            }
        } else if (param.default_value) {
            value = types_.substitute(*param.default_value, substitution);
        } else {
            error(codes::bad_generic_arguments,
                  span,
                  "missing generic argument for `" + std::string(param.name) + "`");
            return std::nullopt;
        }
        switch (param.kind) {
        case GenericKind::Dim:
            substitution.dims[param.symbol] = value->dim;
            break;
        case GenericKind::Pack:
            substitution.packs[param.symbol] = value->shape;
            break;
        case GenericKind::DType:
            substitution.dtypes[param.dtype_var] = value->dtype;
            break;
        }
        if (values != nullptr) {
            values->push_back(std::move(*value));
        }
    }
    return substitution;
}

TypeId Checker::eval_named_type(const ast::NamedType& named, SourceSpan span) {
    if (named.path.empty() || named.path.front().text.empty()) {
        return types_.error();
    }
    EntityId entity = no_entity;
    const ast::Name& last = named.path.back();
    if (named.path.size() == 1) {
        if (const auto scalar = scalar_from_name(last.text)) {
            if (!named.args.empty()) {
                error(codes::bad_generic_arguments, span, "scalar types take no arguments");
            }
            return types_.scalar(*scalar);
        }
        entity = lookup(last.text);
    } else {
        const EntityId module =
            named.path.size() == 2 ? lookup(named.path.front().text) : no_entity;
        if (module == no_entity || entities_[module].kind != EntityKind::Module) {
            error(codes::unknown_symbol,
                  path_span(named.path),
                  "cannot find `" + join_path(named.path) + "`")
                .note("qualified names have the form `module.Item`, where the module was "
                      "brought in with `use`");
            return types_.error();
        }
        entity = lookup_in_module(entities_[module].module_ref, last);
        if (entity == no_entity) {
            return types_.error();
        }
    }
    if (entity == no_entity) {
        error(codes::unknown_symbol,
              last.span,
              "cannot find type `" + std::string(last.text) + "` in this scope");
        return types_.error();
    }

    resolve(entity);
    const Entity& target = entities_[entity];
    switch (target.kind) {
    case EntityKind::GenericDType:
        return types_.scalar(DType::variable(target.dtype_var));
    case EntityKind::TypeAlias: {
        const DeclInfo& info = decls_[entity];
        const auto substitution = bind_generic_args(info, named.args, span, nullptr);
        return substitution && info.aliased != no_type
                   ? types_.substitute(info.aliased, *substitution)
                   : types_.error();
    }
    case EntityKind::Struct:
    case EntityKind::Enum:
    case EntityKind::Block: {
        std::vector<GenericValue> values;
        if (!bind_generic_args(decls_[entity], named.args, span, &values)) {
            return types_.error();
        }
        const TypeKind kind = target.kind == EntityKind::Struct ? TypeKind::Struct
                              : target.kind == EntityKind::Enum ? TypeKind::Enum
                                                                : TypeKind::Block;
        return types_.nominal(kind, entity, std::move(values));
    }
    default:
        error(codes::wrong_symbol_kind, last.span, "`" + std::string(last.text) + "` is not a type")
            .label(target.span, "declared here");
        return types_.error();
    }
}

TypeId Checker::eval_type(ast::TypeId id) {
    const ast::Type& node = ast().type(id);
    return std::visit(
        Overloaded{
            [&](const ast::ErrorType&) { return types_.error(); },
            [&](const ast::NamedType& named) { return eval_named_type(named, node.span); },
            [&](const ast::TensorType& tensor) {
                Shape shape;
                bool is_valid = true;
                for (const ast::ShapeElement& element : tensor.shape) {
                    if (element.dim != ast::no_id) {
                        shape::Poly dim = eval_dim(element.dim);
                        check_dimension(dim, ast().expr(element.dim).span);
                        is_valid = is_valid && dim.is_valid();
                        shape.push_back(ShapeElem::of(std::move(dim)));
                        continue;
                    }
                    const EntityId pack = lookup(element.pack.text);
                    if (pack == no_entity || entities_[pack].kind != EntityKind::GenericPack) {
                        if (!element.pack.text.empty()) {
                            error(pack == no_entity ? codes::unknown_symbol
                                                    : codes::wrong_symbol_kind,
                                  element.pack.span,
                                  "`" + std::string(element.pack.text) +
                                      "` is not a shape pack in this scope")
                                .help("declare it as a generic parameter: `*" +
                                      std::string(element.pack.text) + ": Shape`");
                        }
                        is_valid = false;
                        continue;
                    }
                    shape.push_back(ShapeElem::of_pack(entities_[pack].symbol));
                }
                const auto dtype =
                    tensor.dtype == ast::no_id ? std::nullopt : eval_dtype(tensor.dtype);
                return is_valid && dtype ? types_.tensor(std::move(shape), *dtype) : types_.error();
            },
            [&](const ast::TupleType& tuple) {
                std::vector<TypeId> elements;
                elements.reserve(tuple.elements.size());
                for (const ast::TypeId element : tuple.elements) {
                    elements.push_back(eval_type(element));
                }
                return types_.tuple(std::move(elements));
            },
            [&](const ast::ArrayType& array) {
                const TypeId element = eval_type(array.element);
                shape::Poly length = eval_dim(array.length);
                if (array.length != ast::no_id) {
                    check_dimension(length, ast().expr(array.length).span);
                }
                return length.is_valid() ? types_.array(element, std::move(length))
                                         : types_.error();
            },
            [&](const ast::OptionalType& optional) {
                return types_.optional(eval_type(optional.inner));
            },
        },
        node.data);
}

AnalysisResult analyze(const SourceManager& sources,
                       std::span<const ast::Ast* const> modules,
                       DiagnosticSink& sink,
                       const ImportTable* imports) {
    return Checker(sources, modules, sink, imports).run();
}

} // namespace linnet::sema
