// Kernels: tile programs launched over a grid, reading tensor parameters
// with `load` and writing results with `store`, and the ops that name them.

#include "linnet/diagnostic/codes.hpp"

#include "checker.hpp"

#include <algorithm>
#include <string>

namespace linnet::sema {

namespace {

bool has_pack(const TypeData& data) {
    return std::ranges::any_of(data.shape, [](const ShapeElem& unit) { return unit.is_pack; });
}

} // namespace

// Parameters are tensors in memory or scalars, results tensors in memory,
// all of a known rank; the grid is one to three compile-time integers.
void Checker::check_kernel_signature(EntityId entity) {
    const Entity& target = entities_[entity];
    const auto& decl = std::get<ast::FunctionDecl>(modules_[target.module]->item(target.item).data);
    const DeclInfo& info = decls_[entity];
    if (target.parent != no_entity) {
        error(codes::invalid_kernel, target.span, "a kernel is declared at module level");
    }
    for (const ParamInfo& param : info.params) {
        const TypeData& data = types_.get(param.type);
        if (data.kind == TypeKind::Error || data.kind == TypeKind::Scalar) {
            continue;
        }
        if (data.kind != TypeKind::Tensor) {
            error(codes::invalid_kernel, param.span, "a kernel parameter is a tensor or a scalar")
                .note("`" + std::string(param.name) + "` is `" + str(param.type) + "`");
        } else if (has_pack(data)) {
            error(codes::invalid_kernel,
                  param.span,
                  "a kernel indexes its tensors axis by axis: no shape pack");
        }
    }
    for (const EntityId result : info.kernel_results) {
        const TypeData& data = types_.get(entities_[result].type);
        if (data.kind == TypeKind::Error) {
            continue;
        }
        if (data.kind != TypeKind::Tensor || has_pack(data)) {
            error(codes::invalid_kernel,
                  entities_[result].span,
                  "a kernel's result is a tensor of a known rank");
        }
    }
    if (decl.grid.empty() || decl.grid.size() > 3) {
        error(codes::invalid_kernel, target.span, "a kernel's grid has one to three axes");
    }
    for (const ast::ExprId size : decl.grid) {
        const TypeId type = check_expr(size);
        if (types_.kind(type) != TypeKind::CompileInt && !types_.is_error(type)) {
            error(codes::not_compile_time,
                  ast().expr(size).span,
                  "a grid size is a compile-time integer, found `" + str(type) + "`");
        }
    }
    // Launch hints: integer literals, warps a power of two.
    for (const auto& [hint, value] :
         {std::pair{"warps", decl.warps}, std::pair{"stages", decl.stages}}) {
        if (value == ast::no_id) {
            continue;
        }
        const TypeData& data = types_.get(check_expr(value));
        const std::int64_t count =
            data.kind == TypeKind::CompileInt ? data.value.constant().value_or(0) : 0;
        const bool is_warps = std::string_view(hint) == "warps";
        if (count <= 0 || (is_warps && (count & (count - 1)) != 0)) {
            error(codes::invalid_kernel,
                  ast().expr(value).span,
                  is_warps ? "`warps` is a power of two written as a literal"
                           : "`stages` is a positive integer written as a literal");
        }
    }
    kernel_writes_.clear();
}

TypeId Checker::check_kernel_call(const ast::Expr& node,
                                  const ast::CallExpr& call,
                                  std::string_view name) {
    const auto positional = [&](std::size_t low, std::size_t high, const char* usage) {
        const bool is_valid = call.args.size() >= low && call.args.size() <= high &&
                              std::ranges::all_of(call.args, [](const ast::Argument& arg) {
                                  return arg.keyword.text.empty();
                              });
        if (!is_valid) {
            error(codes::bad_arguments, node.span, std::string("expected ") + usage);
            for (const ast::Argument& arg : call.args) {
                check_expr(arg.value);
            }
        }
        return is_valid;
    };
    const TypeId program = types_.scalar(ScalarKind::I32);
    if (name == "program_id") {
        if (!positional(1, 1, "`program_id(axis)`")) {
            return program;
        }
        const TypeData& axis = types_.get(check_expr(call.args.front().value));
        const auto value = axis.kind == TypeKind::CompileInt ? axis.value.constant() : std::nullopt;
        if (!value || *value < 0 || *value > 2) {
            error(codes::invalid_kernel,
                  ast().expr(call.args.front().value).span,
                  "`program_id` takes a grid axis: 0, 1, or 2");
        }
        return program;
    }
    if (!positional(1, 3, "`load(x[...])`, `load(x[...], mask)`, or `load(x[...], mask, other)`")) {
        return types_.error();
    }
    const auto access = check_memory_access(call.args.front().value, false);
    if (!access) {
        for (std::size_t i = 1; i < call.args.size(); ++i) {
            check_expr(call.args[i].value);
        }
        return types_.error();
    }
    const TypeId element = types_.scalar(access->dtype);
    const TypeId result =
        access->shape.empty() ? element : types_.tensor(access->shape, access->dtype);
    facts(call.args.front().value).type = result;
    if (call.args.size() >= 2) {
        check_mask(call.args[1].value, access->shape);
    }
    if (call.args.size() == 3) {
        const ast::ExprId other = call.args[2].value;
        coerce(check_expr(other, element),
               element,
               ast().expr(other).span,
               "the value of masked-off elements");
    }
    return result;
}

void Checker::check_store(const ast::StoreStmt& store, SourceSpan span) {
    const std::string name(ast::store_spelling(store.kind));
    if (!env_->in_kernel) {
        error(codes::invalid_kernel,
              span,
              "`" + name + "` writes a kernel's result; it stands only in a kernel");
        return;
    }
    const auto access = check_memory_access(store.target, true);
    if (!access) {
        check_expr(store.value);
        check_expr(store.mask);
        return;
    }
    if (store.kind != ast::StoreKind::Store) {
        // What Triton and Pallas both write atomically.
        const ScalarKind dtype = access->dtype.is_var ? ScalarKind::F32 : access->dtype.scalar;
        const bool is_supported =
            !access->dtype.is_var &&
            (dtype == ScalarKind::F32 || dtype == ScalarKind::I32 || dtype == ScalarKind::U32 ||
             dtype == ScalarKind::I64 || dtype == ScalarKind::U64 ||
             (dtype == ScalarKind::F16 && store.kind == ast::StoreKind::AtomicAdd));
        if (!is_supported) {
            error(codes::invalid_kernel,
                  span,
                  "`" + name +
                      "` writes `f32`, `i32`, `u32`, `i64` or `u64` (and `f16` adds), not `" +
                      types_.to_string(access->dtype) + "`");
        }
    }
    // One kind of write a result: an atomic's result starts from the
    // operation's identity, which a plain store would not keep.
    const auto [written, is_first] = kernel_writes_.try_emplace(access->memory, store.kind, span);
    if (!is_first && written->second.first != store.kind) {
        error(codes::invalid_kernel,
              span,
              "`" + std::string(entities_[access->memory].name) + "` is written with both `" +
                  std::string(ast::store_spelling(written->second.first)) + "` and `" + name + "`")
            .label(written->second.second, "first written here")
            .note("a result takes one kind of write");
    }
    const TypeId element = types_.scalar(access->dtype);
    const TypeId tile =
        access->shape.empty() ? element : types_.tensor(access->shape, access->dtype);
    facts(store.target).type = tile;
    // A scalar spreads over the tile.
    const TypeId value = check_expr(store.value, tile);
    const TypeKind kind = types_.kind(value);
    const bool is_scalar =
        kind == TypeKind::Scalar || kind == TypeKind::FloatLiteral || kind == TypeKind::CompileInt;
    coerce(value, is_scalar ? element : tile, ast().expr(store.value).span, "stored value");
    if (store.mask != ast::no_id) {
        check_mask(store.mask, access->shape);
    }
}

std::optional<Checker::MemoryAccess> Checker::check_memory_access(ast::ExprId target,
                                                                  bool is_store) {
    const ast::Expr& node = ast().expr(target);
    const auto* index = std::get_if<ast::IndexExpr>(&node.data);
    const auto* base =
        index != nullptr ? std::get_if<ast::NameExpr>(&ast().expr(index->base).data) : nullptr;
    const char* expected = is_store ? "`store` writes one of the kernel's results at indices: "
                                      "`out[i, j]`"
                                    : "`load` reads one of the kernel's tensor parameters at "
                                      "indices: `x[i, j]`";
    if (base == nullptr) {
        error(codes::invalid_kernel, node.span, expected);
        return std::nullopt;
    }
    memory_access_ = true;
    const TypeId base_type = check_expr(index->base);
    memory_access_ = false;
    const EntityId entity = facts(index->base).entity;
    if (types_.is_error(base_type)) {
        return std::nullopt;
    }
    if (entity == no_entity || !entities_[entity].is_memory ||
        entities_[entity].is_parameter == is_store) {
        error(codes::invalid_kernel, ast().expr(index->base).span, expected);
        return std::nullopt;
    }
    const TypeData& data = types_.get(base_type);
    if (index->components.size() != data.shape.size()) {
        error(codes::rank_mismatch,
              node.span,
              "`" + std::string(base->name.text) + "` has " + std::to_string(data.shape.size()) +
                  " axes, indexed with " + std::to_string(index->components.size()));
        return std::nullopt;
    }
    // Each index tile adds its axes, in order: `x[rows, cols]` with tiles
    // `[BM]` and `[BN]` is a `[BM, BN]` tile.
    MemoryAccess access{data.dtype, {}, entity};
    bool is_valid = true;
    for (const ast::IndexComponent& component : index->components) {
        if (component.kind != ast::IndexKind::Expr) {
            error(codes::invalid_kernel,
                  component.span,
                  "a kernel indexes memory with integers and integer tiles, not slices");
            is_valid = false;
            continue;
        }
        const TypeId type = check_expr(component.value);
        const TypeData& position = types_.get(type);
        const bool is_integer =
            position.kind == TypeKind::CompileInt ||
            ((position.kind == TypeKind::Scalar || position.kind == TypeKind::Tensor) &&
             types_.class_of(position.dtype) == DTypeClass::Integer);
        if (!is_integer) {
            if (position.kind != TypeKind::Error) {
                error(codes::type_mismatch,
                      ast().expr(component.value).span,
                      "an index is an integer or an integer tile, found `" + str(type) + "`");
            }
            is_valid = false;
            continue;
        }
        if (position.kind == TypeKind::Tensor) {
            access.shape.insert(access.shape.end(), position.shape.begin(), position.shape.end());
        }
    }
    if (!is_valid) {
        return std::nullopt;
    }
    return access;
}

void Checker::check_mask(ast::ExprId mask, const Shape& shape) {
    const TypeId type = check_expr(mask);
    const TypeData& data = types_.get(type);
    if (data.kind == TypeKind::Error) {
        return;
    }
    const bool is_bool = data.dtype == DType::of(ScalarKind::Bool);
    const bool fits =
        (data.kind == TypeKind::Scalar && is_bool) ||
        (data.kind == TypeKind::Tensor && is_bool && types_.equal(data.shape, shape, env_->solver));
    if (!fits) {
        error(codes::type_mismatch,
              ast().expr(mask).span,
              "a mask is a `bool` of the accessed shape `" +
                  (shape.empty() ? std::string("bool")
                                 : str(types_.tensor(shape, DType::of(ScalarKind::Bool)))) +
                  "`, found `" + str(type) + "`");
    }
}

// An op's `kernel name<args>`: the kernel takes the op's parameters, in
// order and of the same types, and writes what the op returns.
void Checker::check_kernel_binding(EntityId op) {
    const Entity& target = entities_[op];
    const auto& decl = std::get<ast::FunctionDecl>(modules_[target.module]->item(target.item).data);
    if (!decl.kernel) {
        return;
    }
    const ast::KernelBinding& binding = *decl.kernel;
    DeclInfo& info = decls_[op];
    const EntityId kernel = lookup(binding.name);
    if (kernel == no_entity) {
        error(codes::unknown_symbol,
              binding.name.span,
              "cannot find kernel `" + std::string(binding.name.text) + "` in this scope");
        return;
    }
    record_ref(binding.name.span, kernel);
    const Entity& found = entities_[kernel];
    if (found.kind != EntityKind::Function ||
        std::get<ast::FunctionDecl>(modules_[found.module]->item(found.item).data).kind !=
            ast::FunctionKind::Kernel) {
        error(codes::kernel_binding,
              binding.name.span,
              "`" + std::string(binding.name.text) + "` is not a kernel");
        return;
    }
    resolve(kernel);
    const DeclInfo& callee = decls_[kernel];
    const std::string name(found.name);
    if (binding.generic_args.size() != callee.generics.size()) {
        error(codes::bad_generic_arguments,
              binding.span,
              "`" + name + "` takes " + std::to_string(callee.generics.size()) +
                  " generic arguments, found " + std::to_string(binding.generic_args.size()));
        return;
    }
    Substitution substitution;
    std::vector<GenericValue> arguments;
    for (std::size_t i = 0; i < callee.generics.size(); ++i) {
        const auto value =
            eval_generic_arg(binding.generic_args[i], callee.generics[i], binding.span);
        if (!value) {
            return;
        }
        bind(substitution, callee.generics[i], *value);
        arguments.push_back(*value);
    }
    if (callee.params.size() != info.params.size()) {
        error(codes::kernel_binding,
              binding.span,
              "`" + name + "` takes " + std::to_string(callee.params.size()) +
                  " parameters, the op " + std::to_string(info.params.size()));
        return;
    }
    bool is_valid = true;
    for (std::size_t i = 0; i < info.params.size(); ++i) {
        const TypeId expected = types_.substitute(callee.params[i].type, substitution);
        if (!types_.equal(expected, info.params[i].type, env_->solver)) {
            error(codes::kernel_binding,
                  binding.span,
                  "parameter `" + std::string(callee.params[i].name) + "` of `" + name + "` is `" +
                      str(expected) + "` here, the op's `" + std::string(info.params[i].name) +
                      "` is `" + str(info.params[i].type) + "`");
            is_valid = false;
        }
    }
    std::vector<TypeId> results;
    results.reserve(callee.kernel_results.size());
    for (const EntityId result : callee.kernel_results) {
        results.push_back(types_.substitute(entities_[result].type, substitution));
    }
    const TypeId written = results.size() == 1 ? results.front() : types_.tuple(results);
    if (!types_.equal(written, info.result, env_->solver)) {
        error(codes::kernel_binding,
              binding.span,
              "`" + name + "` writes `" + str(written) + "` here, the op returns `" +
                  str(info.result) + "`");
        is_valid = false;
    }
    if (is_valid) {
        info.kernel = kernel;
        info.kernel_args = std::move(arguments);
    }
}

} // namespace linnet::sema
