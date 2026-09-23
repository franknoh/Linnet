#include "linnet/diagnostic/codes.hpp"

#include "checker.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace linnet::sema {

namespace {

bool is_literal(TypeKind kind) {
    return kind == TypeKind::CompileInt || kind == TypeKind::FloatLiteral;
}

constexpr auto float_functions =
    std::to_array<std::string_view>({"exp", "log", "sqrt", "rsqrt", "sin", "cos", "tanh"});

} // namespace

// ------------------------------------------------------------------- inference

bool Checker::mentions_unbound(const Inference& inference, const shape::Poly& poly) const {
    for (const GenericInfo& generic : inference.callee->generics) {
        const bool is_unbound = generic.kind == GenericKind::Dim
                                    ? !inference.bound.dims.contains(generic.symbol)
                                    : generic.kind == GenericKind::Pack &&
                                          !inference.bound.packs.contains(generic.symbol);
        if (is_unbound && dims_.mentions(poly, generic.symbol)) {
            return true;
        }
    }
    return false;
}

bool Checker::unify_dim(Inference& inference, const shape::Poly& param, const shape::Poly& arg) {
    const shape::Poly expected = types_.substitute(param, inference.bound);
    if (const auto atom = expected.single_atom()) {
        const shape::Atom& symbol = dims_.atom(*atom);
        if (symbol.kind == shape::AtomKind::Symbol && mentions_unbound(inference, expected)) {
            inference.bound.dims[symbol.symbol] = arg;
            return true;
        }
    }
    if (mentions_unbound(inference, expected)) {
        inference.deferred_dims.emplace_back(param, arg);
        return true;
    }
    if (env_->solver.prove_equal(expected, arg)) {
        return true;
    }
    inference.failure =
        "dimension `" + str(arg) + "` does not match the expected `" + str(expected) + "`";
    return false;
}

bool Checker::unify_dtype(Inference& inference, DType param, DType arg) {
    const DType expected = types_.substitute(param, inference.bound);
    if (expected.is_var) {
        const bool is_bindable =
            std::any_of(inference.callee->generics.begin(),
                        inference.callee->generics.end(),
                        [&](const GenericInfo& generic) {
                            return generic.kind == GenericKind::DType &&
                                   generic.dtype_var == expected.var &&
                                   !inference.bound.dtypes.contains(expected.var);
                        });
        if (is_bindable) {
            inference.bound.dtypes[expected.var] = arg;
            return true;
        }
    }
    if (expected == arg) {
        return true;
    }
    inference.failure = "dtype `" + types_.to_string(arg) + "` does not match the expected `" +
                        types_.to_string(expected) + "`";
    return false;
}

bool Checker::unify_shape(Inference& inference, const Shape& param, const Shape& arg) {
    const Shape expected = types_.substitute(param, inference.bound);
    std::optional<std::size_t> open_pack;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const bool is_open =
            expected[i].is_pack && std::any_of(inference.callee->generics.begin(),
                                               inference.callee->generics.end(),
                                               [&](const GenericInfo& generic) {
                                                   return generic.kind == GenericKind::Pack &&
                                                          generic.symbol == expected[i].pack;
                                               });
        if (!is_open) {
            continue;
        }
        if (open_pack) {
            inference.failure = "two shape packs in one tensor type cannot be inferred";
            return false;
        }
        open_pack = i;
    }

    const auto mismatch = [&] {
        inference.failure = "shape [" + types_.to_string(arg) + "] does not match the expected [" +
                            types_.to_string(expected) + "]";
        return false;
    };
    const auto unify_unit = [&](const ShapeElem& want, const ShapeElem& have) {
        if (want.is_pack || have.is_pack) {
            return want.is_pack && have.is_pack && want.pack == have.pack ? true : mismatch();
        }
        return unify_dim(inference, want.dim, have.dim);
    };

    if (!open_pack) {
        if (expected.size() != arg.size()) {
            return mismatch();
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (!unify_unit(expected[i], arg[i])) {
                return false;
            }
        }
        return true;
    }
    const std::size_t before = *open_pack;
    const std::size_t after = expected.size() - before - 1;
    if (arg.size() < before + after) {
        return mismatch();
    }
    for (std::size_t i = 0; i < before; ++i) {
        if (!unify_unit(expected[i], arg[i])) {
            return false;
        }
    }
    for (std::size_t i = 0; i < after; ++i) {
        if (!unify_unit(expected[expected.size() - 1 - i], arg[arg.size() - 1 - i])) {
            return false;
        }
    }
    inference.bound.packs[expected[before].pack] =
        Shape(arg.begin() + static_cast<std::ptrdiff_t>(before),
              arg.end() - static_cast<std::ptrdiff_t>(after));
    return true;
}

bool Checker::unify_value(Inference& inference,
                          const GenericValue& param,
                          const GenericValue& arg) {
    if (param.kind != arg.kind) {
        return false;
    }
    switch (param.kind) {
    case GenericValue::Kind::Dim:
        return unify_dim(inference, param.dim, arg.dim);
    case GenericValue::Kind::Pack:
        return unify_shape(inference, param.shape, arg.shape);
    case GenericValue::Kind::DType:
        return unify_dtype(inference, param.dtype, arg.dtype);
    }
    return false;
}

bool Checker::unify(Inference& inference, TypeId param, TypeId arg) {
    const TypeData& want = types_.get(param);
    const TypeData& have = types_.get(arg);
    if (want.kind == TypeKind::Error || have.kind == TypeKind::Error) {
        return true;
    }
    const auto mismatch = [&] {
        if (inference.failure.empty()) {
            inference.failure = "found `" + str(arg) + "`, expected `" +
                                str(types_.substitute(param, inference.bound)) + "`";
        }
        return false;
    };
    switch (want.kind) {
    case TypeKind::Scalar:
        if (is_literal(have.kind)) {
            inference.deferred_literals.push_back({arg, want.dtype, inference.span});
            return true;
        }
        return have.kind == TypeKind::Scalar && unify_dtype(inference, want.dtype, have.dtype)
                   ? true
                   : mismatch();
    case TypeKind::Tensor:
        return have.kind == TypeKind::Tensor && unify_dtype(inference, want.dtype, have.dtype) &&
                       unify_shape(inference, want.shape, have.shape)
                   ? true
                   : mismatch();
    case TypeKind::Optional:
        if (have.kind == TypeKind::NoneLiteral) {
            return true;
        }
        if (have.kind != TypeKind::Optional) {
            inference.failure = "found `" + str(arg) + "` where an optional is expected; wrap " +
                                "the value in `some(...)`";
            return false;
        }
        return unify(inference, want.elements.front(), have.elements.front()) ? true : mismatch();
    case TypeKind::Tuple:
        if (have.kind != TypeKind::Tuple || have.elements.size() != want.elements.size()) {
            return mismatch();
        }
        for (std::size_t i = 0; i < want.elements.size(); ++i) {
            if (!unify(inference, want.elements[i], have.elements[i])) {
                return mismatch();
            }
        }
        return true;
    case TypeKind::Array:
        return have.kind == TypeKind::Array &&
                       unify(inference, want.elements.front(), have.elements.front()) &&
                       unify_dim(inference, want.value, have.value)
                   ? true
                   : mismatch();
    case TypeKind::Struct:
    case TypeKind::Enum:
    case TypeKind::Block:
        if (have.kind != want.kind || have.decl != want.decl ||
            have.args.size() != want.args.size()) {
            return mismatch();
        }
        for (std::size_t i = 0; i < want.args.size(); ++i) {
            if (!unify_value(inference, want.args[i], have.args[i])) {
                return mismatch();
            }
        }
        return true;
    default:
        return types_.equal(types_.substitute(param, inference.bound), arg, env_->solver)
                   ? true
                   : mismatch();
    }
}

// ----------------------------------------------------------------------- calls

TypeId Checker::check_call(const ast::Expr& node, const ast::CallExpr& call) {
    const ast::ExprId self = current_expr_;
    const ast::Expr& callee = ast().expr(call.callee);
    const auto check_arguments_only = [&] {
        for (const ast::Argument& arg : call.args) {
            check_expr(arg.value);
        }
        return types_.error();
    };
    const auto call_entity = [&](EntityId entity, SourceSpan span, const Substitution& receiver) {
        current_expr_ = self; // checking the receiver moved it
        if (entities_[entity].kind != EntityKind::Function) {
            error(codes::not_callable,
                  span,
                  "`" + std::string(entities_[entity].name) + "` is not a function");
            return check_arguments_only();
        }
        return check_user_call(node, call, entity, receiver);
    };

    if (const auto* name = std::get_if<ast::NameExpr>(&callee.data)) {
        const EntityId entity = lookup(name->name);
        if (entity != no_entity) {
            return call_entity(entity, callee.span, {});
        }
        if (is_prelude_name(name->name.text) && !scalar_from_name(name->name.text)) {
            facts(self).builtin = std::string(name->name.text);
            return check_builtin_call(node, call, name->name.text);
        }
        error(codes::unknown_symbol,
              callee.span,
              "cannot find function `" + std::string(name->name.text) + "` in this scope");
        return check_arguments_only();
    }

    if (const auto* member = std::get_if<ast::MemberExpr>(&callee.data)) {
        if (const auto* base = std::get_if<ast::NameExpr>(&ast().expr(member->base).data)) {
            const EntityId module = lookup(base->name);
            if (module != no_entity && entities_[module].kind == EntityKind::Module) {
                const EntityId item =
                    lookup_in_module(entities_[module].module_ref, member->member);
                return item == no_entity ? check_arguments_only()
                                         : call_entity(item, member->member.span, {});
            }
        }
        const TypeId receiver = check_expr(member->base);
        const TypeData& data = types_.get(receiver);
        if (data.kind == TypeKind::Error) {
            return check_arguments_only();
        }
        if (data.kind == TypeKind::Block) {
            const Scope& scope = decls_[data.decl].scope;
            const auto found = scope.find(member->member.text);
            if (found != scope.end() && entities_[found->second].kind == EntityKind::Function) {
                const Entity& method = entities_[found->second];
                if (!method.is_pub && method.module != env_->module) {
                    error(codes::private_item,
                          member->member.span,
                          "method `" + std::string(method.name) + "` is private");
                    return check_arguments_only();
                }
                record_ref(member->member.span, found->second);
                current_expr_ = self;
                return check_user_call(node, call, found->second, substitution_of(data));
            }
        }
        error(codes::unknown_member,
              member->member.span,
              "`" + str(receiver) + "` has no method `" + std::string(member->member.text) + "`");
        return check_arguments_only();
    }

    error(codes::not_callable, callee.span, "this expression cannot be called");
    return check_arguments_only();
}

TypeId Checker::check_user_call(const ast::Expr& node,
                                const ast::CallExpr& call,
                                EntityId callee,
                                const Substitution& receiver) {
    const ast::ExprId self = current_expr_;
    facts(self).entity = callee;
    resolve(callee);
    const DeclInfo& info = decls_[callee];
    const std::string name(entities_[callee].name);
    if (env_->function != no_entity) {
        call_edges_.push_back({env_->function, callee, ast().expr(call.callee).span});
    }

    Inference inference;
    inference.callee = &info;
    inference.bound = receiver;
    bool is_valid = true;

    if (call.generic_args.size() > info.generics.size()) {
        error(codes::bad_generic_arguments,
              node.span,
              "`" + name + "` takes " + std::to_string(info.generics.size()) +
                  " generic arguments, found " + std::to_string(call.generic_args.size()));
        for (const ast::Argument& arg : call.args) {
            check_expr(arg.value);
        }
        return types_.error();
    } else {
        for (std::size_t i = 0; i < call.generic_args.size(); ++i) {
            const GenericInfo& param = info.generics[i];
            const auto value = eval_generic_arg(call.generic_args[i], param, node.span);
            if (!value) {
                is_valid = false;
                continue;
            }
            switch (param.kind) {
            case GenericKind::Dim:
                inference.bound.dims[param.symbol] = value->dim;
                break;
            case GenericKind::Pack:
                inference.bound.packs[param.symbol] = value->shape;
                break;
            case GenericKind::DType:
                inference.bound.dtypes[param.dtype_var] = value->dtype;
                break;
            }
        }
    }

    // Match arguments to parameters: positional first, then by keyword.
    std::vector<const ast::Argument*> assigned(info.params.size(), nullptr);
    std::size_t next_positional = 0;
    for (const ast::Argument& arg : call.args) {
        std::size_t slot = info.params.size();
        if (arg.keyword.text.empty()) {
            slot = next_positional++;
        } else {
            for (std::size_t i = 0; i < info.params.size(); ++i) {
                if (info.params[i].name == arg.keyword.text) {
                    slot = i;
                }
            }
        }
        if (slot >= info.params.size() || assigned[slot] != nullptr) {
            const SourceSpan span =
                arg.keyword.text.empty() ? ast().expr(arg.value).span : arg.keyword.span;
            error(codes::bad_arguments,
                  span,
                  arg.keyword.text.empty()
                      ? "`" + name + "` takes " + std::to_string(info.params.size()) +
                            " arguments, found " + std::to_string(call.args.size())
                  : slot >= info.params.size()
                      ? "`" + name + "` has no parameter `" + std::string(arg.keyword.text) + "`"
                      : "parameter `" + std::string(arg.keyword.text) + "` is given twice");
            check_expr(arg.value);
            is_valid = false;
            continue;
        }
        assigned[slot] = &arg;
        facts(self).argument_slots.push_back(static_cast<std::uint32_t>(slot));
    }

    for (std::size_t i = 0; i < info.params.size(); ++i) {
        const ParamInfo& param = info.params[i];
        if (assigned[i] == nullptr) {
            if (!param.has_default) {
                error(codes::bad_arguments,
                      node.span,
                      "missing argument `" + std::string(param.name) + "` in call to `" + name +
                          "`");
                is_valid = false;
            }
            continue;
        }
        const ast::ExprId value = assigned[i]->value;
        const TypeId arg_type = check_expr(value);
        if (types_.is_error(arg_type)) {
            is_valid = false;
            continue;
        }
        inference.span = ast().expr(value).span;
        inference.failure.clear();
        if (!unify(inference, param.type, arg_type)) {
            const TypeId expected = types_.substitute(param.type, inference.bound);
            auto report =
                error(types_.kind(arg_type) == TypeKind::Tensor &&
                              types_.kind(expected) == TypeKind::Tensor
                          ? (types_.get(arg_type).dtype != types_.get(expected).dtype
                                 ? codes::dtype_mismatch
                                 : codes::shape_mismatch)
                          : codes::type_mismatch,
                      inference.span,
                      "argument `" + std::string(param.name) + "` of `" + name + "` does not fit");
            report.note("parameter type is `" + str(param.type) + "`")
                .note("argument type is `" + str(arg_type) + "`");
            if (!inference.failure.empty()) {
                report.note(inference.failure);
            }
            is_valid = false;
        }
    }
    if (!is_valid) {
        return types_.error();
    }

    // Generics that no argument determined fall back to their defaults.
    for (const GenericInfo& generic : info.generics) {
        const bool is_bound = generic.kind == GenericKind::Dim
                                  ? inference.bound.dims.contains(generic.symbol)
                              : generic.kind == GenericKind::Pack
                                  ? inference.bound.packs.contains(generic.symbol)
                                  : inference.bound.dtypes.contains(generic.dtype_var);
        if (is_bound) {
            continue;
        }
        if (!generic.default_value) {
            error(codes::cannot_infer,
                  node.span,
                  "cannot infer `" + std::string(generic.name) + "` in call to `" + name + "`")
                .help("pass it explicitly: `" + name + "<...>(...)`");
            return types_.error();
        }
        const GenericValue value = types_.substitute(*generic.default_value, inference.bound);
        switch (generic.kind) {
        case GenericKind::Dim:
            inference.bound.dims[generic.symbol] = value.dim;
            break;
        case GenericKind::Pack:
            inference.bound.packs[generic.symbol] = value.shape;
            break;
        case GenericKind::DType:
            inference.bound.dtypes[generic.dtype_var] = value.dtype;
            break;
        }
    }

    for (const auto& [param, arg] : inference.deferred_dims) {
        const shape::Poly expected = types_.substitute(param, inference.bound);
        if (!env_->solver.prove_equal(expected, arg)) {
            error(codes::shape_mismatch,
                  node.span,
                  "cannot prove the dimensions required by `" + name + "`")
                .note("expected `" + str(expected) + "`")
                .note("found `" + str(arg) + "`")
                .help("add a `where` constraint that proves they are equal");
            return types_.error();
        }
    }
    for (const GenericInfo& generic : info.generics) {
        if (generic.kind != GenericKind::DType) {
            continue;
        }
        const DType bound = inference.bound.dtypes[generic.dtype_var];
        const bool is_satisfied = bound.is_var
                                      ? class_contains(generic.constraint, types_.class_of(bound))
                                      : class_contains(generic.constraint, bound.scalar);
        if (!is_satisfied) {
            error(codes::dtype_constraint,
                  node.span,
                  "`" + types_.to_string(bound) + "` does not satisfy the constraint on `" +
                      std::string(generic.name) + "` of `" + name + "`");
            return types_.error();
        }
    }
    for (const DeferredLiteral& literal : inference.deferred_literals) {
        if (!literal_fits(types_.get(literal.literal),
                          types_.substitute(literal.dtype, inference.bound),
                          literal.span)) {
            return types_.error();
        }
    }
    for (const ConstraintInfo& constraint : info.constraints) {
        const shape::Poly lhs = types_.substitute(constraint.lhs, inference.bound);
        const shape::Poly rhs = types_.substitute(constraint.rhs, inference.bound);
        if (!env_->solver.prove(constraint.relation, lhs, rhs)) {
            error(codes::constraint_unsatisfied,
                  node.span,
                  "cannot prove a constraint of `" + name + "`")
                .label(constraint.span, "required here")
                .note("required: `" + std::string(text(constraint.span)) + "`")
                .note("with `" + str(lhs) + "` on the left and `" + str(rhs) + "` on the right")
                .help("state the same constraint in the calling function's `where` clause");
            return types_.error();
        }
    }
    facts(self).substitution = inference.bound;
    return types_.substitute(info.result, inference.bound);
}

// -------------------------------------------------------------------- builtins

TypeId Checker::check_builtin_call(const ast::Expr& node,
                                   const ast::CallExpr& call,
                                   std::string_view name) {
    const std::string callee(name);
    std::vector<TypeId> args;
    const ast::Argument* axis_argument = nullptr;
    bool is_valid = true;
    for (const ast::Argument& arg : call.args) {
        if (!arg.keyword.text.empty()) {
            if (name == "concat" && arg.keyword.text == "axis" && axis_argument == nullptr) {
                axis_argument = &arg;
                continue;
            }
            error(codes::bad_arguments,
                  arg.keyword.span,
                  "`" + callee + "` has no parameter `" + std::string(arg.keyword.text) + "`");
            is_valid = false;
        }
        args.push_back(check_expr(arg.value));
        is_valid = is_valid && !types_.is_error(args.back());
    }
    const auto arg_span = [&](std::size_t i) { return ast().expr(call.args[i].value).span; };
    const auto expect_count = [&](std::size_t count, std::string_view signature) {
        if (args.size() == count) {
            return true;
        }
        error(codes::bad_arguments,
              node.span,
              "`" + callee + "` takes " + std::to_string(count) + " arguments, found " +
                  std::to_string(args.size()))
            .note("signature: `" + std::string(signature) + "`");
        return false;
    };
    const auto explicit_dtype = [&](std::size_t max_generics) -> std::optional<DType> {
        if (call.generic_args.size() > max_generics) {
            error(codes::bad_generic_arguments,
                  node.span,
                  "`" + callee + "` takes at most " + std::to_string(max_generics) +
                      " generic arguments");
            is_valid = false;
            return std::nullopt;
        }
        if (call.generic_args.empty()) {
            return std::nullopt;
        }
        GenericInfo param;
        param.name = "T";
        param.kind = GenericKind::DType;
        const auto value = eval_generic_arg(call.generic_args.front(), param, node.span);
        is_valid = is_valid && value.has_value();
        return value ? std::optional<DType>(value->dtype) : std::nullopt;
    };
    const auto shape_argument = [&](std::size_t i) -> const Shape* {
        const TypeData& data = types_.get(args[i]);
        if (data.kind == TypeKind::ShapeValue) {
            return &data.shape;
        }
        error(codes::type_mismatch,
              arg_span(i),
              "`" + callee + "` expects a shape such as `[B, S, H]` here, found `" + str(args[i]) +
                  "`");
        return nullptr;
    };
    const auto tensor_argument = [&](std::size_t i) -> const TypeData* {
        const TypeData& data = types_.get(args[i]);
        if (data.kind == TypeKind::Tensor) {
            return &data;
        }
        error(codes::type_mismatch,
              arg_span(i),
              "`" + callee + "` expects a tensor here, found `" + str(args[i]) + "`");
        return nullptr;
    };

    if (name == "pad" || name == "gather" || name == "scatter") {
        error(codes::not_implemented, node.span, "`" + callee + "` is not supported yet");
        return types_.error();
    }

    if (name == "cast") {
        const auto dtype = explicit_dtype(1);
        if (!is_valid || !expect_count(1, "cast<T>(x)")) {
            return types_.error();
        }
        if (!dtype) {
            error(codes::cannot_infer, node.span, "`cast` needs a target dtype")
                .help("write `cast<f32>(x)`");
            return types_.error();
        }
        const TypeData& data = types_.get(args[0]);
        if (data.kind == TypeKind::Tensor) {
            return types_.tensor(data.shape, *dtype);
        }
        if (data.kind == TypeKind::Scalar || is_literal(data.kind)) {
            return types_.scalar(*dtype);
        }
        error(codes::type_mismatch, arg_span(0), "`" + str(args[0]) + "` cannot be cast");
        return types_.error();
    }

    const bool is_float_function =
        std::find(float_functions.begin(), float_functions.end(), name) != float_functions.end();
    if (is_float_function || name == "abs") {
        if (!explicit_dtype(0).has_value() && !is_valid) {
            return types_.error();
        }
        if (!is_valid || !expect_count(1, callee + "(x)")) {
            return types_.error();
        }
        const TypeData& data = types_.get(args[0]);
        if (data.kind == TypeKind::FloatLiteral ||
            (data.kind == TypeKind::CompileInt && !is_float_function)) {
            return data.kind == TypeKind::FloatLiteral ? types_.float_literal(std::nullopt)
                                                       : args[0];
        }
        const auto dtype = numeric_dtype(args[0]);
        const DTypeClass needed = is_float_function ? DTypeClass::Float : DTypeClass::Numeric;
        const bool is_accepted =
            dtype && (dtype->is_var ? class_contains(needed, types_.class_of(*dtype))
                                    : class_contains(needed, dtype->scalar));
        if (!is_accepted) {
            error(codes::invalid_operand,
                  arg_span(0),
                  "`" + callee + "` needs a " + (is_float_function ? "floating-point" : "numeric") +
                      " scalar or tensor, found `" + str(args[0]) + "`");
            return types_.error();
        }
        return args[0];
    }

    if (name == "min" || name == "max") {
        if (!is_valid || !expect_count(2, callee + "(a, b)")) {
            return types_.error();
        }
        const TypeData& a = types_.get(args[0]);
        const TypeData& b = types_.get(args[1]);
        if (a.kind == TypeKind::CompileInt && b.kind == TypeKind::CompileInt) {
            return types_.compile_int(name == "min" ? dims_.min(a.value, b.value)
                                                    : dims_.max(a.value, b.value));
        }
        return elementwise(args[0], args[1], node.span, true, false);
    }

    if (name == "shl" || name == "shr") {
        if (!is_valid || !expect_count(2, callee + "(x, bits)")) {
            return types_.error();
        }
        return check_bitwise(
            node.span, name, args[0], args[1], [&](std::int64_t a, std::int64_t b) {
                if (b < 0 || b >= 64) {
                    return std::int64_t{0};
                }
                return name == "shl" ? static_cast<std::int64_t>(static_cast<std::uint64_t>(a) << b)
                                     : (a >> b);
            });
    }

    if (name == "select") {
        if (!is_valid || !expect_count(3, "select(condition, a, b)")) {
            return types_.error();
        }
        const auto condition = numeric_dtype(args[0]);
        if (!condition || *condition != DType::of(ScalarKind::Bool)) {
            error(codes::condition_not_bool,
                  arg_span(0),
                  "the condition of `select` must be `bool`, found `" + str(args[0]) + "`");
            return types_.error();
        }
        const TypeId values = elementwise(args[1], args[2], node.span, false, false);
        if (types_.is_error(values)) {
            return values;
        }
        const TypeData& picked = types_.get(values);
        const TypeData& mask = types_.get(args[0]);
        if (mask.kind != TypeKind::Tensor) {
            return values;
        }
        auto shape = broadcast(
            mask.shape, picked.kind == TypeKind::Tensor ? picked.shape : Shape{}, node.span);
        return shape ? types_.tensor(std::move(*shape), picked.dtype) : types_.error();
    }

    if (name == "reshape" || name == "broadcast_to") {
        if (!is_valid || !expect_count(2, callee + "(x, shape)")) {
            return types_.error();
        }
        const TypeData* tensor = tensor_argument(0);
        const Shape* target = shape_argument(1);
        if (tensor == nullptr || target == nullptr) {
            return types_.error();
        }
        if (name == "reshape") {
            const shape::Poly from = types_.element_count(tensor->shape);
            const shape::Poly to = types_.element_count(*target);
            if (!env_->solver.prove_equal(from, to)) {
                error(codes::reshape_count,
                      node.span,
                      "cannot prove that `reshape` keeps the number of elements")
                    .note("the input has `" + str(env_->solver.simplify(from)) + "` elements")
                    .note("the result has `" + str(env_->solver.simplify(to)) + "` elements")
                    .help("state divisibility in a `where` clause, for example `H % N == 0`");
                return types_.error();
            }
            return types_.tensor(*target, tensor->dtype);
        }
        if (tensor->shape.size() > target->size()) {
            error(codes::unproven_broadcast,
                  node.span,
                  "`broadcast_to` cannot reduce the number of axes");
            return types_.error();
        }
        for (std::size_t i = 0; i < tensor->shape.size(); ++i) {
            const ShapeElem& from = tensor->shape[tensor->shape.size() - 1 - i];
            const ShapeElem& to = (*target)[target->size() - 1 - i];
            const bool is_compatible =
                !from.is_pack && (env_->solver.prove_equal(from.dim, to.dim) ||
                                  env_->solver.prove_equal(from.dim, shape::Poly(1)));
            if (!is_compatible) {
                error(codes::unproven_broadcast,
                      node.span,
                      "cannot prove that [" + types_.to_string(tensor->shape) +
                          "] broadcasts to [" + types_.to_string(*target) + "]");
                return types_.error();
            }
        }
        return types_.tensor(*target, tensor->dtype);
    }

    if (name == "permute") {
        if (!is_valid || !expect_count(2, "permute(x, axes)")) {
            return types_.error();
        }
        const TypeData* tensor = tensor_argument(0);
        const Shape* axes = shape_argument(1);
        if (tensor == nullptr || axes == nullptr) {
            return types_.error();
        }
        const bool has_pack = std::any_of(tensor->shape.begin(),
                                          tensor->shape.end(),
                                          [](const ShapeElem& unit) { return unit.is_pack; });
        if (has_pack || axes->size() != tensor->shape.size()) {
            error(codes::invalid_axis,
                  node.span,
                  has_pack ? "`permute` needs a tensor whose rank is known"
                           : "`permute` needs one axis per dimension");
            return types_.error();
        }
        Shape result;
        std::vector<bool> used(axes->size(), false);
        for (const ShapeElem& axis : *axes) {
            const auto position = axis.dim.constant();
            if (!position || *position < 0 ||
                *position >= static_cast<std::int64_t>(axes->size()) ||
                used[static_cast<std::size_t>(*position)]) {
                error(codes::invalid_axis,
                      arg_span(1),
                      "the axes of `permute` must be a permutation of 0.." +
                          std::to_string(axes->size() - 1));
                return types_.error();
            }
            used[static_cast<std::size_t>(*position)] = true;
            result.push_back(tensor->shape[static_cast<std::size_t>(*position)]);
        }
        return types_.tensor(std::move(result), tensor->dtype);
    }

    if (name == "concat") {
        if (!is_valid) {
            return types_.error();
        }
        if (args.size() < 2 || axis_argument == nullptr) {
            error(codes::bad_arguments,
                  node.span,
                  "`concat` takes at least two tensors and a keyword `axis`")
                .note("signature: `concat(a, b, ..., axis = 0)`");
            return types_.error();
        }
        const auto axis = constant_index(axis_argument->value, "`axis`");
        const TypeData* first = tensor_argument(0);
        if (!axis || first == nullptr) {
            return types_.error();
        }
        // Non-negative axes count from the front and negative ones from the
        // back; a shape pack on that side would make the position unknown.
        const auto rank = static_cast<std::int64_t>(first->shape.size());
        const std::int64_t position = *axis < 0 ? rank + *axis : *axis;
        bool is_addressable = position >= 0 && position < rank;
        for (std::int64_t i = 0; is_addressable && i < rank; ++i) {
            const bool is_on_counted_side = *axis < 0 ? i >= position : i <= position;
            is_addressable =
                !(is_on_counted_side && first->shape[static_cast<std::size_t>(i)].is_pack);
        }
        if (!is_addressable) {
            error(codes::invalid_axis,
                  ast().expr(axis_argument->value).span,
                  "axis " + std::to_string(*axis) + " does not name a dimension of `" +
                      str(args[0]) + "`");
            return types_.error();
        }
        Shape result = first->shape;
        const auto at = static_cast<std::size_t>(position);
        for (std::size_t i = 1; i < args.size(); ++i) {
            const TypeData* next = tensor_argument(i);
            if (next == nullptr) {
                return types_.error();
            }
            if (next->dtype != first->dtype) {
                error(codes::dtype_mismatch, arg_span(i), "tensor dtypes do not match")
                    .note("first tensor has dtype " + types_.to_string(first->dtype))
                    .note("this tensor has dtype " + types_.to_string(next->dtype));
                return types_.error();
            }
            Shape expected = next->shape;
            if (expected.size() == result.size() && !expected[at].is_pack) {
                expected[at] = result[at];
            }
            if (!types_.equal(expected, result, env_->solver)) {
                error(codes::shape_mismatch,
                      arg_span(i),
                      "`concat` needs equal shapes except along the axis")
                    .note("first tensor has shape [" + types_.to_string(first->shape) + "]")
                    .note("this tensor has shape [" + types_.to_string(next->shape) + "]");
                return types_.error();
            }
            result[at].dim = result[at].dim + next->shape[at].dim;
        }
        return types_.tensor(std::move(result), first->dtype);
    }

    if (name == "iota") {
        const auto dtype = explicit_dtype(1);
        if (!is_valid || !expect_count(1, "iota<T = i64>(n)")) {
            return types_.error();
        }
        const TypeData& count = types_.get(args[0]);
        if (count.kind != TypeKind::CompileInt) {
            error(codes::not_compile_time,
                  arg_span(0),
                  "`iota` needs a compile-time length, found `" + str(args[0]) + "`");
            return types_.error();
        }
        check_dimension(count.value, arg_span(0));
        const DType element = dtype.value_or(DType::of(ScalarKind::I64));
        if (dtype &&
            !(element.is_var ? class_contains(DTypeClass::Numeric, types_.class_of(element))
                             : class_contains(DTypeClass::Numeric, element.scalar))) {
            error(codes::dtype_constraint, node.span, "`iota` needs a numeric dtype");
            return types_.error();
        }
        return types_.tensor({ShapeElem::of(count.value)}, element);
    }

    if (name == "fill") {
        const auto dtype = explicit_dtype(1);
        if (!is_valid || !expect_count(2, "fill<T>(shape, value)")) {
            return types_.error();
        }
        const Shape* target = shape_argument(0);
        if (target == nullptr) {
            return types_.error();
        }
        const TypeData& value = types_.get(args[1]);
        DType element;
        if (dtype) {
            element = *dtype;
            if (types_.is_error(
                    coerce(args[1], types_.scalar(element), arg_span(1), "fill value"))) {
                return types_.error();
            }
        } else if (value.kind == TypeKind::Scalar) {
            element = value.dtype;
        } else {
            error(codes::cannot_infer, node.span, "cannot infer the dtype of `fill`")
                .help("write `fill<f32>(shape, value)`");
            return types_.error();
        }
        return types_.tensor(*target, element);
    }

    error(codes::not_callable, node.span, "`" + callee + "` is not a function");
    return types_.error();
}

} // namespace linnet::sema
