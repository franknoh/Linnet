#include "linnet/ir/lower.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <utility>

namespace linnet::ir {

namespace {

using namespace sema;

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
    std::int64_t value = 0;
    for (const char c : text) {
        if (c == '_') {
            continue;
        }
        const int digit = c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;
        value = value * base + digit;
    }
    return value;
}

class Lowerer {
public:
    Lowerer(const SourceManager& sources,
            std::span<const ast::Ast* const> modules,
            std::shared_ptr<Model> model)
        : sources_(sources), modules_(modules.begin(), modules.end()), module_(std::move(model)) {}

    Module run() {
        Model& model = module_.model();
        for (EntityId id = 0; id < model.entities.size(); ++id) {
            if (model.entities[id].kind == EntityKind::Function) {
                lower_function(id);
            }
        }
        return std::move(module_);
    }

private:
    // The function being lowered and the values its names denote.
    struct Frame {
        std::uint32_t module = 0;
        EntityId function = no_entity;
        EntityId block_entity = no_entity;
        ValueId self = no_id;
        std::map<EntityId, ValueId> locals;
        std::vector<std::map<std::string_view, ValueId>> index_scopes;
        Substitution substitution; // applied to types taken from the model
        std::vector<TypeId> results;
    };

    Model& model() { return module_.model(); }
    TypeStore& types() { return module_.types(); }
    const ast::Ast& ast() const { return *modules_[frame_->module]; }
    const ExprFacts& facts(ast::ExprId id) const { return model_facts(frame_->module, id); }
    const ExprFacts& model_facts(std::uint32_t module, ast::ExprId id) const {
        return module_.model().expr(module, id);
    }

    TypeId substituted(TypeId type) {
        return type == no_type ? types().error() : types().substitute(type, frame_->substitution);
    }

    // The type an expression's value has: its checked type, with literals
    // resolved by the context they were used in.
    TypeId value_type(ast::ExprId id, TypeId expected) {
        const TypeId checked = substituted(facts(id).type);
        const TypeKind kind = types().kind(checked);
        if (kind == TypeKind::CompileInt || kind == TypeKind::FloatLiteral ||
            kind == TypeKind::NoneLiteral) {
            if (expected != no_type && !types().is_error(expected)) {
                const TypeKind expected_kind = types().kind(expected);
                if (expected_kind == TypeKind::Scalar || expected_kind == TypeKind::Tensor ||
                    expected_kind == TypeKind::Optional) {
                    return expected_kind == TypeKind::Tensor
                               ? types().scalar(types().get(expected).dtype)
                               : expected;
                }
            }
            if (kind == TypeKind::CompileInt) {
                return types().scalar(ScalarKind::I64);
            }
            if (kind == TypeKind::FloatLiteral) {
                return types().scalar(ScalarKind::F64);
            }
        }
        return checked;
    }

    ValueId emit(OpKind kind,
                 std::vector<ValueId> operands,
                 TypeId result,
                 Attributes attributes = {},
                 SourceSpan span = {}) {
        std::vector<TypeId> results;
        if (result != no_type) {
            results.push_back(result);
        }
        const OpId op =
            module_.add_op(block_, kind, std::move(operands), results, std::move(attributes), span);
        return results.empty() ? no_id : module_.op(op).results.front();
    }

    ValueId const_int(std::int64_t value, TypeId type) {
        Attributes attributes;
        attributes.integer = value;
        return emit(OpKind::ConstInt, {}, type, attributes);
    }

    ValueId const_dim(const shape::Poly& value) {
        Attributes attributes;
        attributes.dim = types().substitute(value, frame_->substitution);
        return emit(OpKind::ConstDim, {}, types().scalar(ScalarKind::I64), attributes);
    }

    // Runs `body` inside a new single-block region attached to `parent`,
    // with block arguments of the given types. Returns the region.
    RegionId nested_region(OpId parent,
                           const std::vector<std::pair<TypeId, std::string>>& arguments,
                           const std::function<void(const std::vector<ValueId>&)>& body) {
        const RegionId region = module_.add_region(parent);
        const BlockId saved = block_;
        block_ = module_.add_block(region);
        std::vector<ValueId> values;
        values.reserve(arguments.size());
        for (const auto& [type, name] : arguments) {
            values.push_back(module_.add_argument(block_, type, name));
        }
        body(values);
        block_ = saved;
        return region;
    }

    // Creates an operation with regions filled in afterwards: the op is
    // appended first so that its regions have a parent.
    OpId region_op(OpKind kind,
                   std::vector<ValueId> operands,
                   const std::vector<TypeId>& results,
                   Attributes attributes,
                   SourceSpan span) {
        return module_.add_op(
            block_, kind, std::move(operands), results, std::move(attributes), span);
    }

    void yield(std::vector<ValueId> values) {
        module_.add_op(block_, OpKind::Yield, std::move(values), {});
    }

    // --------------------------------------------------------------- functions

    std::string qualified_name(EntityId entity) {
        const Entity& target = model().entities[entity];
        std::string name;
        for (const ast::Name& segment : modules_[target.module]->module_path) {
            name += name.empty() ? "" : ".";
            name += segment.text;
        }
        name += "::";
        if (target.parent != no_entity) {
            name += std::string(model().entities[target.parent].name) + ".";
        }
        return name + std::string(target.name);
    }

    TypeId self_type(EntityId block) {
        const DeclInfo& info = model().decls.at(block);
        std::vector<GenericValue> args;
        for (const GenericInfo& generic : info.generics) {
            GenericValue value;
            switch (generic.kind) {
            case GenericKind::Dim:
                value.kind = GenericValue::Kind::Dim;
                value.dim = model().dims.symbol(generic.symbol);
                break;
            case GenericKind::Pack:
                value.kind = GenericValue::Kind::Pack;
                value.shape = {ShapeElem::of_pack(generic.symbol)};
                break;
            case GenericKind::DType:
                value.kind = GenericValue::Kind::DType;
                value.dtype = DType::variable(generic.dtype_var);
                break;
            }
            args.push_back(std::move(value));
        }
        return types().nominal(TypeKind::Block, block, std::move(args));
    }

    void lower_function(EntityId entity) {
        const Entity& target = model().entities[entity];
        const DeclInfo& info = model().decls.at(entity);
        const auto& decl =
            std::get<ast::FunctionDecl>(modules_[target.module]->item(target.item).data);

        Function function;
        function.name = qualified_name(entity);
        function.is_op = decl.kind == ast::FunctionKind::Op;
        function.is_entry = decl.kind == ast::FunctionKind::Entry;
        function.entity = entity;
        function.generics = info.generics;
        function.constraints = info.constraints;
        if (info.result != no_type && types().kind(info.result) != TypeKind::Unit) {
            function.results.push_back(info.result);
        }
        function.body = module_.add_region(no_id);
        const FunctionId id = module_.add_function(function);

        Frame frame;
        frame.module = target.module;
        frame.function = entity;
        frame.block_entity = target.parent;
        frame.results = function.results;
        frame_ = &frame;
        block_ = module_.add_block(function.body);
        if (target.parent != no_entity) {
            frame.self = module_.add_argument(block_, self_type(target.parent), "self");
        }
        for (std::size_t i = 0; i < info.params.size(); ++i) {
            const ValueId argument =
                module_.add_argument(block_, info.params[i].type, std::string(info.params[i].name));
            frame.locals[info.param_entities[i]] = argument;
        }
        lower_body(decl.body);
        // A function without a result ends in an implicit return.
        const Block& body = module_.block(block_);
        if (body.ops.empty() || module_.op(body.ops.back()).kind != OpKind::Return) {
            module_.add_op(block_, OpKind::Return, {}, {});
        }
        frame_ = nullptr;
        (void)id;
    }

    // -------------------------------------------------------------- statements

    void lower_body(const std::vector<ast::StmtId>& body) {
        for (const ast::StmtId id : body) {
            lower_stmt(id);
        }
    }

    // Locals assigned anywhere inside `body`, in first-assignment order.
    void assigned_locals(const std::vector<ast::StmtId>& body, std::vector<EntityId>& out) {
        for (const ast::StmtId id : body) {
            const ast::Stmt& node = ast().stmt(id);
            if (const auto* assign = std::get_if<ast::AssignStmt>(&node.data)) {
                for (const auto& [entity, value] : frame_->locals) {
                    const Entity& local = model().entities[entity];
                    if (local.name == assign->target.text && local.is_mutable) {
                        if (std::find(out.begin(), out.end(), entity) == out.end()) {
                            out.push_back(entity);
                        }
                    }
                }
            } else if (const auto* loop = std::get_if<ast::StaticForStmt>(&node.data)) {
                assigned_locals(loop->body, out);
            }
        }
    }

    void lower_stmt(ast::StmtId id) {
        const ast::Stmt& node = ast().stmt(id);
        const StmtFacts& stmt_facts = module_.model().stmt(frame_->module, id);
        std::visit(
            Overloaded{
                [&](const ast::ErrorStmt&) {},
                [&](const ast::LetStmt& let) {
                    const TypeId declared =
                        let.type == ast::no_id ? no_type : pattern_type(let.pattern);
                    const ValueId value = lower_expr(let.value, declared);
                    bind_pattern(let.pattern, value);
                },
                [&](const ast::VarStmt& var) {
                    const TypeId declared = substituted(stmt_facts.type);
                    const ValueId value = lower_expr(var.value, declared);
                    module_.value(value).name = std::string(var.name.text);
                    frame_->locals[stmt_facts.entity] = value;
                },
                [&](const ast::ComprehensionStmt& comprehension) {
                    const ValueId value = lower_comprehension(comprehension, stmt_facts, node.span);
                    module_.value(value).name = std::string(comprehension.target.text);
                    frame_->locals[stmt_facts.entity] = value;
                },
                [&](const ast::AssignStmt& assign) {
                    const EntityId entity = local_named(assign.target.text);
                    const TypeId type = substituted(model().entities[entity].type);
                    frame_->locals[entity] = lower_expr(assign.value, type);
                },
                [&](const ast::ReturnStmt& ret) {
                    std::vector<ValueId> values;
                    if (ret.value != ast::no_id) {
                        values.push_back(lower_expr(
                            ret.value,
                            frame_->results.empty() ? no_type : frame_->results.front()));
                    }
                    module_.add_op(block_, OpKind::Return, std::move(values), {}, {}, node.span);
                },
                [&](const ast::StaticForStmt& loop) { lower_static_for(loop, node.span); },
            },
            node.data);
    }

    EntityId local_named(std::string_view name) {
        EntityId found = no_entity;
        for (const auto& [entity, value] : frame_->locals) {
            if (model().entities[entity].name == name) {
                found = entity; // the latest declaration wins
            }
        }
        return found;
    }

    TypeId pattern_type(ast::PatternId id) {
        const ast::Pattern& pattern = ast().pattern(id);
        if (const auto* binding = std::get_if<ast::BindingPattern>(&pattern.data)) {
            (void)binding;
            const auto found = module_.model().bound[frame_->module].find(id);
            return found == module_.model().bound[frame_->module].end()
                       ? no_type
                       : substituted(model().entities[found->second].type);
        }
        if (const auto* tuple = std::get_if<ast::TuplePattern>(&pattern.data)) {
            std::vector<TypeId> elements;
            elements.reserve(tuple->elements.size());
            for (const ast::PatternId element : tuple->elements) {
                elements.push_back(pattern_type(element));
            }
            return types().tuple(std::move(elements));
        }
        return no_type;
    }

    void bind_pattern(ast::PatternId id, ValueId value) {
        const ast::Pattern& pattern = ast().pattern(id);
        if (std::holds_alternative<ast::BindingPattern>(pattern.data)) {
            const auto found = module_.model().bound[frame_->module].find(id);
            if (found != module_.model().bound[frame_->module].end()) {
                const auto& binding = std::get<ast::BindingPattern>(pattern.data);
                if (module_.value(value).name.empty()) {
                    module_.value(value).name = std::string(binding.name.text);
                }
                frame_->locals[found->second] = value;
            }
            return;
        }
        if (const auto* tuple = std::get_if<ast::TuplePattern>(&pattern.data)) {
            const TypeData& data = types().get(module_.value(value).type);
            for (std::size_t i = 0; i < tuple->elements.size(); ++i) {
                Attributes attributes;
                attributes.integer = static_cast<std::int64_t>(i);
                const TypeId element =
                    i < data.elements.size() ? data.elements[i] : types().error();
                bind_pattern(tuple->elements[i],
                             emit(OpKind::TupleGet, {value}, element, attributes, pattern.span));
            }
        }
    }

    void lower_static_for(const ast::StaticForStmt& loop, SourceSpan span) {
        const ValueId array = lower_expr(loop.iterable, no_type);
        const TypeData& array_type = types().get(module_.value(array).type);
        const TypeId element =
            array_type.elements.empty() ? types().error() : array_type.elements.front();

        std::vector<EntityId> carried;
        assigned_locals(loop.body, carried);
        std::vector<ValueId> operands{array};
        std::vector<TypeId> results;
        std::vector<std::pair<TypeId, std::string>> arguments{{element, "element"}};
        for (const EntityId entity : carried) {
            operands.push_back(frame_->locals.at(entity));
            results.push_back(module_.value(frame_->locals.at(entity)).type);
            arguments.emplace_back(results.back(), std::string(model().entities[entity].name));
        }
        const OpId op = region_op(OpKind::StaticFor, operands, results, {}, span);
        const RegionId region = nested_region(op, arguments, [&](const std::vector<ValueId>& args) {
            bind_pattern(loop.pattern, args.front());
            for (std::size_t i = 0; i < carried.size(); ++i) {
                frame_->locals[carried[i]] = args[i + 1];
            }
            lower_body(loop.body);
            std::vector<ValueId> yielded;
            yielded.reserve(carried.size());
            for (const EntityId entity : carried) {
                yielded.push_back(frame_->locals.at(entity));
            }
            yield(std::move(yielded));
        });
        module_.op(op).regions.push_back(region);
        for (std::size_t i = 0; i < carried.size(); ++i) {
            frame_->locals[carried[i]] = module_.op(op).results[i];
        }
    }

    // ---------------------------------------------------------- comprehension

    // Block arguments for a list of index names and their domains: plain
    // indices are i64 positions, pack indices are positions in their shape.
    std::vector<std::pair<TypeId, std::string>>
    index_arguments(const std::vector<std::string>& names, const std::vector<Shape>& domains) {
        std::vector<std::pair<TypeId, std::string>> arguments;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const Shape domain = types().substitute(domains[i], frame_->substitution);
            const bool is_pack = domain.size() != 1 || domain.front().is_pack;
            arguments.emplace_back(
                is_pack ? types().shape_value(domain) : types().scalar(ScalarKind::I64), names[i]);
        }
        return arguments;
    }

    Attributes index_attributes(const std::vector<std::string>& names,
                                const std::vector<Shape>& domains) {
        Attributes attributes;
        attributes.names = names;
        for (const Shape& domain : domains) {
            const Shape substituted_domain = types().substitute(domain, frame_->substitution);
            // A plain index has one unit; a pack index is recorded as its
            // pack when it is exactly one, otherwise by its first unit only.
            attributes.shape.push_back(substituted_domain.empty() ? ShapeElem{}
                                                                  : substituted_domain.front());
        }
        return attributes;
    }

    ValueId lower_comprehension(const ast::ComprehensionStmt& comprehension,
                                const StmtFacts& stmt_facts,
                                SourceSpan span) {
        std::vector<std::string> names;
        names.reserve(comprehension.outputs.size());
        for (const ast::IndexOutput& output : comprehension.outputs) {
            names.push_back(std::string(output.name.text));
        }
        const TypeId result = substituted(stmt_facts.type);
        const OpId op = region_op(OpKind::Comprehension,
                                  {},
                                  {result},
                                  index_attributes(names, stmt_facts.output_domains),
                                  span);
        const RegionId region =
            nested_region(op,
                          index_arguments(names, stmt_facts.output_domains),
                          [&](const std::vector<ValueId>& args) {
                              frame_->index_scopes.emplace_back();
                              for (std::size_t i = 0; i < names.size(); ++i) {
                                  frame_->index_scopes.back()[comprehension.outputs[i].name.text] =
                                      args[i];
                              }
                              const TypeId element = types().scalar(types().get(result).dtype);
                              yield({lower_expr(comprehension.value, element)});
                              frame_->index_scopes.pop_back();
                          });
        module_.op(op).regions.push_back(region);
        return module_.op(op).results.front();
    }

    ValueId
    lower_reduction(const ast::Expr& node, const ast::ReductionExpr& reduction, ast::ExprId id) {
        const ExprFacts& reduction_facts = facts(id);
        std::vector<std::string> names;
        names.reserve(reduction.indices.size());
        for (const ast::Name& index : reduction.indices) {
            names.push_back(std::string(index.text));
        }
        const TypeId result = value_type(id, no_type);
        Attributes attributes = index_attributes(names, reduction_facts.index_domains);
        attributes.reduce = reduction.kind == ast::ReductionKind::Sum    ? ReduceKind::Sum
                            : reduction.kind == ast::ReductionKind::Prod ? ReduceKind::Prod
                            : reduction.kind == ast::ReductionKind::Max  ? ReduceKind::Max
                            : reduction.kind == ast::ReductionKind::Min  ? ReduceKind::Min
                            : reduction.kind == ast::ReductionKind::Any  ? ReduceKind::Any
                                                                         : ReduceKind::All;
        const OpId op = region_op(OpKind::Reduce, {}, {result}, attributes, node.span);
        const RegionId region =
            nested_region(op,
                          index_arguments(names, reduction_facts.index_domains),
                          [&](const std::vector<ValueId>& args) {
                              frame_->index_scopes.emplace_back();
                              for (std::size_t i = 0; i < names.size(); ++i) {
                                  frame_->index_scopes.back()[reduction.indices[i].text] = args[i];
                              }
                              yield({lower_expr(reduction.body, result)});
                              frame_->index_scopes.pop_back();
                          });
        module_.op(op).regions.push_back(region);
        return module_.op(op).results.front();
    }

    ValueId find_index(std::string_view name) const {
        for (auto scope = frame_->index_scopes.rbegin(); scope != frame_->index_scopes.rend();
             ++scope) {
            if (const auto found = scope->find(name); found != scope->end()) {
                return found->second;
            }
        }
        return no_id;
    }

    // ------------------------------------------------------------- expressions

    ValueId lower_expr(ast::ExprId id, TypeId expected) {
        const ast::Expr& node = ast().expr(id);
        const TypeId type = value_type(id, expected);
        return std::visit(
            Overloaded{
                [&](const ast::ErrorExpr&) { return emit(OpKind::ConstInt, {}, types().error()); },
                [&](const ast::LiteralExpr& literal) { return lower_literal(node, literal, type); },
                [&](const ast::NameExpr&) { return lower_name(node, facts(id).entity, type); },
                [&](const ast::NoneExpr&) {
                    return emit(OpKind::OptionNone, {}, type, {}, node.span);
                },
                [&](const ast::SomeExpr& some) {
                    const TypeData& data = types().get(type);
                    const TypeId inner =
                        data.kind == TypeKind::Optional ? data.elements.front() : no_type;
                    return emit(
                        OpKind::OptionSome, {lower_expr(some.value, inner)}, type, {}, node.span);
                },
                [&](const ast::ParenExpr& paren) { return lower_expr(paren.inner, expected); },
                [&](const ast::TupleExpr& tuple) {
                    const TypeData& data = types().get(type);
                    std::vector<ValueId> elements;
                    elements.reserve(tuple.elements.size());
                    for (std::size_t i = 0; i < tuple.elements.size(); ++i) {
                        elements.push_back(
                            lower_expr(tuple.elements[i],
                                       i < data.elements.size() ? data.elements[i] : no_type));
                    }
                    return emit(OpKind::TupleMake, std::move(elements), type, {}, node.span);
                },
                [&](const ast::ShapeExpr&) {
                    // A shape literal only reaches here as an argument that
                    // the callee consumed as an attribute.
                    return emit(OpKind::ConstInt, {}, type);
                },
                [&](const ast::UnaryExpr& unary) { return lower_unary(node, unary, type); },
                [&](const ast::BinaryExpr& binary) { return lower_binary(node, binary, type); },
                [&](const ast::CallExpr& call) { return lower_call(node, call, id, type); },
                [&](const ast::IndexExpr& index) { return lower_index(node, index, id, type); },
                [&](const ast::MemberExpr& member) { return lower_member(node, member, id, type); },
                [&](const ast::ReductionExpr& reduction) {
                    return lower_reduction(node, reduction, id);
                },
                [&](const ast::IfExpr& conditional) {
                    const ValueId condition =
                        lower_expr(conditional.condition, types().scalar(ScalarKind::Bool));
                    const OpId op = region_op(OpKind::If, {condition}, {type}, {}, node.span);
                    for (const ast::ExprId branch :
                         {conditional.then_value, conditional.else_value}) {
                        const RegionId region =
                            nested_region(op, {}, [&](const std::vector<ValueId>&) {
                                yield({lower_expr(branch, type)});
                            });
                        module_.op(op).regions.push_back(region);
                    }
                    return module_.op(op).results.front();
                },
                [&](const ast::MatchExpr& match) { return lower_match(node, match, type); },
            },
            node.data);
    }

    ValueId lower_literal(const ast::Expr& node, const ast::LiteralExpr& literal, TypeId type) {
        const std::string_view text = sources_.text(node.span);
        Attributes attributes;
        switch (literal.kind) {
        case ast::LiteralKind::Integer: {
            const DType dtype = types().get(type).dtype;
            if (!dtype.is_var && is_float(dtype.scalar)) {
                attributes.number = static_cast<double>(parse_integer(text).value_or(0));
                return emit(OpKind::ConstFloat, {}, type, attributes, node.span);
            }
            attributes.integer = parse_integer(text).value_or(0);
            return emit(OpKind::ConstInt, {}, type, attributes, node.span);
        }
        case ast::LiteralKind::Float: {
            std::string digits(text);
            std::erase(digits, '_');
            attributes.number = std::strtod(digits.c_str(), nullptr);
            return emit(OpKind::ConstFloat, {}, type, attributes, node.span);
        }
        case ast::LiteralKind::True:
        case ast::LiteralKind::False:
            attributes.integer = literal.kind == ast::LiteralKind::True ? 1 : 0;
            return emit(OpKind::ConstBool, {}, type, attributes, node.span);
        case ast::LiteralKind::String:
            return emit(OpKind::ConstInt, {}, types().error());
        }
        return emit(OpKind::ConstInt, {}, types().error());
    }

    ValueId lower_name(const ast::Expr& node, EntityId entity, TypeId type) {
        if (entity == no_entity) {
            return emit(OpKind::ConstInt, {}, types().error());
        }
        const Entity& target = model().entities[entity];
        switch (target.kind) {
        case EntityKind::Local:
            return frame_->locals.at(entity);
        case EntityKind::Member: {
            const auto& decl =
                std::get<ast::MemberDecl>(modules_[target.module]->item(target.item).data);
            Attributes attributes;
            attributes.name = std::string(target.name);
            return emit(decl.kind == ast::MemberKind::Sub ? OpKind::BlockSub : OpKind::BlockParam,
                        {frame_->self},
                        type,
                        attributes,
                        node.span);
        }
        case EntityKind::GenericDim:
            return const_dim(model().dims.symbol(target.symbol));
        case EntityKind::Const: {
            const TypeData& data = types().get(type);
            if (data.kind == TypeKind::CompileInt) {
                return const_dim(data.value);
            }
            // The constant's expression is lowered where it is used, in its
            // own module's context.
            const auto& decl =
                std::get<ast::ConstDecl>(modules_[target.module]->item(target.item).data);
            Frame* outer = frame_;
            Frame frame;
            frame.module = target.module;
            frame_ = &frame;
            const ValueId value = lower_expr(decl.value, type);
            frame_ = outer;
            return value;
        }
        default:
            return emit(OpKind::ConstInt, {}, types().error());
        }
    }

    ValueId lower_unary(const ast::Expr& node, const ast::UnaryExpr& unary, TypeId type) {
        const ValueId operand = lower_expr(unary.operand, type);
        switch (unary.op) {
        case ast::UnaryOp::Not:
            return emit(OpKind::Not, {operand}, type, {}, node.span);
        case ast::UnaryOp::Negate:
            return emit(OpKind::Neg, {operand}, type, {}, node.span);
        case ast::UnaryOp::Plus:
            return operand;
        }
        return operand;
    }

    // The type the other operand of an elementwise operator gives a literal.
    TypeId operand_context(ast::ExprId other, TypeId result) {
        const TypeId type = substituted(facts(other).type);
        const TypeKind kind = types().kind(type);
        if (kind == TypeKind::Scalar || kind == TypeKind::Tensor) {
            return types().scalar(types().get(type).dtype);
        }
        const TypeKind result_kind = types().kind(result);
        if (result_kind == TypeKind::Scalar || result_kind == TypeKind::Tensor) {
            const DType dtype = types().get(result).dtype;
            return dtype == DType::of(ScalarKind::Bool) ? no_type : types().scalar(dtype);
        }
        return no_type;
    }

    ValueId lower_binary(const ast::Expr& node, const ast::BinaryExpr& binary, TypeId type) {
        const ValueId lhs = lower_expr(binary.lhs, operand_context(binary.rhs, type));
        const ValueId rhs = lower_expr(binary.rhs, operand_context(binary.lhs, type));
        Attributes attributes;
        OpKind kind = OpKind::Add;
        switch (binary.op) {
        case ast::BinaryOp::Or:
            kind = OpKind::Or;
            break;
        case ast::BinaryOp::And:
            kind = OpKind::And;
            break;
        case ast::BinaryOp::Add:
            kind = OpKind::Add;
            break;
        case ast::BinaryOp::Subtract:
            kind = OpKind::Sub;
            break;
        case ast::BinaryOp::Multiply:
            kind = OpKind::Mul;
            break;
        case ast::BinaryOp::Divide:
            kind = OpKind::Div;
            break;
        case ast::BinaryOp::Remainder:
            kind = OpKind::Rem;
            break;
        default:
            kind = OpKind::Compare;
            attributes.compare = binary.op == ast::BinaryOp::Equal       ? CompareKind::Eq
                                 : binary.op == ast::BinaryOp::NotEqual  ? CompareKind::Ne
                                 : binary.op == ast::BinaryOp::Less      ? CompareKind::Lt
                                 : binary.op == ast::BinaryOp::LessEqual ? CompareKind::Le
                                 : binary.op == ast::BinaryOp::Greater   ? CompareKind::Gt
                                                                         : CompareKind::Ge;
            break;
        }
        return emit(kind, {lhs, rhs}, type, attributes, node.span);
    }

    ValueId lower_member(const ast::Expr& node,
                         const ast::MemberExpr& member,
                         ast::ExprId id,
                         TypeId type) {
        const ExprFacts& member_facts = facts(id);
        if (member_facts.is_field) {
            const ValueId base = lower_expr(member.base, no_type);
            Attributes attributes;
            attributes.integer = member_facts.field;
            return emit(OpKind::StructGet, {base}, type, attributes, node.span);
        }
        if (member_facts.entity != no_entity) {
            const Entity& target = model().entities[member_facts.entity];
            if (target.kind == EntityKind::Member) {
                const ValueId base = lower_expr(member.base, no_type);
                const auto& decl =
                    std::get<ast::MemberDecl>(modules_[target.module]->item(target.item).data);
                Attributes attributes;
                attributes.name = std::string(target.name);
                return emit(decl.kind == ast::MemberKind::Sub ? OpKind::BlockSub
                                                              : OpKind::BlockParam,
                            {base},
                            type,
                            attributes,
                            node.span);
            }
            return lower_name(node, member_facts.entity, type);
        }
        // An enum variant.
        if (types().kind(type) == TypeKind::Enum) {
            Attributes attributes;
            attributes.name = std::string(member.member.text);
            return emit(OpKind::EnumConst, {}, type, attributes, node.span);
        }
        return emit(OpKind::ConstInt, {}, types().error());
    }

    ValueId lower_match(const ast::Expr& node, const ast::MatchExpr& match, TypeId type) {
        const ValueId scrutinee = lower_expr(match.scrutinee, no_type);
        const TypeData& subject = types().get(module_.value(scrutinee).type);
        if (subject.kind == TypeKind::Optional) {
            const ast::MatchArm* some_arm = nullptr;
            const ast::MatchArm* none_arm = nullptr;
            const ast::MatchArm* other_arm = nullptr;
            for (const ast::MatchArm& arm : match.arms) {
                const ast::Pattern& pattern = ast().pattern(arm.pattern);
                if (std::holds_alternative<ast::SomePattern>(pattern.data)) {
                    some_arm = some_arm ? some_arm : &arm;
                } else if (std::holds_alternative<ast::NonePattern>(pattern.data)) {
                    none_arm = none_arm ? none_arm : &arm;
                } else {
                    other_arm = other_arm ? other_arm : &arm;
                }
            }
            const OpId op = region_op(OpKind::OptionMatch, {scrutinee}, {type}, {}, node.span);
            const TypeId inner = subject.elements.front();
            const RegionId some_region =
                nested_region(op, {{inner, "value"}}, [&](const std::vector<ValueId>& args) {
                    if (some_arm != nullptr) {
                        bind_pattern(
                            std::get<ast::SomePattern>(ast().pattern(some_arm->pattern).data).inner,
                            args.front());
                        yield({lower_expr(some_arm->value, type)});
                    } else {
                        bind_pattern(other_arm->pattern, scrutinee);
                        yield({lower_expr(other_arm->value, type)});
                    }
                });
            const RegionId none_region = nested_region(op, {}, [&](const std::vector<ValueId>&) {
                const ast::MatchArm* arm = none_arm != nullptr ? none_arm : other_arm;
                if (arm == other_arm) {
                    bind_pattern(arm->pattern, scrutinee);
                }
                yield({lower_expr(arm->value, type)});
            });
            module_.op(op).regions = {some_region, none_region};
            return module_.op(op).results.front();
        }
        // Enum: one region per arm, in source order; a binding arm is a
        // catch-all named "_".
        Attributes attributes;
        const OpId op = region_op(OpKind::EnumMatch, {scrutinee}, {type}, {}, node.span);
        for (const ast::MatchArm& arm : match.arms) {
            const ast::Pattern& pattern = ast().pattern(arm.pattern);
            const auto* binding = std::get_if<ast::BindingPattern>(&pattern.data);
            const bool is_catch_all =
                binding != nullptr && module_.model().bound[frame_->module].contains(arm.pattern);
            attributes.names.push_back(
                is_catch_all || binding == nullptr ? "_" : std::string(binding->name.text));
            const RegionId region = nested_region(op, {}, [&](const std::vector<ValueId>&) {
                if (is_catch_all) {
                    bind_pattern(arm.pattern, scrutinee);
                }
                yield({lower_expr(arm.value, type)});
            });
            module_.op(op).regions.push_back(region);
        }
        module_.op(op).attributes = attributes;
        return module_.op(op).results.front();
    }

    // ------------------------------------------------------------------ index

    ValueId
    lower_index(const ast::Expr& node, const ast::IndexExpr& index, ast::ExprId id, TypeId type) {
        const ValueId base = lower_expr(index.base, no_type);
        const TypeData& data = types().get(module_.value(base).type);
        if (data.kind == TypeKind::Array) {
            const ValueId position =
                lower_expr(index.components.front().value, types().scalar(ScalarKind::I64));
            return emit(OpKind::ArrayGet, {base, position}, type, {}, node.span);
        }
        const bool is_element_access =
            !frame_->index_scopes.empty() && types().kind(type) == TypeKind::Scalar &&
            std::all_of(
                index.components.begin(), index.components.end(), [](const ast::IndexComponent& c) {
                    return c.kind == ast::IndexKind::Expr || c.kind == ast::IndexKind::Pack;
                });
        if (is_element_access) {
            std::vector<ValueId> operands{base};
            for (const ast::IndexComponent& component : index.components) {
                if (component.kind == ast::IndexKind::Pack) {
                    operands.push_back(find_index(component.pack.text));
                    continue;
                }
                const auto* name = std::get_if<ast::NameExpr>(&ast().expr(component.value).data);
                const ValueId variable = name != nullptr ? find_index(name->name.text) : no_id;
                operands.push_back(variable != no_id ? variable
                                                     : lower_expr(component.value,
                                                                  types().scalar(ScalarKind::I64)));
            }
            return emit(OpKind::Element, std::move(operands), type, {}, node.span);
        }
        // Slicing: one entry per axis of the base tensor.
        Attributes attributes;
        std::size_t explicit_count = 0;
        for (const ast::IndexComponent& component : index.components) {
            explicit_count += component.kind == ast::IndexKind::Ellipsis ? 0 : 1;
        }
        std::size_t axis = 0;
        const auto full_axis = [&] {
            const ShapeElem& unit = data.shape[axis];
            attributes.starts.push_back(shape::Poly(0));
            attributes.stops.push_back(unit.is_pack ? shape::Poly(0) : unit.dim);
            attributes.steps.push_back(1);
            attributes.squeezed.push_back(false);
            attributes.whole.push_back(true);
            ++axis;
        };
        for (const ast::IndexComponent& component : index.components) {
            if (component.kind == ast::IndexKind::Ellipsis) {
                const std::size_t kept = data.shape.size() - explicit_count;
                for (std::size_t i = 0; i < kept; ++i) {
                    full_axis();
                }
                continue;
            }
            const ShapeElem& unit = data.shape[axis];
            const auto dim_of = [&](ast::ExprId expr, const shape::Poly& fallback) {
                if (expr == ast::no_id) {
                    return fallback;
                }
                const TypeData& value = types().get(substituted(facts(expr).type));
                return value.kind == TypeKind::CompileInt ? value.value : fallback;
            };
            if (component.kind == ast::IndexKind::Expr) {
                const shape::Poly position = dim_of(component.value, shape::Poly::invalid());
                attributes.starts.push_back(position);
                attributes.stops.push_back(position + shape::Poly(1));
                attributes.steps.push_back(1);
                attributes.squeezed.push_back(true);
                attributes.whole.push_back(false);
            } else {
                attributes.starts.push_back(dim_of(component.start, shape::Poly(0)));
                attributes.stops.push_back(dim_of(component.stop, unit.dim));
                attributes.steps.push_back(
                    dim_of(component.step, shape::Poly(1)).constant().value_or(1));
                attributes.squeezed.push_back(false);
                attributes.whole.push_back(false);
            }
            ++axis;
        }
        while (axis < data.shape.size()) {
            full_axis();
        }
        (void)id;
        return emit(OpKind::Slice, {base}, type, attributes, node.span);
    }

    // ------------------------------------------------------------------ calls

    ValueId
    lower_call(const ast::Expr& node, const ast::CallExpr& call, ast::ExprId id, TypeId type) {
        const ExprFacts& call_facts = facts(id);
        if (!call_facts.builtin.empty()) {
            return lower_builtin(node, call, call_facts.builtin, type);
        }
        const EntityId callee = call_facts.entity;
        if (callee == no_entity) {
            return emit(OpKind::ConstInt, {}, types().error());
        }
        const Entity& target = model().entities[callee];
        const DeclInfo& info = model().decls.at(callee);
        const auto& decl =
            std::get<ast::FunctionDecl>(modules_[target.module]->item(target.item).data);

        // Generic bindings of this call, on top of the frame's own.
        Substitution substitution = call_facts.substitution;
        for (auto& [symbol, dim] : substitution.dims) {
            dim = types().substitute(dim, frame_->substitution);
        }
        for (auto& [symbol, shape] : substitution.packs) {
            shape = types().substitute(shape, frame_->substitution);
        }
        for (auto& [var, dtype] : substitution.dtypes) {
            dtype = types().substitute(dtype, frame_->substitution);
        }

        std::vector<ValueId> operands;
        if (target.parent != no_entity) {
            // A method call: the receiver is the block value.
            const auto* member = std::get_if<ast::MemberExpr>(&ast().expr(call.callee).data);
            operands.push_back(member != nullptr ? lower_expr(member->base, no_type)
                                                 : frame_->self);
        }
        std::vector<ValueId> by_slot(info.params.size(), no_id);
        for (std::size_t i = 0; i < call.args.size() && i < call_facts.argument_slots.size(); ++i) {
            const std::uint32_t slot = call_facts.argument_slots[i];
            const TypeId param_type = types().substitute(info.params[slot].type, substitution);
            by_slot[slot] = lower_expr(call.args[i].value, param_type);
        }
        for (std::size_t slot = 0; slot < by_slot.size(); ++slot) {
            if (by_slot[slot] != no_id) {
                continue;
            }
            // A defaulted parameter: its expression lives in the callee.
            const TypeId param_type = types().substitute(info.params[slot].type, substitution);
            Frame* outer = frame_;
            Frame frame;
            frame.module = target.module;
            frame.substitution = substitution;
            frame_ = &frame;
            by_slot[slot] = lower_expr(decl.parameters[slot].default_value, param_type);
            frame_ = outer;
        }
        operands.insert(operands.end(), by_slot.begin(), by_slot.end());

        Attributes attributes;
        attributes.name = qualified_name(callee);
        attributes.substitution = substitution;
        const OpKind kind =
            decl.kind == ast::FunctionKind::Op ? OpKind::SemanticCall : OpKind::Call;
        const TypeId result = types().kind(type) == TypeKind::Unit ? no_type : type;
        const ValueId value = emit(kind, std::move(operands), result, attributes, node.span);
        return value == no_id ? emit(OpKind::ConstInt, {}, types().unit()) : value;
    }

    Shape shape_argument(ast::ExprId expr) {
        const TypeData& data = types().get(substituted(facts(expr).type));
        return data.kind == TypeKind::ShapeValue ? data.shape : Shape{};
    }

    ValueId lower_builtin(const ast::Expr& node,
                          const ast::CallExpr& call,
                          const std::string& name,
                          TypeId type) {
        const auto arg = [&](std::size_t i, TypeId expected) {
            return lower_expr(call.args[i].value, expected);
        };
        Attributes attributes;
        static const std::map<std::string, OpKind> unary{{"exp", OpKind::Exp},
                                                         {"log", OpKind::Log},
                                                         {"sqrt", OpKind::Sqrt},
                                                         {"rsqrt", OpKind::Rsqrt},
                                                         {"sin", OpKind::Sin},
                                                         {"cos", OpKind::Cos},
                                                         {"tanh", OpKind::Tanh},
                                                         {"abs", OpKind::Abs}};
        if (const auto found = unary.find(name); found != unary.end()) {
            return emit(found->second, {arg(0, type)}, type, {}, node.span);
        }
        if (name == "cast") {
            return emit(OpKind::Cast, {arg(0, no_type)}, type, {}, node.span);
        }
        if (name == "min" || name == "max") {
            if (types().kind(type) == TypeKind::CompileInt) {
                return const_dim(types().get(type).value);
            }
            const TypeId context = types().scalar(types().get(type).dtype);
            return emit(name == "min" ? OpKind::Min : OpKind::Max,
                        {arg(0, context), arg(1, context)},
                        type,
                        {},
                        node.span);
        }
        if (name == "select") {
            const TypeId context = types().scalar(types().get(type).dtype);
            return emit(
                OpKind::Select,
                {arg(0, types().scalar(ScalarKind::Bool)), arg(1, context), arg(2, context)},
                type,
                {},
                node.span);
        }
        if (name == "reshape" || name == "broadcast_to") {
            attributes.shape = types().get(type).shape;
            return emit(name == "reshape" ? OpKind::Reshape : OpKind::Broadcast,
                        {arg(0, no_type)},
                        type,
                        attributes,
                        node.span);
        }
        if (name == "permute") {
            attributes.shape = shape_argument(call.args[1].value);
            return emit(OpKind::Permute, {arg(0, no_type)}, type, attributes, node.span);
        }
        if (name == "concat") {
            std::vector<ValueId> operands;
            for (const ast::Argument& argument : call.args) {
                if (argument.keyword.text == "axis") {
                    const TypeData& axis = types().get(substituted(facts(argument.value).type));
                    attributes.axis = axis.value.constant().value_or(0);
                } else {
                    operands.push_back(lower_expr(argument.value, no_type));
                }
            }
            return emit(OpKind::Concat, std::move(operands), type, attributes, node.span);
        }
        if (name == "iota") {
            attributes.shape = types().get(type).shape;
            return emit(OpKind::Iota, {}, type, attributes, node.span);
        }
        if (name == "fill") {
            attributes.shape = types().get(type).shape;
            const TypeId context = types().scalar(types().get(type).dtype);
            return emit(OpKind::Fill, {arg(1, context)}, type, attributes, node.span);
        }
        return emit(OpKind::ConstInt, {}, types().error());
    }

    const SourceManager& sources_;
    std::vector<const ast::Ast*> modules_;
    Module module_;
    Frame* frame_ = nullptr;
    BlockId block_ = no_id;
};

} // namespace

Module lower(const SourceManager& sources,
             std::span<const ast::Ast* const> modules,
             std::shared_ptr<sema::Model> model) {
    return Lowerer(sources, modules, std::move(model)).run();
}

} // namespace linnet::ir
