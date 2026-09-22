#include "linnet/backend/plan.hpp"

#include "linnet/diagnostic/json.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace linnet::backend {

namespace {

using namespace sema;

class PlanWriter {
public:
    PlanWriter(ir::Module& module, const PlanOptions& options)
        : module_(module), model_(module.model()), types_(module.types()), options_(options) {}

    std::expected<std::string, std::string> run() {
        const auto root = find_root();
        if (!root) {
            return std::unexpected(root.error());
        }
        std::string out = "{\"version\":1";
        out += ",\"root\":" + block_description(*root);
        out += ",\"manifest\":" + manifest(*root);
        out += ",\"blocks\":" + blocks_json();
        out += ",\"functions\":[";
        bool is_first = true;
        for (const ir::Function& function : module_.functions()) {
            out += is_first ? "" : ",";
            is_first = false;
            out += function_json(function);
        }
        out += "]}\n";
        return out;
    }

private:
    // ------------------------------------------------------------------ root

    std::expected<EntityId, std::string> find_root() const {
        std::vector<EntityId> candidates;
        for (EntityId id = 0; id < model_.entities.size(); ++id) {
            const Entity& entity = model_.entities[id];
            if (entity.kind != EntityKind::Block || entity.parent != no_entity ||
                entity.module != options_.root_module) {
                continue;
            }
            if (!options_.root.empty()) {
                if (entity.name == options_.root) {
                    return id;
                }
                continue;
            }
            if (has_entry(id)) {
                candidates.push_back(id);
            }
        }
        if (!options_.root.empty()) {
            return std::unexpected("no block named `" + options_.root + "` in this file");
        }
        if (candidates.size() != 1) {
            return std::unexpected(candidates.empty()
                                       ? "no block with an `entry`; name one with --root"
                                       : "several blocks have entries; name one with --root");
        }
        return candidates.front();
    }

    bool has_entry(EntityId block) const {
        for (const ir::Function& function : module_.functions()) {
            if (function.is_entry && model_.entities[function.entity].parent == block) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------ dimensions

    std::string dim(const shape::Poly& poly) const {
        if (!poly.is_valid()) {
            return "null";
        }
        if (const auto constant = poly.constant()) {
            return std::to_string(*constant);
        }
        std::string terms;
        for (const shape::Term& term : poly.terms()) {
            std::vector<std::string> factors;
            if (term.coefficient != 1 || term.atoms.empty()) {
                factors.push_back(std::to_string(term.coefficient));
            }
            for (const shape::AtomId id : term.atoms) {
                factors.push_back(atom(id));
            }
            std::string product = factors.front();
            if (factors.size() > 1) {
                product = "{\"op\":\"mul\",\"args\":[";
                for (std::size_t i = 0; i < factors.size(); ++i) {
                    product += (i == 0 ? "" : ",") + factors[i];
                }
                product += "]}";
            }
            terms += (terms.empty() ? "" : ",") + product;
        }
        return poly.terms().size() == 1 ? terms : "{\"op\":\"add\",\"args\":[" + terms + "]}";
    }

    std::string atom(shape::AtomId id) const {
        const shape::Atom& atom = model_.dims.atom(id);
        const std::string name = json_string(model_.dims.symbol_name(atom.symbol));
        switch (atom.kind) {
        case shape::AtomKind::Symbol:
            return "{\"sym\":" + std::to_string(atom.symbol) + ",\"name\":" + name + "}";
        case shape::AtomKind::PackSize:
            return "{\"packsize\":" + std::to_string(atom.symbol) + ",\"name\":" + name + "}";
        case shape::AtomKind::FloorDiv:
            return binary("floordiv", atom);
        case shape::AtomKind::Mod:
            return binary("mod", atom);
        case shape::AtomKind::Min:
            return binary("min", atom);
        case shape::AtomKind::Max:
            return binary("max", atom);
        }
        return "null";
    }

    std::string binary(const char* op, const shape::Atom& atom) const {
        return std::string("{\"op\":\"") + op + "\",\"args\":[" + dim(atom.lhs) + "," +
               dim(atom.rhs) + "]}";
    }

    std::string shape_json(const Shape& shape) const {
        std::string out = "[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            out += i == 0 ? "" : ",";
            const ShapeElem& unit = shape[i];
            out += unit.is_pack
                       ? "{\"pack\":" + std::to_string(unit.pack) +
                             ",\"name\":" + json_string(model_.dims.symbol_name(unit.pack)) + "}"
                       : dim(unit.dim);
        }
        return out + "]";
    }

    // ----------------------------------------------------------------- types

    std::string dtype_json(DType dtype) const {
        if (dtype.is_var) {
            return "{\"var\":" + std::to_string(dtype.var) +
                   ",\"name\":" + json_string(model_.types.dtype_var(dtype.var).name) + "}";
        }
        return json_string(scalar_name(dtype.scalar));
    }

    std::string generic_value(const GenericValue& value) const {
        switch (value.kind) {
        case GenericValue::Kind::Dim:
            return "{\"dim\":" + dim(value.dim) + "}";
        case GenericValue::Kind::Pack:
            return "{\"shape\":" + shape_json(value.shape) + "}";
        case GenericValue::Kind::DType:
            return "{\"dtype\":" + dtype_json(value.dtype) + "}";
        }
        return "null";
    }

    std::string type_json(TypeId id) const {
        if (id == no_type) {
            return "null";
        }
        const TypeData& data = model_.types.get(id);
        switch (data.kind) {
        case TypeKind::Scalar:
            return "{\"kind\":\"scalar\",\"dtype\":" + dtype_json(data.dtype) + "}";
        case TypeKind::Tensor:
            return "{\"kind\":\"tensor\",\"shape\":" + shape_json(data.shape) +
                   ",\"dtype\":" + dtype_json(data.dtype) + "}";
        case TypeKind::Tuple: {
            std::string out = "{\"kind\":\"tuple\",\"elements\":[";
            for (std::size_t i = 0; i < data.elements.size(); ++i) {
                out += (i == 0 ? "" : ",") + type_json(data.elements[i]);
            }
            return out + "]}";
        }
        case TypeKind::Optional:
            return "{\"kind\":\"optional\",\"inner\":" + type_json(data.elements.front()) + "}";
        case TypeKind::Array:
            return "{\"kind\":\"array\",\"element\":" + type_json(data.elements.front()) +
                   ",\"length\":" + dim(data.value) + "}";
        case TypeKind::Struct:
        case TypeKind::Enum:
        case TypeKind::Block: {
            std::string out = std::string("{\"kind\":\"") +
                              (data.kind == TypeKind::Block  ? "block"
                               : data.kind == TypeKind::Enum ? "enum"
                                                             : "struct") +
                              "\",\"name\":" + json_string(model_.entities[data.decl].name) +
                              ",\"args\":[";
            for (std::size_t i = 0; i < data.args.size(); ++i) {
                out += (i == 0 ? "" : ",") + generic_value(data.args[i]);
            }
            return out + "]}";
        }
        case TypeKind::ShapeValue:
            return "{\"kind\":\"shape\",\"shape\":" + shape_json(data.shape) + "}";
        case TypeKind::Unit:
            return "{\"kind\":\"unit\"}";
        case TypeKind::CompileInt:
            return "{\"kind\":\"scalar\",\"dtype\":\"i64\"}";
        default:
            return "null";
        }
    }

    // ------------------------------------------------------------- generics

    std::string generics_json(const std::vector<GenericInfo>& generics) const {
        std::string out = "[";
        for (std::size_t i = 0; i < generics.size(); ++i) {
            const GenericInfo& generic = generics[i];
            out += i == 0 ? "" : ",";
            out += "{\"name\":" + json_string(generic.name) + ",\"kind\":\"" +
                   (generic.kind == GenericKind::Dim    ? "dim"
                    : generic.kind == GenericKind::Pack ? "shape"
                                                        : "dtype") +
                   "\"";
            if (generic.kind == GenericKind::DType) {
                out += ",\"var\":" + std::to_string(generic.dtype_var);
            } else {
                out += ",\"sym\":" + std::to_string(generic.symbol);
            }
            if (generic.default_value) {
                out += ",\"default\":" + generic_value(*generic.default_value);
            }
            out += "}";
        }
        return out + "]";
    }

    std::string constraints_json(const std::vector<ConstraintInfo>& constraints) const {
        std::string out = "[";
        for (std::size_t i = 0; i < constraints.size(); ++i) {
            const ConstraintInfo& constraint = constraints[i];
            const char* relation = constraint.relation == shape::Relation::Equal       ? "=="
                                   : constraint.relation == shape::Relation::NotEqual  ? "!="
                                   : constraint.relation == shape::Relation::Less      ? "<"
                                   : constraint.relation == shape::Relation::LessEqual ? "<="
                                   : constraint.relation == shape::Relation::Greater   ? ">"
                                                                                       : ">=";
            out += i == 0 ? "" : ",";
            out += std::string("{\"relation\":\"") + relation +
                   "\",\"lhs\":" + dim(constraint.lhs) + ",\"rhs\":" + dim(constraint.rhs) + "}";
        }
        return out + "]";
    }

    std::string substitution_json(const Substitution& substitution) const {
        std::string out = "{\"dims\":{";
        bool is_first = true;
        for (const auto& [symbol, value] : substitution.dims) {
            out += (is_first ? "" : ",") + json_string(std::to_string(symbol)) + ":" + dim(value);
            is_first = false;
        }
        out += "},\"packs\":{";
        is_first = true;
        for (const auto& [symbol, shape] : substitution.packs) {
            out += (is_first ? "" : ",") + json_string(std::to_string(symbol)) + ":" +
                   shape_json(shape);
            is_first = false;
        }
        out += "},\"dtypes\":{";
        is_first = true;
        for (const auto& [var, dtype] : substitution.dtypes) {
            out +=
                (is_first ? "" : ",") + json_string(std::to_string(var)) + ":" + dtype_json(dtype);
            is_first = false;
        }
        return out + "}}";
    }

    // ---------------------------------------------------------------- blocks

    std::string block_description(EntityId block) const {
        const Entity& entity = model_.entities[block];
        const DeclInfo& info = model_.decls.at(block);
        return "{\"name\":" + json_string(entity.name) +
               ",\"generics\":" + generics_json(info.generics) +
               ",\"constraints\":" + constraints_json(info.constraints) + "}";
    }

    // Every block declaration: its generics and members with their declared
    // types, so that a materializer can instantiate the structure.
    std::string blocks_json() const {
        std::string out = "{";
        bool is_first = true;
        for (EntityId id = 0; id < model_.entities.size(); ++id) {
            const Entity& entity = model_.entities[id];
            if (entity.kind != EntityKind::Block) {
                continue;
            }
            const DeclInfo& info = model_.decls.at(id);
            std::vector<std::pair<EntityId, std::string_view>> members;
            for (const auto& [name, member] : info.scope) {
                if (model_.entities[member].kind == EntityKind::Member) {
                    members.emplace_back(member, name);
                }
            }
            std::sort(members.begin(), members.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            out += is_first ? "" : ",";
            is_first = false;
            out += json_string(entity.name) + ":{\"generics\":" + generics_json(info.generics) +
                   ",\"constraints\":" + constraints_json(info.constraints) + ",\"members\":[";
            for (std::size_t i = 0; i < members.size(); ++i) {
                const Entity& member = model_.entities[members[i].first];
                const auto& decl = std::get<ast::MemberDecl>(module_ast_item(member));
                out += i == 0 ? "" : ",";
                out += "{\"name\":" + json_string(members[i].second) + ",\"kind\":\"" +
                       (decl.kind == ast::MemberKind::Param    ? "param"
                        : decl.kind == ast::MemberKind::Buffer ? "buffer"
                                                               : "sub") +
                       "\",\"type\":" + type_json(member.type) + "}";
            }
            out += "]}";
        }
        return out + "}";
    }

    // Manifest of the root block with symbolic shapes and per-level generic
    // bindings, mirroring sema's manifest but with structured dimensions.
    std::string manifest(EntityId root) const {
        std::string out = "[";
        bool is_first = true;
        std::vector<EntityId> active;
        std::vector<std::string> repeat;
        walk_block(root, {}, "", repeat, active, out, is_first);
        return out + "]";
    }

    void walk_block(EntityId block,
                    const Substitution& substitution,
                    const std::string& prefix,
                    std::vector<std::string>& repeat,
                    std::vector<EntityId>& active,
                    std::string& out,
                    bool& is_first) const {
        if (std::find(active.begin(), active.end(), block) != active.end()) {
            return;
        }
        active.push_back(block);
        const DeclInfo& info = model_.decls.at(block);
        // Members appear in the scope in hash order; use declaration order.
        std::vector<std::pair<EntityId, std::string_view>> members;
        for (const auto& [name, entity] : info.scope) {
            if (model_.entities[entity].kind == EntityKind::Member) {
                members.emplace_back(entity, name);
            }
        }
        std::sort(members.begin(), members.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (const auto& [entity, name] : members) {
            const Entity& member = model_.entities[entity];
            if (member.type == no_type) {
                continue;
            }
            const TypeId type = types_.substitute(member.type, substitution);
            const TypeData& data = model_.types.get(type);
            const std::string path = prefix + std::string(name);
            const bool is_sub = data.kind == TypeKind::Block || data.kind == TypeKind::Array;
            if (is_sub) {
                const TypeData* element = &data;
                std::string inner_prefix = path;
                bool is_array = false;
                if (data.kind == TypeKind::Array) {
                    is_array = true;
                    repeat.push_back(dim(data.value));
                    inner_prefix += "[*]";
                    element = &model_.types.get(data.elements.front());
                }
                if (element->kind == TypeKind::Block) {
                    Substitution inner;
                    const DeclInfo& inner_info = model_.decls.at(element->decl);
                    for (std::size_t i = 0;
                         i < inner_info.generics.size() && i < element->args.size();
                         ++i) {
                        const GenericInfo& generic = inner_info.generics[i];
                        switch (generic.kind) {
                        case GenericKind::Dim:
                            inner.dims[generic.symbol] = element->args[i].dim;
                            break;
                        case GenericKind::Pack:
                            inner.packs[generic.symbol] = element->args[i].shape;
                            break;
                        case GenericKind::DType:
                            inner.dtypes[generic.dtype_var] = element->args[i].dtype;
                            break;
                        }
                    }
                    walk_block(
                        element->decl, inner, inner_prefix + ".", repeat, active, out, is_first);
                }
                if (is_array) {
                    repeat.pop_back();
                }
                continue;
            }
            const bool is_optional = data.kind == TypeKind::Optional;
            const TypeData& tensor = is_optional ? model_.types.get(data.elements.front()) : data;
            if (tensor.kind != TypeKind::Tensor) {
                continue;
            }
            const auto& decl = std::get<ast::MemberDecl>(module_ast_item(member));
            out += is_first ? "" : ",";
            is_first = false;
            out += "{\"path\":" + json_string(path) + ",\"kind\":\"" +
                   (decl.kind == ast::MemberKind::Param ? "param" : "buffer") +
                   "\",\"dtype\":" + dtype_json(tensor.dtype) +
                   ",\"shape\":" + shape_json(tensor.shape) + ",\"repeat\":[";
            for (std::size_t i = 0; i < repeat.size(); ++i) {
                out += (i == 0 ? "" : ",") + repeat[i];
            }
            out += std::string("],\"optional\":") + (is_optional ? "true" : "false") + "}";
        }
        active.pop_back();
    }

    const ast::ItemData& module_ast_item(const Entity& entity) const {
        return asts_.at(entity.module)->item(entity.item).data;
    }

    // ------------------------------------------------------------- functions

    std::string function_json(const ir::Function& function) const {
        const Entity& entity = model_.entities[function.entity];
        std::string out = "{\"name\":" + json_string(function.name) + ",\"kind\":\"" +
                          (function.is_op      ? "op"
                           : function.is_entry ? "entry"
                                               : "fn") +
                          "\"";
        out += ",\"block\":" + (entity.parent == no_entity
                                    ? std::string("null")
                                    : json_string(model_.entities[entity.parent].name));
        out += ",\"generics\":" + generics_json(function.generics);
        out += ",\"constraints\":" + constraints_json(function.constraints);
        out += ",\"results\":[";
        for (std::size_t i = 0; i < function.results.size(); ++i) {
            out += (i == 0 ? "" : ",") + type_json(function.results[i]);
        }
        out += "],\"body\":" + region_json(function.body) + "}";
        return out;
    }

    std::string value_json(ir::ValueId id) const {
        const ir::Value& value = module_.value(id);
        return "{\"id\":" + std::to_string(id) + ",\"name\":" + json_string(value.name) +
               ",\"type\":" + type_json(value.type) + "}";
    }

    std::string region_json(ir::RegionId id) const {
        const ir::Block& block = module_.block(module_.region(id).blocks.front());
        std::string out = "{\"args\":[";
        for (std::size_t i = 0; i < block.arguments.size(); ++i) {
            out += (i == 0 ? "" : ",") + value_json(block.arguments[i]);
        }
        out += "],\"ops\":[";
        for (std::size_t i = 0; i < block.ops.size(); ++i) {
            out += (i == 0 ? "" : ",") + op_json(module_.op(block.ops[i]));
        }
        return out + "]}";
    }

    std::string op_json(const ir::Operation& op) const {
        const ir::Attributes& a = op.attributes;
        std::string out = "{\"kind\":" + json_string(ir::op_spelling(op.kind)) + ",\"operands\":[";
        for (std::size_t i = 0; i < op.operands.size(); ++i) {
            out += (i == 0 ? "" : ",") + std::to_string(op.operands[i]);
        }
        out += "],\"results\":[";
        for (std::size_t i = 0; i < op.results.size(); ++i) {
            out += (i == 0 ? "" : ",") + value_json(op.results[i]);
        }
        out += "],\"attrs\":{";
        std::vector<std::string> attrs;
        switch (op.kind) {
        case ir::OpKind::ConstInt:
        case ir::OpKind::ConstBool:
        case ir::OpKind::TupleGet:
        case ir::OpKind::StructGet:
            attrs.push_back("\"value\":" + std::to_string(a.integer));
            break;
        case ir::OpKind::ConstFloat: {
            char buffer[64];
            std::snprintf(buffer, sizeof buffer, "%.17g", a.number);
            attrs.push_back(std::string("\"value\":") + buffer);
            break;
        }
        case ir::OpKind::ConstDim:
            attrs.push_back("\"value\":" + dim(a.dim));
            break;
        case ir::OpKind::EnumConst:
        case ir::OpKind::BlockParam:
        case ir::OpKind::BlockSub:
            attrs.push_back("\"name\":" + json_string(a.name));
            break;
        case ir::OpKind::Compare:
            attrs.push_back("\"compare\":" + json_string(ir::compare_spelling(a.compare)));
            break;
        case ir::OpKind::Concat:
            attrs.push_back("\"axis\":" + std::to_string(a.axis));
            break;
        case ir::OpKind::Reshape:
        case ir::OpKind::Broadcast:
        case ir::OpKind::Fill:
        case ir::OpKind::Permute:
        case ir::OpKind::Iota:
            attrs.push_back("\"shape\":" + shape_json(a.shape));
            break;
        case ir::OpKind::Slice: {
            std::string axes = "\"axes\":[";
            for (std::size_t i = 0; i < a.starts.size(); ++i) {
                axes += i == 0 ? "" : ",";
                axes += a.whole[i]
                            ? "{\"whole\":" + shape_json(Shape{a.pack_units[i]}) + "}"
                            : "{\"start\":" + dim(a.starts[i]) + ",\"stop\":" + dim(a.stops[i]) +
                                  ",\"step\":" + std::to_string(a.steps[i]) +
                                  ",\"squeeze\":" + (a.squeezed[i] ? "true" : "false") + "}";
            }
            attrs.push_back(axes + "]");
            break;
        }
        case ir::OpKind::Comprehension:
        case ir::OpKind::Reduce: {
            std::string indices = "\"indices\":[";
            for (std::size_t i = 0; i < a.names.size(); ++i) {
                indices += i == 0 ? "" : ",";
                indices += "{\"name\":" + json_string(a.names[i]) + ",\"domain\":" +
                           shape_json(i < a.shape.size() ? Shape{a.shape[i]} : Shape{}) + "}";
            }
            attrs.push_back(indices + "]");
            if (op.kind == ir::OpKind::Reduce) {
                attrs.push_back("\"reduce\":" + json_string(ir::reduce_spelling(a.reduce)));
            }
            break;
        }
        case ir::OpKind::Call:
        case ir::OpKind::SemanticCall:
            attrs.push_back("\"callee\":" + json_string(a.name));
            attrs.push_back("\"substitution\":" + substitution_json(a.substitution));
            if (op.kind == ir::OpKind::SemanticCall && !a.names.empty()) {
                attrs.push_back("\"selected\":" + json_string(a.names.front()));
            }
            break;
        case ir::OpKind::EnumMatch: {
            std::string variants = "\"variants\":[";
            for (std::size_t i = 0; i < a.names.size(); ++i) {
                variants += (i == 0 ? "" : ",") + json_string(a.names[i]);
            }
            attrs.push_back(variants + "]");
            break;
        }
        default:
            break;
        }
        for (std::size_t i = 0; i < attrs.size(); ++i) {
            out += (i == 0 ? "" : ",") + attrs[i];
        }
        out += "},\"regions\":[";
        for (std::size_t i = 0; i < op.regions.size(); ++i) {
            out += (i == 0 ? "" : ",") + region_json(op.regions[i]);
        }
        return out + "]}";
    }

public:
    void set_asts(std::span<const ast::Ast* const> asts) { asts_ = {asts.begin(), asts.end()}; }

private:
    const ir::Module& module_;
    const Model& model_;
    TypeStore& types_; // substitution creates types
    const PlanOptions& options_;
    std::vector<const ast::Ast*> asts_;
};

} // namespace

std::expected<std::string, std::string> export_plan(ir::Module& module,
                                                    const PlanOptions& options) {
    PlanWriter writer(module, options);
    writer.set_asts(options.modules);
    return writer.run();
}

} // namespace linnet::backend
