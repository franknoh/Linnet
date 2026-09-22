#include "linnet/emit/source.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace linnet::emit {

namespace {

using namespace sema;

class Emitter {
public:
    Emitter(const ir::Module& module, const EmitOptions& options)
        : module_(module), model_(module.model()), options_(options) {
        for (const ir::BlockId block : module_.all_blocks()) {
            for (const ir::OpId id : module_.block(block).ops) {
                if (module_.op(id).kind == ir::OpKind::TupleGet) {
                    continue; // destructuring consumes its tuple once, inline
                }
                for (const ir::ValueId operand : module_.op(id).operands) {
                    ++uses_[operand];
                    if (module_.value(operand).block != block) {
                        used_across_blocks_.insert(operand);
                    }
                }
            }
        }
        for (const ir::Function& function : module_.functions()) {
            functions_by_name_[function.name] = &function;
            functions_by_entity_[function.entity] = &function;
            collect_statement_blocks(function.body);
        }
    }

    std::expected<std::string, std::string> run() {
        std::string body;
        for (EntityId id = 0; id < model_.entities.size(); ++id) {
            const Entity& entity = model_.entities[id];
            if (entity.module != options_.module || entity.parent != no_entity) {
                continue;
            }
            std::expected<std::string, std::string> text;
            if (entity.kind == EntityKind::Function) {
                text = emit_function(id, 0);
            } else if (entity.kind == EntityKind::Block) {
                text = emit_block(id);
            } else {
                continue; // constants and types are folded into their uses
            }
            if (!text) {
                return text;
            }
            body += "\n" + *text;
        }
        std::string out = "module " + module_path() + "\n" + (imports_.empty() ? "" : "\n");
        for (const auto& [path, names] : imports_) {
            out += "use " + path + "::{";
            bool is_first = true;
            for (const std::string& name : names) {
                out += (is_first ? "" : ", ") + name;
                is_first = false;
            }
            out += "}\n";
        }
        return out + body;
    }

private:
    // ------------------------------------------------------------------ names

    std::string module_path() const {
        if (!options_.module_path.empty()) {
            return options_.module_path;
        }
        return options_.module < model_.module_paths.size() ? model_.module_paths[options_.module]
                                                            : "generated";
    }

    // Records a `use` for a declaration of another module.
    void import_entity(EntityId id) {
        const Entity& entity = model_.entities[id];
        if (entity.module != options_.module && entity.module < model_.module_paths.size()) {
            imports_[model_.module_paths[entity.module]].insert(std::string(entity.name));
        }
    }

    // Imports the nominal types a type mentions.
    void import_types(TypeId id) {
        if (id == no_type) {
            return;
        }
        const TypeData& data = model_.types.get(id);
        if (data.kind == TypeKind::Block || data.kind == TypeKind::Struct ||
            data.kind == TypeKind::Enum) {
            import_entity(data.decl);
        }
        for (const TypeId element : data.elements) {
            import_types(element);
        }
    }

    static std::string sanitize(const std::string& name) {
        std::string clean;
        for (const char c : name) {
            const bool is_word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                 (c >= '0' && c <= '9') || c == '_';
            clean += is_word ? c : '_';
        }
        if (clean.empty() || (clean.front() >= '0' && clean.front() <= '9')) {
            clean = "v" + clean;
        }
        return clean;
    }

    // A fresh local name for a value, or the one it already has.
    std::string name_of(ir::ValueId id) {
        if (const auto found = names_.find(id); found != names_.end()) {
            return found->second;
        }
        const std::string base =
            module_.value(id).name.empty() ? "v" : sanitize(module_.value(id).name);
        std::string candidate = base;
        for (int suffix = 1; used_names_.contains(candidate); ++suffix) {
            candidate = base + "_" + std::to_string(suffix);
        }
        used_names_.insert(candidate);
        names_[id] = candidate;
        return candidate;
    }

    void bind_name(ir::ValueId id, const std::string& name) {
        names_[id] = name;
        used_names_.insert(name);
    }

    // ------------------------------------------------------------------ types

    std::string dim(const shape::Poly& poly) const { return model_.dims.to_string(poly); }

    // A dimension as an expression operand: parenthesized unless atomic.
    std::string dim_operand(const shape::Poly& poly) const {
        const std::string text = dim(poly);
        const bool is_atomic = std::all_of(text.begin(), text.end(), [](char c) {
            return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                   (c >= '0' && c <= '9');
        });
        return is_atomic ? text : "(" + text + ")";
    }
    std::string type(TypeId id) {
        import_types(id);
        return model_.types.to_string(id);
    }
    std::string dtype_of(ir::ValueId id) const {
        return model_.types.to_string(model_.types.get(module_.value(id).type).dtype);
    }

    std::string generics_header(const std::vector<GenericInfo>& generics) const {
        std::string text;
        for (const GenericInfo& generic : generics) {
            text += text.empty() ? "<" : ", ";
            text += generic.kind == GenericKind::Pack ? "*" : "";
            text += generic.name;
            text += generic.kind == GenericKind::Dim            ? ": Dim"
                    : generic.kind == GenericKind::Pack         ? ": Shape"
                    : generic.constraint == DTypeClass::Float   ? ": Float"
                    : generic.constraint == DTypeClass::Integer ? ": Integer"
                    : generic.constraint == DTypeClass::Numeric ? ": Numeric"
                                                                : ": DType";
            if (generic.default_value) {
                text += " = " + model_.types.to_string(*generic.default_value);
            }
        }
        return text.empty() ? text : text + ">";
    }

    std::string where_clause(const std::vector<ConstraintInfo>& constraints) const {
        std::string text;
        for (const ConstraintInfo& constraint : constraints) {
            const char* relation = constraint.relation == shape::Relation::Equal       ? "=="
                                   : constraint.relation == shape::Relation::NotEqual  ? "!="
                                   : constraint.relation == shape::Relation::Less      ? "<"
                                   : constraint.relation == shape::Relation::LessEqual ? "<="
                                   : constraint.relation == shape::Relation::Greater   ? ">"
                                                                                       : ">=";
            text += text.empty() ? " where " : ", ";
            text += dim(constraint.lhs) + " " + relation + " " + dim(constraint.rhs);
        }
        return text;
    }

    // --------------------------------------------------------------- functions

    std::expected<std::string, std::string> emit_function(EntityId entity, int indent) {
        const auto found = functions_by_entity_.find(entity);
        if (found == functions_by_entity_.end()) {
            return std::unexpected("no IR for function `" +
                                   std::string(model_.entities[entity].name) + "`");
        }
        const ir::Function& function = *found->second;
        const Entity& target = model_.entities[entity];
        names_.clear();
        used_names_.clear();
        const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');

        // Defaults are not part of the IR: emitted signatures have none and
        // emitted calls pass every argument.
        std::string header = pad + (target.is_pub ? "pub " : "") +
                             (function.is_op      ? "op "
                              : function.is_entry ? "entry "
                                                  : "fn ") +
                             std::string(target.name) + generics_header(function.generics) + "(";
        const ir::Block& body = module_.block(module_.region(function.body).blocks.front());
        const std::size_t first_param = target.parent == no_entity ? 0 : 1;
        if (first_param == 1) {
            bind_name(body.arguments.front(), "self");
        }
        for (std::size_t i = first_param; i < body.arguments.size(); ++i) {
            const ir::ValueId argument = body.arguments[i];
            header += i == first_param ? "" : ", ";
            header += name_of(argument) + ": " + type(module_.value(argument).type);
        }
        header += ")";
        if (!function.results.empty()) {
            header += " -> " + type(function.results.front());
        }
        header += where_clause(function.constraints) + " {\n";

        auto statements = emit_statements(body, indent + 1);
        if (!statements) {
            return statements;
        }
        return header + *statements + pad + "}\n";
    }

    std::expected<std::string, std::string> emit_block(EntityId entity) {
        const Entity& target = model_.entities[entity];
        const DeclInfo& info = model_.decls.at(entity);
        std::string out = std::string(target.is_pub ? "pub " : "") + "block " +
                          std::string(target.name) + generics_header(info.generics) +
                          where_clause(info.constraints) + " {\n";
        std::vector<std::pair<EntityId, std::string>> members;
        for (const auto& [name, member] : info.scope) {
            const EntityKind kind = model_.entities[member].kind;
            if (kind == EntityKind::Member || kind == EntityKind::Function) {
                members.emplace_back(member, std::string(name));
            }
        }
        std::sort(members.begin(), members.end());
        for (const auto& [member, name] : members) {
            const Entity& declared = model_.entities[member];
            if (declared.kind != EntityKind::Member) {
                continue;
            }
            const TypeKind kind = model_.types.kind(declared.type);
            const bool is_sub = kind == TypeKind::Block || kind == TypeKind::Array;
            const bool is_optional = kind == TypeKind::Optional;
            out += std::string("    ") +
                   (is_sub               ? "sub "
                    : declared.is_buffer ? "buffer "
                                         : "param ") +
                   name + ": " + type(declared.type) + (is_optional ? " = none" : "") + "\n";
        }
        for (const auto& [member, name] : members) {
            if (model_.entities[member].kind != EntityKind::Function) {
                continue;
            }
            auto method = emit_function(member, 1);
            if (!method) {
                return method;
            }
            out += "\n" + *method;
        }
        return out + "}\n";
    }

    // -------------------------------------------------------------- statements

    // Blocks whose operations become statements: function bodies and
    // `static for` bodies. Every other region is a single expression.
    void collect_statement_blocks(ir::RegionId region) {
        for (const ir::BlockId block : module_.region(region).blocks) {
            statement_blocks_.insert(block);
            for (const ir::OpId id : module_.block(block).ops) {
                if (module_.op(id).kind == ir::OpKind::StaticFor) {
                    collect_statement_blocks(module_.op(id).regions.front());
                }
            }
        }
    }

    static bool is_constant(ir::OpKind kind) {
        return kind == ir::OpKind::ConstInt || kind == ir::OpKind::ConstFloat ||
               kind == ir::OpKind::ConstBool || kind == ir::OpKind::ConstDim ||
               kind == ir::OpKind::EnumConst || kind == ir::OpKind::OptionNone;
    }

    // Whether a value gets a `let` of its own: it is used more than once or
    // from inside a region (so no region body repeats an outer computation),
    // or it is a comprehension, which only exists as a statement. Constants
    // are always inlined so literals keep adopting their type from context.
    bool needs_binding(const ir::Operation& op) const {
        if (op.kind == ir::OpKind::Comprehension) {
            return true;
        }
        if (is_constant(op.kind) || op.results.size() != 1 ||
            !statement_blocks_.contains(op.block)) {
            return false;
        }
        const auto found = uses_.find(op.results.front());
        return (found != uses_.end() && found->second > 1) ||
               used_across_blocks_.contains(op.results.front());
    }

    // `let (a, b) = t` for the tuple.get operations of `tuple` in `block`,
    // binding every result name at once.
    std::string destructure(const ir::Block& block, ir::ValueId tuple) {
        const TypeData& data = model_.types.get(module_.value(tuple).type);
        std::vector<std::string> names(data.elements.size(), "_");
        for (const ir::OpId id : block.ops) {
            const ir::Operation& op = module_.op(id);
            if (op.kind == ir::OpKind::TupleGet && op.operands.front() == tuple) {
                const auto index = static_cast<std::size_t>(op.attributes.integer);
                if (index < names.size() && names[index] == "_") {
                    names[index] = name_of(op.results.front());
                }
            }
        }
        std::string text = "let (";
        for (std::size_t i = 0; i < names.size(); ++i) {
            text += (i == 0 ? "" : ", ") + names[i];
        }
        return text + (names.size() == 1 ? ",)" : ")") + " = " + expr(tuple);
    }

    std::expected<std::string, std::string> emit_statements(const ir::Block& block, int indent) {
        const std::string pad(static_cast<std::size_t>(indent) * 4, ' ');
        std::string out;
        for (const ir::OpId id : block.ops) {
            const ir::Operation& op = module_.op(id);
            switch (op.kind) {
            case ir::OpKind::Return:
                out += pad + "return" +
                       (op.operands.empty() ? "" : " " + expr(op.operands.front())) + "\n";
                break;
            case ir::OpKind::Yield:
                break; // handled by the enclosing loop
            case ir::OpKind::TupleGet:
                if (!names_.contains(op.results.front())) {
                    out += pad + destructure(block, op.operands.front()) + "\n";
                }
                break;
            case ir::OpKind::Comprehension: {
                const std::string indices = index_list(op);
                auto body = region_value(op.regions.front());
                if (!body) {
                    return body;
                }
                out += pad;
                out += "let " + name_of(op.results.front()) + "[" + indices + "] = " + *body + "\n";
                break;
            }
            case ir::OpKind::StaticFor: {
                const ir::Block& body =
                    module_.block(module_.region(op.regions.front()).blocks.front());
                for (std::size_t i = 1; i < op.operands.size(); ++i) {
                    const std::string name = name_of(body.arguments[i]);
                    out += pad;
                    out += "var " + name + " = " + expr(op.operands[i]) + "\n";
                    bind_name(op.results[i - 1], name);
                }
                out += pad + "static for " + name_of(body.arguments.front()) + " in " +
                       expr(op.operands.front()) + " {\n";
                auto inner = emit_statements(body, indent + 1);
                if (!inner) {
                    return inner;
                }
                out += *inner;
                const ir::Operation& yield = module_.op(body.ops.back());
                for (std::size_t i = 0; i < yield.operands.size(); ++i) {
                    out += pad + "    " + name_of(op.results[i]) + " = " + expr(yield.operands[i]) +
                           "\n";
                }
                out += pad + "}\n";
                break;
            }
            default:
                if (needs_binding(op)) {
                    out += pad + "let " + name_of(op.results.front()) + " = " + define(op) + "\n";
                }
                break;
            }
        }
        return out;
    }

    // ------------------------------------------------------------- expressions

    bool index_is_pack(const ir::Operation& op, std::size_t i) const {
        const ir::Block& body = module_.block(module_.region(op.regions.front()).blocks.front());
        return i < body.arguments.size() &&
               model_.types.kind(module_.value(body.arguments[i]).type) == TypeKind::ShapeValue;
    }

    std::string index_list(const ir::Operation& op) {
        const ir::Block& body = module_.block(module_.region(op.regions.front()).blocks.front());
        std::string text;
        for (std::size_t i = 0; i < op.attributes.names.size(); ++i) {
            text += i == 0 ? "" : ", ";
            text += index_is_pack(op, i) ? "*" : "";
            text += op.attributes.names[i];
            if (i < body.arguments.size()) {
                bind_name(body.arguments[i], op.attributes.names[i]);
            }
        }
        return text;
    }

    std::expected<std::string, std::string> region_value(ir::RegionId region) {
        const ir::Block& block = module_.block(module_.region(region).blocks.front());
        const ir::Operation& terminator = module_.op(block.ops.back());
        if (terminator.kind != ir::OpKind::Yield || terminator.operands.size() != 1) {
            return std::unexpected("a region does not yield exactly one value");
        }
        return expr(terminator.operands.front());
    }

    std::string region_text(ir::RegionId region) {
        auto text = region_value(region);
        return text ? *text : "<error>";
    }

    std::string operand_list(const ir::Operation& op, std::size_t from = 0) {
        std::string text;
        for (std::size_t i = from; i < op.operands.size(); ++i) {
            text += i == from ? "" : ", ";
            text += expr(op.operands[i]);
        }
        return text;
    }

    std::string shape_list(const Shape& shape) const {
        return "[" + model_.types.to_string(shape) + "]";
    }

    // The expression for a value: its name when it is bound, else its
    // definition inlined.
    std::string expr(ir::ValueId id) {
        if (const auto found = names_.find(id); found != names_.end()) {
            return found->second;
        }
        const ir::OpId producer = module_.value(id).producer;
        if (producer == ir::no_id) {
            return name_of(id);
        }
        const ir::Operation& op = module_.op(producer);
        if (needs_binding(op) || op.results.size() != 1) {
            return name_of(id);
        }
        return define(op);
    }

    // The expression that computes an operation's single result.
    std::string define(const ir::Operation& op) {
        const ir::Attributes& a = op.attributes;
        const ir::ValueId result = op.results.front();
        const auto binary = [&](const char* symbol) {
            return "(" + expr(op.operands[0]) + " " + symbol + " " + expr(op.operands[1]) + ")";
        };
        const auto call = [&](const std::string& name) {
            return name + "(" + operand_list(op) + ")";
        };
        switch (op.kind) {
        case ir::OpKind::ConstInt:
            return std::to_string(a.integer);
        case ir::OpKind::ConstFloat: {
            std::ostringstream stream;
            stream.precision(17);
            stream << a.number;
            std::string text = stream.str();
            if (text.find_first_of(".eE") == std::string::npos) {
                text += ".0";
            }
            return text;
        }
        case ir::OpKind::ConstBool:
            return a.integer != 0 ? "true" : "false";
        case ir::OpKind::ConstDim:
            return dim_operand(a.dim);
        case ir::OpKind::EnumConst:
            return type(module_.value(result).type) + "." + a.name;
        case ir::OpKind::Add:
            return binary("+");
        case ir::OpKind::Sub:
            return binary("-");
        case ir::OpKind::Mul:
            return binary("*");
        case ir::OpKind::Div:
            return binary("/");
        case ir::OpKind::Rem:
            return binary("%");
        case ir::OpKind::And:
            return binary("&&");
        case ir::OpKind::Or:
            return binary("||");
        case ir::OpKind::Compare:
            return binary(a.compare == ir::CompareKind::Eq   ? "=="
                          : a.compare == ir::CompareKind::Ne ? "!="
                          : a.compare == ir::CompareKind::Lt ? "<"
                          : a.compare == ir::CompareKind::Le ? "<="
                          : a.compare == ir::CompareKind::Gt ? ">"
                                                             : ">=");
        case ir::OpKind::Min:
            return call("min");
        case ir::OpKind::Max:
            return call("max");
        case ir::OpKind::Not:
            return "!" + expr(op.operands[0]);
        case ir::OpKind::Neg:
            return "-" + expr(op.operands[0]);
        case ir::OpKind::Exp:
        case ir::OpKind::Log:
        case ir::OpKind::Sqrt:
        case ir::OpKind::Rsqrt:
        case ir::OpKind::Sin:
        case ir::OpKind::Cos:
        case ir::OpKind::Tanh:
        case ir::OpKind::Abs:
            return call(std::string(ir::op_spelling(op.kind)));
        case ir::OpKind::Cast:
            return "cast<" + dtype_of(result) + ">(" + expr(op.operands[0]) + ")";
        case ir::OpKind::Select:
            return call("select");
        case ir::OpKind::Reshape:
            return "reshape(" + expr(op.operands[0]) + ", " + shape_list(a.shape) + ")";
        case ir::OpKind::Broadcast:
            return "broadcast_to(" + expr(op.operands[0]) + ", " + shape_list(a.shape) + ")";
        case ir::OpKind::Permute:
            return "permute(" + expr(op.operands[0]) + ", " + shape_list(a.shape) + ")";
        case ir::OpKind::Concat:
            return "concat(" + operand_list(op) + ", axis = " + std::to_string(a.axis) + ")";
        case ir::OpKind::Fill:
            return "fill<" + dtype_of(result) + ">(" + shape_list(a.shape) + ", " +
                   expr(op.operands[0]) + ")";
        case ir::OpKind::Iota:
            return "iota<" + dtype_of(result) + ">(" + dim(a.shape.front().dim) + ")";
        case ir::OpKind::Slice: {
            std::string text = expr(op.operands[0]) + "[";
            for (std::size_t i = 0; i < a.starts.size(); ++i) {
                text += i == 0 ? "" : ", ";
                if (a.whole[i]) {
                    text += a.pack_units[i].is_pack ? "..." : ":";
                } else if (a.squeezed[i]) {
                    text += dim(a.starts[i]);
                } else {
                    text += dim(a.starts[i]) + ":" + dim(a.stops[i]) +
                            (a.steps[i] == 1 ? "" : ":" + std::to_string(a.steps[i]));
                }
            }
            return text + "]";
        }
        case ir::OpKind::Element: {
            std::string text = expr(op.operands[0]) + "[";
            for (std::size_t i = 1; i < op.operands.size(); ++i) {
                const ir::ValueId index = op.operands[i];
                const bool is_pack =
                    model_.types.kind(module_.value(index).type) == TypeKind::ShapeValue;
                text += (i == 1 ? "" : ", ") + std::string(is_pack ? "*" : "") + expr(index);
            }
            return text + "]";
        }
        case ir::OpKind::Reduce: {
            const ir::Block& body =
                module_.block(module_.region(op.regions.front()).blocks.front());
            const ir::Operation& yield = module_.op(body.ops.back());
            const bool converts =
                !yield.operands.empty() && dtype_of(yield.operands.front()) != dtype_of(result);
            std::string text = std::string(ir::reduce_spelling(a.reduce)) +
                               (converts ? "<" + dtype_of(result) + ">" : "");
            text += "[" + index_list(op) + "] ";
            return text + region_text(op.regions.front());
        }
        case ir::OpKind::StructGet: {
            const TypeData& data = model_.types.get(module_.value(op.operands[0]).type);
            const auto& fields = model_.decls.at(data.decl).fields;
            const auto index = static_cast<std::size_t>(a.integer);
            return expr(op.operands[0]) + "." +
                   (index < fields.size() ? std::string(fields[index].name) : "<field>");
        }
        case ir::OpKind::TupleMake:
            return "(" + operand_list(op) + (op.operands.size() == 1 ? ",)" : ")");
        case ir::OpKind::OptionSome:
            return "some(" + expr(op.operands[0]) + ")";
        case ir::OpKind::OptionNone:
            return "none";
        case ir::OpKind::OptionMatch: {
            const ir::Block& some_block =
                module_.block(module_.region(op.regions[0]).blocks.front());
            std::string text = "match " + expr(op.operands[0]) + " { some(" +
                               name_of(some_block.arguments.front()) + ") => ";
            text += region_text(op.regions[0]);
            return text + " none => " + region_text(op.regions[1]) + " }";
        }
        case ir::OpKind::EnumMatch: {
            std::string text = "match " + expr(op.operands[0]) + " {";
            for (std::size_t i = 0; i < op.regions.size(); ++i) {
                text += " " + a.names[i] + " => ";
                text += region_text(op.regions[i]);
            }
            return text + " }";
        }
        case ir::OpKind::If: {
            std::string text = "if " + expr(op.operands[0]) + " { ";
            text += region_text(op.regions[0]);
            text += " } else { ";
            text += region_text(op.regions[1]);
            return text + " }";
        }
        case ir::OpKind::Call:
        case ir::OpKind::SemanticCall: {
            // A callee the module defines is named through its entity; one
            // it only references (an external plan) through its qualified
            // name `path::name` or `path::Block.method`.
            const auto callee = functions_by_name_.find(a.name);
            const bool is_defined = callee != functions_by_name_.end();
            std::string name;
            bool is_method = false;
            if (is_defined) {
                const Entity& target = model_.entities[callee->second->entity];
                name = std::string(target.name);
                is_method = target.parent != no_entity;
                if (!is_method) {
                    import_entity(callee->second->entity);
                }
            } else {
                const std::size_t separator = a.name.find("::");
                const std::string path = a.name.substr(0, separator);
                name = separator == std::string::npos ? a.name : a.name.substr(separator + 2);
                if (const std::size_t dot = name.find('.'); dot != std::string::npos) {
                    name = name.substr(dot + 1);
                    is_method = true;
                } else if (path != module_path()) {
                    imports_[path].insert(name);
                }
            }
            if (is_method) {
                const std::string receiver = expr(op.operands[0]);
                const std::string prefix = receiver == "self" ? "" : receiver + ".";
                return prefix + name + "(" + operand_list(op, 1) + ")";
            }
            return name + generic_arguments(a) + "(" + operand_list(op) + ")";
        }
        case ir::OpKind::BlockParam:
        case ir::OpKind::BlockSub: {
            const std::string base = expr(op.operands[0]);
            return base == "self" ? a.name : base + "." + a.name;
        }
        case ir::OpKind::ArrayGet:
            return expr(op.operands[0]) + "[" + expr(op.operands[1]) + "]";
        default:
            return "<unsupported " + std::string(ir::op_spelling(op.kind)) + ">";
        }
    }

    // Explicit generic arguments make emitted calls independent of inference.
    std::string generic_arguments(const ir::Attributes& a) const {
        std::string text;
        for (const GenericValue& generic : a.generic_args) {
            std::string value;
            switch (generic.kind) {
            case GenericValue::Kind::Dim:
                value = dim(generic.dim);
                break;
            case GenericValue::Kind::Pack: {
                const Shape& shape = generic.shape;
                const bool has_pack = std::any_of(
                    shape.begin(), shape.end(), [](const ShapeElem& e) { return e.is_pack; });
                if (shape.size() == 1 && shape.front().is_pack) {
                    value = std::string(model_.dims.symbol_name(shape.front().pack));
                } else if (!has_pack) {
                    value = "[" + model_.types.to_string(shape) + "]";
                } // a shape mixing packs and dimensions has no literal
                break;
            }
            case GenericValue::Kind::DType:
                value = model_.types.to_string(generic.dtype);
                break;
            }
            if (value.empty()) {
                return ""; // let inference decide when a binding has no spelling
            }
            text += (text.empty() ? "<" : ", ") + value;
        }
        return text.empty() ? text : text + ">";
    }

    const ir::Module& module_;
    const Model& model_;
    const EmitOptions& options_;
    std::map<std::string, const ir::Function*> functions_by_name_;
    std::map<EntityId, const ir::Function*> functions_by_entity_;
    std::map<std::string, std::set<std::string>> imports_;
    std::map<ir::ValueId, std::string> names_;
    std::set<std::string> used_names_;
    std::map<ir::ValueId, std::size_t> uses_;
    std::set<ir::BlockId> statement_blocks_;
    std::set<ir::ValueId> used_across_blocks_;
};

} // namespace

std::expected<std::string, std::string> emit_source(const ir::Module& module,
                                                    const EmitOptions& options) {
    return Emitter(module, options).run();
}

} // namespace linnet::emit
