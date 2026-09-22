#include "linnet/backend/plan_reader.hpp"

#include "linnet/lsp/json.hpp"

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace linnet::backend {

namespace {

using namespace sema;
using lsp::Json;

// A malformed document aborts the read with a message naming what was
// wrong; the exception never leaves `import_plan`.
struct ReadError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) {
    throw ReadError(message);
}

const Json& require(const Json& object, std::string_view key, const char* context) {
    const Json& value = object[key];
    if (value.is_null()) {
        fail(std::string(context) + " is missing `" + std::string(key) + "`");
    }
    return value;
}

std::optional<ir::OpKind> op_kind_of(std::string_view spelling) {
#define LINNET_IR_MATCH(name, text, operands, regions)                                             \
    if (spelling == (text)) {                                                                      \
        return ir::OpKind::name;                                                                   \
    }
    LINNET_IR_OPS(LINNET_IR_MATCH)
#undef LINNET_IR_MATCH
    return std::nullopt;
}

class PlanReader {
public:
    explicit PlanReader(const Json& document)
        : document_(document), model_(std::make_shared<Model>()), module_(model_) {
        Model* model = model_.get();
        model->types.set_entity_namer(
            [model](EntityId id) { return std::string(model->entities[id].name); });
    }

    ir::Module run() {
        if (document_["version"].as_int() != 1) {
            fail("unsupported plan version");
        }
        if (document_["module"].is_string()) {
            module_index(document_["module"].as_string());
        }
        declare();
        define_blocks();
        for (const Json& function : document_["functions"].as_array()) {
            define_function(function);
        }
        for (const Json& constant : document_["constants"].as_array()) {
            define_constant(constant);
        }
        return std::move(module_);
    }

private:
    Model& model() { return *model_; }
    TypeStore& types() { return model_->types; }
    shape::DimContext& dims() { return model_->dims; }

    // ------------------------------------------------------------- entities

    std::uint32_t module_index(const std::string& path) {
        for (std::uint32_t i = 0; i < model().module_paths.size(); ++i) {
            if (model().module_paths[i] == path) {
                return i;
            }
        }
        model().module_paths.push_back(path);
        return static_cast<std::uint32_t>(model().module_paths.size() - 1);
    }

    EntityId add_entity(EntityKind kind, const std::string& name, std::uint32_t module) {
        model().names.push_back(name);
        Entity entity;
        entity.kind = kind;
        entity.name = model().names.back();
        entity.module = module;
        entity.state = ResolveState::Done;
        model().entities.push_back(entity);
        return static_cast<EntityId>(model().entities.size() - 1);
    }

    // Splits `path::Block.method` or `path::name`.
    struct QualifiedName {
        std::string path;
        std::string block; // empty for free functions
        std::string name;
    };

    static QualifiedName split_name(const std::string& qualified) {
        const std::size_t separator = qualified.find("::");
        if (separator == std::string::npos) {
            fail("function name `" + qualified + "` is not qualified by a module path");
        }
        QualifiedName parts;
        parts.path = qualified.substr(0, separator);
        parts.name = qualified.substr(separator + 2);
        if (const std::size_t dot = parts.name.find('.'); dot != std::string::npos) {
            parts.block = parts.name.substr(0, dot);
            parts.name = parts.name.substr(dot + 1);
        }
        return parts;
    }

    std::string root_path() const {
        return model_->module_paths.empty() ? std::string("generated") : model_->module_paths[0];
    }

    EntityId declare_block(const std::string& name, const Json& block) {
        const std::string path =
            block["module"].is_string() ? block["module"].as_string() : root_path();
        const EntityId id = add_entity(EntityKind::Block, name, module_index(path));
        model().entities[id].is_pub = block["pub"].is_null() || block["pub"].as_bool();
        blocks_[name] = id;
        return id;
    }

    // Creates every block and function entity, and their generics, before
    // any body is read. Entities follow the order of the function list, each
    // block just before its first method, which recovers the declaration
    // order of a module whose blocks have methods.
    void declare() {
        const Json::Object& blocks = document_["blocks"].as_object();
        for (const Json& function : document_["functions"].as_array()) {
            const std::string qualified = require(function, "name", "function").as_string();
            const QualifiedName parts = split_name(qualified);
            if (!parts.block.empty() && !blocks_.contains(parts.block)) {
                const auto block = blocks.find(parts.block);
                if (block == blocks.end()) {
                    fail("method `" + qualified + "` belongs to an undeclared block");
                }
                declare_block(block->first, block->second);
            }
            const EntityId id =
                add_entity(EntityKind::Function, parts.name, module_index(parts.path));
            Entity& entity = model().entities[id];
            entity.is_pub = function["pub"].is_null() || function["pub"].as_bool();
            if (!parts.block.empty()) {
                entity.parent = blocks_.at(parts.block);
                model().decls[entity.parent].scope[entity.name] = id;
            }
            functions_[qualified] = id;
        }
        for (const auto& [name, block] : blocks) {
            if (!blocks_.contains(name)) {
                declare_block(name, block);
            }
        }
        // Generics first, so that symbols referenced from bodies and call
        // substitutions carry their declared names.
        for (const auto& [name, block] : blocks) {
            DeclInfo& info = model().decls[blocks_.at(name)];
            info.generics = generics(block["generics"]);
            info.constraints = constraints(block["constraints"]);
        }
        for (const Json& function : document_["functions"].as_array()) {
            DeclInfo& info = model().decls[functions_.at(function["name"].as_string())];
            info.generics = generics(function["generics"]);
            info.constraints = constraints(function["constraints"]);
        }
    }

    void define_blocks() {
        for (const auto& [name, block] : document_["blocks"].as_object()) {
            const EntityId id = blocks_.at(name);
            DeclInfo& info = model().decls[id];
            for (const Json& member : block["members"].as_array()) {
                const std::string member_name = require(member, "name", "member").as_string();
                const EntityId member_id =
                    add_entity(EntityKind::Member, member_name, model().entities[id].module);
                Entity& entity = model().entities[member_id];
                entity.parent = id;
                entity.is_buffer = member["kind"].as_string() == "buffer";
                entity.type = type(require(member, "type", "member"));
                info.scope[entity.name] = member_id;
            }
        }
    }

    // ------------------------------------------------------------- generics

    // Symbols and dtype variables are numbered by the document; each number
    // maps to one symbol of this model, created on first sight.
    shape::SymbolId symbol_by_id(std::int64_t id, const Json& name, shape::SymbolKind kind) {
        if (const auto found = symbols_.find(id); found != symbols_.end()) {
            return found->second;
        }
        return symbols_[id] = dims().add_symbol(
                   name.is_string() ? name.as_string() : "g" + std::to_string(id), kind);
    }

    shape::SymbolId symbol(const Json& reference, std::string_view key, shape::SymbolKind kind) {
        return symbol_by_id(reference[key].as_int(), reference["name"], kind);
    }

    DTypeVarId dtype_var_by_id(std::int64_t id, const Json& name, DTypeClass constraint) {
        if (const auto found = dtype_vars_.find(id); found != dtype_vars_.end()) {
            return found->second;
        }
        return dtype_vars_[id] = types().add_dtype_var(
                   name.is_string() ? name.as_string() : "T" + std::to_string(id), constraint);
    }

    DTypeVarId dtype_var(const Json& reference, DTypeClass constraint) {
        return dtype_var_by_id(reference["var"].as_int(), reference["name"], constraint);
    }

    static std::int64_t key_number(const std::string& key) {
        try {
            return std::stoll(key);
        } catch (const std::exception&) {
            fail("substitution key `" + key + "` is not a number");
        }
    }

    std::vector<GenericInfo> generics(const Json& list) {
        std::vector<GenericInfo> out;
        for (const Json& generic : list.as_array()) {
            GenericInfo info;
            model().names.push_back(require(generic, "name", "generic").as_string());
            info.name = model().names.back();
            const std::string& kind = generic["kind"].as_string();
            if (kind == "dim") {
                info.kind = GenericKind::Dim;
                info.symbol = symbol(generic, "sym", shape::SymbolKind::Dim);
            } else if (kind == "shape") {
                info.kind = GenericKind::Pack;
                info.symbol = symbol(generic, "sym", shape::SymbolKind::Pack);
            } else if (kind == "dtype") {
                info.kind = GenericKind::DType;
                const std::string& klass = generic["class"].as_string();
                info.constraint = klass == "float"     ? DTypeClass::Float
                                  : klass == "integer" ? DTypeClass::Integer
                                  : klass == "numeric" ? DTypeClass::Numeric
                                                       : DTypeClass::Any;
                info.dtype_var = dtype_var(generic, info.constraint);
            } else {
                fail("generic `" + std::string(info.name) + "` has unknown kind `" + kind + "`");
            }
            if (!generic["default"].is_null()) {
                info.default_value = generic_value(generic["default"]);
            }
            out.push_back(std::move(info));
        }
        return out;
    }

    std::vector<ConstraintInfo> constraints(const Json& list) {
        std::vector<ConstraintInfo> out;
        for (const Json& constraint : list.as_array()) {
            const std::string& relation = constraint["relation"].as_string();
            ConstraintInfo info;
            info.relation = relation == "=="   ? shape::Relation::Equal
                            : relation == "!=" ? shape::Relation::NotEqual
                            : relation == "<"  ? shape::Relation::Less
                            : relation == "<=" ? shape::Relation::LessEqual
                            : relation == ">"  ? shape::Relation::Greater
                            : relation == ">="
                                ? shape::Relation::GreaterEqual
                                : (fail("unknown constraint relation `" + relation + "`"),
                                   shape::Relation::Equal);
            info.lhs = dim(constraint["lhs"]);
            info.rhs = dim(constraint["rhs"]);
            out.push_back(std::move(info));
        }
        return out;
    }

    GenericValue generic_value(const Json& value) {
        GenericValue out;
        if (!value["dim"].is_null()) {
            out.kind = GenericValue::Kind::Dim;
            out.dim = dim(value["dim"]);
        } else if (!value["shape"].is_null()) {
            out.kind = GenericValue::Kind::Pack;
            out.shape = shape(value["shape"]);
        } else if (!value["dtype"].is_null()) {
            out.kind = GenericValue::Kind::DType;
            out.dtype = dtype(value["dtype"]);
        } else {
            fail("a generic value must be a dim, a shape, or a dtype");
        }
        return out;
    }

    // ----------------------------------------------------------- dimensions

    shape::Poly dim(const Json& value) {
        if (value.is_number()) {
            return shape::Poly(value.as_int());
        }
        if (value.is_null()) {
            return shape::Poly::invalid();
        }
        if (!value["sym"].is_null()) {
            return dims().symbol(symbol(value, "sym", shape::SymbolKind::Dim));
        }
        if (!value["packsize"].is_null()) {
            return dims().pack_size(symbol(value, "packsize", shape::SymbolKind::Pack));
        }
        const std::string& op = value["op"].as_string();
        const Json::Array& args = value["args"].as_array();
        if (op == "add" || op == "mul") {
            shape::Poly result(op == "add" ? 0 : 1);
            for (const Json& arg : args) {
                result = op == "add" ? result + dim(arg) : result * dim(arg);
            }
            return result;
        }
        if (args.size() != 2) {
            fail("dimension operation `" + op + "` needs two arguments");
        }
        const shape::Poly lhs = dim(args[0]);
        const shape::Poly rhs = dim(args[1]);
        if (op == "floordiv") {
            return dims().floor_div(lhs, rhs);
        }
        if (op == "mod") {
            return dims().mod(lhs, rhs);
        }
        if (op == "min") {
            return dims().min(lhs, rhs);
        }
        if (op == "max") {
            return dims().max(lhs, rhs);
        }
        fail("unknown dimension operation `" + op + "`");
    }

    Shape shape(const Json& list) {
        Shape out;
        for (const Json& unit : list.as_array()) {
            if (unit.is_object() && !unit["pack"].is_null()) {
                out.push_back(ShapeElem::of_pack(symbol(unit, "pack", shape::SymbolKind::Pack)));
            } else {
                out.push_back(ShapeElem::of(dim(unit)));
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- types

    DType dtype(const Json& value) {
        if (value.is_string()) {
            const auto kind = scalar_from_name(value.as_string());
            if (!kind) {
                fail("unknown dtype `" + value.as_string() + "`");
            }
            return DType::of(*kind);
        }
        if (!value["var"].is_null()) {
            return DType::variable(dtype_var(value, DTypeClass::Any));
        }
        fail("a dtype must be a name or a variable");
    }

    TypeId type(const Json& value) {
        if (value.is_null()) {
            return no_type;
        }
        const std::string& kind = value["kind"].as_string();
        if (kind == "scalar") {
            return types().scalar(dtype(require(value, "dtype", "scalar type")));
        }
        if (kind == "tensor") {
            return types().tensor(shape(require(value, "shape", "tensor type")),
                                  dtype(require(value, "dtype", "tensor type")));
        }
        if (kind == "tuple") {
            std::vector<TypeId> elements;
            for (const Json& element : value["elements"].as_array()) {
                elements.push_back(type(element));
            }
            return types().tuple(std::move(elements));
        }
        if (kind == "optional") {
            return types().optional(type(require(value, "inner", "optional type")));
        }
        if (kind == "array") {
            return types().array(type(require(value, "element", "array type")),
                                 dim(require(value, "length", "array type")));
        }
        if (kind == "block") {
            const std::string& name = require(value, "name", "block type").as_string();
            auto block = blocks_.find(name);
            if (block == blocks_.end()) {
                // A block of another module: known by name and module only.
                if (!value["module"].is_string()) {
                    fail("block type `" + name + "` is not declared in the plan");
                }
                const EntityId id =
                    add_entity(EntityKind::Block, name, module_index(value["module"].as_string()));
                model().entities[id].is_pub = true;
                block = blocks_.emplace(name, id).first;
            }
            std::vector<GenericValue> args;
            for (const Json& arg : value["args"].as_array()) {
                args.push_back(generic_value(arg));
            }
            return types().nominal(TypeKind::Block, block->second, std::move(args));
        }
        if (kind == "shape") {
            return types().shape_value(shape(require(value, "shape", "shape type")));
        }
        if (kind == "unit") {
            return types().unit();
        }
        fail("unsupported type kind `" + kind + "`");
    }

    // ------------------------------------------------------------ functions

    void define_function(const Json& function) {
        const std::string qualified = function["name"].as_string();
        ir::Function out;
        out.name = qualified;
        out.entity = functions_.at(qualified);
        const std::string& kind = function["kind"].as_string();
        out.is_op = kind == "op";
        out.is_entry = kind == "entry";
        DeclInfo& info = model().decls[out.entity];
        out.generics = info.generics;
        out.constraints = info.constraints;
        for (const Json& result : function["results"].as_array()) {
            out.results.push_back(type(result));
        }
        info.result = out.results.empty() ? types().unit() : out.results.front();

        out.body = module_.add_region(ir::no_id);
        const ir::RegionId body = out.body;
        module_.add_function(std::move(out));
        values_.clear();
        fill_region(body, require(function, "body", "function"));
    }

    void define_constant(const Json& description) {
        const std::string qualified = require(description, "name", "constant").as_string();
        const QualifiedName parts = split_name(qualified);
        const EntityId id = add_entity(EntityKind::Const, parts.name, module_index(parts.path));
        Entity& entity = model().entities[id];
        entity.is_pub = description["pub"].is_null() || description["pub"].as_bool();
        ir::Constant constant;
        constant.name = qualified;
        constant.entity = id;
        constant.type = type(require(description, "type", "constant"));
        // A contextual constant keeps its literal type so the emitter leaves
        // the annotation out.
        entity.type = constant.type;
        if (description["contextual"].as_bool()) {
            const TypeData& data = types().get(constant.type);
            entity.type = data.kind == TypeKind::Scalar && is_float(data.dtype.scalar)
                              ? types().float_literal(std::nullopt)
                              : types().compile_int(shape::Poly(0));
        }
        constant.body = module_.add_region(ir::no_id);
        const ir::RegionId body = constant.body;
        module_.add_constant(std::move(constant));
        values_.clear();
        fill_region(body, require(description, "body", "constant"));
    }

    void fill_region(ir::RegionId region, const Json& description) {
        const ir::BlockId block = module_.add_block(region);
        for (const Json& argument : description["args"].as_array()) {
            const ir::ValueId id =
                module_.add_argument(block, type(argument["type"]), argument["name"].as_string());
            values_[argument["id"].as_int()] = id;
        }
        for (const Json& op : description["ops"].as_array()) {
            add_op(block, op);
        }
    }

    void add_op(ir::BlockId block, const Json& description) {
        const std::string& spelling = description["kind"].as_string();
        const auto kind = op_kind_of(spelling);
        if (!kind) {
            fail("unknown operation `" + spelling + "`");
        }
        std::vector<ir::ValueId> operands;
        for (const Json& operand : description["operands"].as_array()) {
            const auto found = values_.find(operand.as_int());
            if (found == values_.end()) {
                fail("operation `" + spelling + "` uses a value that is not defined before it");
            }
            operands.push_back(found->second);
        }
        std::vector<TypeId> result_types;
        for (const Json& result : description["results"].as_array()) {
            result_types.push_back(type(result["type"]));
        }
        ir::Attributes attrs = attributes(*kind, description["attrs"]);
        if (*kind == ir::OpKind::Slice && !operands.empty()) {
            // Every sliced axis records its source unit, so that a range to
            // the end of the axis can be recognized later.
            const TypeData& source = types().get(module_.value(operands.front()).type);
            for (std::size_t i = 0; i < attrs.pack_units.size() && i < source.shape.size(); ++i) {
                if (!attrs.whole[i]) {
                    attrs.pack_units[i] = source.shape[i];
                }
            }
        }
        const ir::OpId op =
            module_.add_op(block, *kind, std::move(operands), result_types, std::move(attrs));
        const Json::Array& results = description["results"].as_array();
        for (std::size_t i = 0; i < results.size(); ++i) {
            const ir::ValueId id = module_.op(op).results[i];
            module_.value(id).name = results[i]["name"].as_string();
            values_[results[i]["id"].as_int()] = id;
        }
        for (const Json& region : description["regions"].as_array()) {
            const ir::RegionId id = module_.add_region(op);
            module_.op(op).regions.push_back(id);
            fill_region(id, region);
        }
    }

    ir::Attributes attributes(ir::OpKind kind, const Json& attrs) {
        ir::Attributes a;
        switch (kind) {
        case ir::OpKind::ConstInt:
        case ir::OpKind::ConstBool:
        case ir::OpKind::TupleGet:
        case ir::OpKind::StructGet:
            a.integer = attrs["value"].as_int();
            break;
        case ir::OpKind::ConstFloat:
            a.number = attrs["value"].as_number();
            break;
        case ir::OpKind::ConstDim:
            a.dim = dim(attrs["value"]);
            break;
        case ir::OpKind::EnumConst:
        case ir::OpKind::BlockParam:
        case ir::OpKind::BlockSub:
            a.name = attrs["name"].as_string();
            break;
        case ir::OpKind::Compare: {
            const std::string& spelling = attrs["compare"].as_string();
            bool is_known = false;
            for (const ir::CompareKind compare : {ir::CompareKind::Eq,
                                                  ir::CompareKind::Ne,
                                                  ir::CompareKind::Lt,
                                                  ir::CompareKind::Le,
                                                  ir::CompareKind::Gt,
                                                  ir::CompareKind::Ge}) {
                if (ir::compare_spelling(compare) == spelling) {
                    a.compare = compare;
                    is_known = true;
                }
            }
            if (!is_known) {
                fail("unknown comparison `" + spelling + "`");
            }
            break;
        }
        case ir::OpKind::Concat:
            a.axis = attrs["axis"].as_int();
            break;
        case ir::OpKind::Reshape:
        case ir::OpKind::Broadcast:
        case ir::OpKind::Fill:
        case ir::OpKind::Permute:
        case ir::OpKind::Iota:
            a.shape = shape(attrs["shape"]);
            break;
        case ir::OpKind::Slice:
            for (const Json& axis : attrs["axes"].as_array()) {
                if (!axis["whole"].is_null()) {
                    const Shape unit = shape(axis["whole"]);
                    if (unit.size() != 1) {
                        fail("a whole slice axis names exactly one shape unit");
                    }
                    a.starts.push_back(shape::Poly(0));
                    a.stops.push_back(unit.front().is_pack ? shape::Poly(0) : unit.front().dim);
                    a.steps.push_back(1);
                    a.squeezed.push_back(false);
                    a.whole.push_back(true);
                    a.pack_units.push_back(unit.front());
                } else {
                    a.starts.push_back(dim(axis["start"]));
                    a.stops.push_back(dim(axis["stop"]));
                    a.steps.push_back(axis["step"].is_null() ? 1 : axis["step"].as_int());
                    a.squeezed.push_back(axis["squeeze"].as_bool());
                    a.whole.push_back(false);
                    a.pack_units.push_back(ShapeElem::of(a.stops.back()));
                }
            }
            break;
        case ir::OpKind::Comprehension:
        case ir::OpKind::Reduce: {
            for (const Json& index : attrs["indices"].as_array()) {
                a.names.push_back(index["name"].as_string());
                const Shape domain = shape(index["domain"]);
                a.shape.push_back(domain.empty() ? ShapeElem{} : domain.front());
            }
            if (kind == ir::OpKind::Reduce) {
                const std::string& spelling = attrs["reduce"].as_string();
                bool is_known = false;
                for (const ir::ReduceKind reduce : {ir::ReduceKind::Sum,
                                                    ir::ReduceKind::Prod,
                                                    ir::ReduceKind::Max,
                                                    ir::ReduceKind::Min,
                                                    ir::ReduceKind::Any,
                                                    ir::ReduceKind::All}) {
                    if (ir::reduce_spelling(reduce) == spelling) {
                        a.reduce = reduce;
                        is_known = true;
                    }
                }
                if (!is_known) {
                    fail("unknown reduction `" + spelling + "`");
                }
            }
            break;
        }
        case ir::OpKind::Call:
        case ir::OpKind::SemanticCall: {
            a.name = attrs["callee"].as_string();
            const Json& substitution = attrs["substitution"];
            for (const auto& [key, value] : substitution["dims"].as_object()) {
                a.substitution.dims[symbol_by_id(key_number(key), {}, shape::SymbolKind::Dim)] =
                    dim(value);
            }
            for (const auto& [key, value] : substitution["packs"].as_object()) {
                a.substitution.packs[symbol_by_id(key_number(key), {}, shape::SymbolKind::Pack)] =
                    shape(value);
            }
            for (const auto& [key, value] : substitution["dtypes"].as_object()) {
                a.substitution.dtypes[dtype_var_by_id(key_number(key), {}, DTypeClass::Any)] =
                    dtype(value);
            }
            for (const Json& value : attrs["generics"].as_array()) {
                a.generic_args.push_back(generic_value(value));
            }
            if (attrs["selected"].is_string()) {
                a.names = {attrs["selected"].as_string()};
            }
            break;
        }
        case ir::OpKind::EnumMatch:
            for (const Json& variant : attrs["variants"].as_array()) {
                a.names.push_back(variant.as_string());
            }
            break;
        default:
            break;
        }
        return a;
    }

    const Json& document_;
    std::shared_ptr<Model> model_;
    ir::Module module_;
    std::map<std::string, EntityId> blocks_;
    std::map<std::string, EntityId> functions_;
    std::map<std::int64_t, shape::SymbolId> symbols_;
    std::map<std::int64_t, DTypeVarId> dtype_vars_;
    std::map<std::int64_t, ir::ValueId> values_;
};

} // namespace

std::expected<ir::Module, std::string> import_plan(std::string_view text) {
    const auto document = lsp::parse_json(text);
    if (!document) {
        return std::unexpected("plan is not valid JSON: " + document.error());
    }
    try {
        ir::Module module = PlanReader(*document).run();
        const std::vector<std::string> problems = ir::verify(module);
        if (!problems.empty()) {
            return std::unexpected("plan does not verify: " + problems.front());
        }
        return module;
    } catch (const ReadError& error) {
        return std::unexpected(error.what());
    }
}

} // namespace linnet::backend
