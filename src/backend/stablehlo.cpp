#include "linnet/backend/stablehlo.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace linnet::backend {

namespace {

using namespace sema;

// A capability failure; caught at the boundary and returned as text.
struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) {
    throw Unsupported(message);
}

using Dims = std::vector<std::int64_t>;

// A value during emission. Tensors (scalars are rank 0) live in the MLIR
// text; the other kinds are resolved statically while inlining.
struct Val {
    enum class Kind : std::uint8_t { Tensor, None, Some, Tuple, Block, Array, Index, Pack, Enum };
    Kind kind = Kind::Tensor;
    std::string name; // Tensor: `%n`
    Dims shape;       // Tensor
    ScalarKind dtype = ScalarKind::F32;
    std::size_t grid_rank = 0;     // Tensor: leading axes that are grid axes
    std::vector<Val> elements;     // Tuple, Array; Some holds one
    std::string path;              // Block: member path prefix, `layers.0.`; Enum: variant
    EntityId block = no_entity;    // Block: declaration
    Substitution subst;            // Block: its generic bindings
    std::vector<std::size_t> axes; // Index (one), Pack (several): grid axes
};

const char* mlir_dtype(ScalarKind kind) {
    switch (kind) {
    case ScalarKind::Bool:
        return "i1";
    case ScalarKind::I8:
        return "i8";
    case ScalarKind::I16:
        return "i16";
    case ScalarKind::I32:
        return "i32";
    case ScalarKind::I64:
        return "i64";
    case ScalarKind::U8:
        return "ui8";
    case ScalarKind::U16:
        return "ui16";
    case ScalarKind::U32:
        return "ui32";
    case ScalarKind::U64:
        return "ui64";
    case ScalarKind::F16:
        return "f16";
    case ScalarKind::BF16:
        return "bf16";
    case ScalarKind::F32:
        return "f32";
    case ScalarKind::F64:
        return "f64";
    }
    return "f32";
}

std::string tensor_type(const Dims& shape, ScalarKind dtype) {
    std::string text = "tensor<";
    for (const std::int64_t dim : shape) {
        text += std::to_string(dim) + "x";
    }
    return text + mlir_dtype(dtype) + ">";
}

std::string i64_array(const std::vector<std::int64_t>& values) {
    std::string text = "array<i64";
    for (std::size_t i = 0; i < values.size(); ++i) {
        text += (i == 0 ? ": " : ", ") + std::to_string(values[i]);
    }
    return text + ">";
}

std::string index_list(const std::vector<std::int64_t>& values) {
    std::string text = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        text += (i == 0 ? "" : ", ") + std::to_string(values[i]);
    }
    return text + "]";
}

std::string float_text(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    std::string text = buffer;
    // MLIR wants a decimal point in the mantissa: `1.0e30`, not `1e30`.
    const std::size_t exponent = text.find_first_of("eE");
    if (text.find('.') == std::string::npos) {
        text.insert(exponent == std::string::npos ? text.size() : exponent, ".0");
    }
    return text;
}

class Exporter {
public:
    Exporter(ir::Module& module, const StableHloOptions& options)
        : module_(module), model_(module.model()), types_(module.types()), options_(options) {
        for (const ir::Function& function : module_.functions()) {
            functions_[function.name] = &function;
        }
    }

    std::string run() {
        const EntityId root = find_root();
        const ir::Function& entry = find_entry(root);
        const Substitution root_subst = root_bindings(root);

        // The root block instance and its parameters as function arguments.
        Val self;
        self.kind = Val::Kind::Block;
        self.block = root;
        self.subst = root_subst;
        collect_parameters(root, root_subst, "");

        Frame frame;
        frame.subst = entry_bindings(entry, root_subst);
        const ir::Block& body = module_.block(module_.region(entry.body).blocks.front());
        std::vector<std::string> arguments;
        frame.values[body.arguments.front()] = self;
        for (std::size_t i = 1; i < body.arguments.size(); ++i) {
            const ir::ValueId argument = body.arguments[i];
            Val value = tensor_value(module_.value(argument).type, frame.subst);
            value.name = "%" + module_.value(argument).name;
            arguments.push_back(value.name + ": " + tensor_type(value.shape, value.dtype));
            frame.values[argument] = value;
        }
        for (const Parameter& parameter : parameters_) {
            arguments.push_back(parameter.value.name + ": " +
                                tensor_type(parameter.value.shape, parameter.value.dtype) +
                                " {linnet.path = \"" + parameter.path + "\"}");
        }
        frames_.push_back(std::move(frame));
        const std::vector<Val> results = run_block(body);
        frames_.pop_back();
        if (results.size() != 1 || results.front().kind != Val::Kind::Tensor) {
            fail("the entry must return one tensor");
        }
        const Val& result = results.front();
        body_ += "    \"func.return\"(" + result.name + ") : (" +
                 tensor_type(result.shape, result.dtype) + ") -> ()\n";

        std::string out = "// " + std::string(model_.entities[root].name) + "." +
                          std::string(model_.entities[entry.entity].name) + " from module " +
                          model_.module_paths.at(options_.root_module) +
                          ". Arguments after the inputs are the\n"
                          "// parameters of the block hierarchy, named by `linnet.path`.\n";
        out += "module @" + sanitized(model_.module_paths.at(options_.root_module)) + " {\n";
        out += "  func.func @main(";
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            out += (i == 0 ? "" : ", ") + arguments[i];
        }
        out += ") -> " + tensor_type(result.shape, result.dtype) + " {\n";
        out += body_;
        out += "  }\n}\n";
        return out;
    }

private:
    static std::string sanitized(const std::string& path) {
        std::string out;
        for (const char c : path) {
            out += c == '.' ? '_' : c;
        }
        return out;
    }

    // Frames live in a deque so that references into an outer frame's
    // values survive the frames a call pushes; a vector would move (or, on
    // some standard libraries, copy) them on growth.
    struct Frame {
        std::map<ir::ValueId, Val> values;
        Substitution subst;
    };

    struct Parameter {
        std::string path;
        Val value;
    };

    // ------------------------------------------------------------- roots

    EntityId find_root() const {
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
            for (const ir::Function& function : module_.functions()) {
                if (function.is_entry && model_.entities[function.entity].parent == id) {
                    candidates.push_back(id);
                    break;
                }
            }
        }
        if (!options_.root.empty()) {
            fail("no block named `" + options_.root + "` in this file");
        }
        if (candidates.size() != 1) {
            fail(candidates.empty() ? "no block with an `entry`; name one with --root"
                                    : "several blocks have entries; name one with --root");
        }
        return candidates.front();
    }

    const ir::Function& find_entry(EntityId root) const {
        const ir::Function* found = nullptr;
        for (const ir::Function& function : module_.functions()) {
            const Entity& entity = model_.entities[function.entity];
            if (!function.is_entry || entity.parent != root) {
                continue;
            }
            if (!options_.entry.empty() ? entity.name == options_.entry : found == nullptr) {
                found = &function;
            } else if (options_.entry.empty()) {
                fail("the block has several entries; name one with --entry");
            }
        }
        if (found == nullptr) {
            fail(options_.entry.empty() ? "the block has no entry"
                                        : "no entry named `" + options_.entry + "`");
        }
        return *found;
    }

    // --------------------------------------------------------- bindings

    // Binds each generic from `--bind name=value`, or its default.
    void bind_generics(const std::vector<GenericInfo>& generics,
                       Substitution& subst,
                       const char* owner) {
        for (const GenericInfo& generic : generics) {
            const auto bound = options_.bindings.find(std::string(generic.name));
            const std::string* text = bound == options_.bindings.end() ? nullptr : &bound->second;
            const GenericValue fallback = generic.default_value.value_or(GenericValue{});
            if (text == nullptr && !generic.default_value) {
                fail(std::string(owner) + " generic `" + std::string(generic.name) +
                     "` needs a value: pass --bind " + std::string(generic.name) + "=...");
            }
            switch (generic.kind) {
            case GenericKind::Dim: {
                std::int64_t value = 0;
                if (text != nullptr) {
                    const auto* end = text->data() + text->size();
                    if (std::from_chars(text->data(), end, value).ptr != end) {
                        fail("`" + *text + "` is not an integer for `" + std::string(generic.name) +
                             "`");
                    }
                } else {
                    value = fallback.dim.constant().value_or(0);
                }
                subst.dims[generic.symbol] = shape::Poly(value);
                break;
            }
            case GenericKind::Pack:
                fail("shape-pack generics of the root block or entry are not supported");
            case GenericKind::DType: {
                DType dtype;
                if (text != nullptr) {
                    const auto kind = scalar_from_name(*text);
                    if (!kind) {
                        fail("`" + *text + "` is not a dtype for `" + std::string(generic.name) +
                             "`");
                    }
                    dtype = DType::of(*kind);
                } else {
                    dtype = fallback.dtype;
                }
                subst.dtypes[generic.dtype_var] = dtype;
                break;
            }
            }
        }
    }

    Substitution root_bindings(EntityId root) {
        Substitution subst;
        bind_generics(model_.decls.at(root).generics, subst, "root block");
        return subst;
    }

    Substitution entry_bindings(const ir::Function& entry, const Substitution& root_subst) {
        Substitution subst = root_subst;
        bind_generics(entry.generics, subst, "entry");
        return subst;
    }

    // ------------------------------------------------------- parameters

    void collect_parameters(EntityId block, const Substitution& subst, const std::string& prefix) {
        const DeclInfo& info = model_.decls.at(block);
        std::vector<std::pair<EntityId, std::string_view>> members;
        for (const auto& [name, entity] : info.scope) {
            if (model_.entities[entity].kind == EntityKind::Member) {
                members.emplace_back(entity, name);
            }
        }
        std::sort(members.begin(), members.end());
        for (const auto& [entity, name] : members) {
            const TypeId type = types_.substitute(model_.entities[entity].type, subst);
            const TypeData& data = types_.get(type);
            const std::string path = prefix + std::string(name);
            if (data.kind == TypeKind::Block) {
                collect_parameters(data.decl, block_substitution(data), path + ".");
            } else if (data.kind == TypeKind::Array) {
                const TypeData& element = types_.get(data.elements.front());
                const std::int64_t length = constant(data.value, "array length");
                for (std::int64_t i = 0; i < length; ++i) {
                    collect_parameters(element.decl,
                                       block_substitution(element),
                                       path + "." + std::to_string(i) + ".");
                }
            } else {
                const bool is_optional = data.kind == TypeKind::Optional;
                if (is_optional && !options_.optionals_present) {
                    continue;
                }
                const TypeId tensor = is_optional ? data.elements.front() : type;
                Val value = tensor_value(tensor, {});
                value.name = "%param" + std::to_string(parameters_.size());
                parameters_.push_back({path, value});
            }
        }
    }

    Substitution block_substitution(const TypeData& block_type) const {
        Substitution subst;
        const DeclInfo& info = model_.decls.at(block_type.decl);
        for (std::size_t i = 0; i < info.generics.size() && i < block_type.args.size(); ++i) {
            const GenericInfo& generic = info.generics[i];
            switch (generic.kind) {
            case GenericKind::Dim:
                subst.dims[generic.symbol] = block_type.args[i].dim;
                break;
            case GenericKind::Pack:
                subst.packs[generic.symbol] = block_type.args[i].shape;
                break;
            case GenericKind::DType:
                subst.dtypes[generic.dtype_var] = block_type.args[i].dtype;
                break;
            }
        }
        return subst;
    }

    const Val* parameter(const std::string& path) const {
        for (const Parameter& parameter : parameters_) {
            if (parameter.path == path) {
                return &parameter.value;
            }
        }
        return nullptr;
    }

    // ------------------------------------------------------------ types

    std::int64_t constant(const shape::Poly& poly, const char* what) const {
        const auto value = poly.constant();
        if (!value) {
            fail(std::string(what) + " `" + model_.dims.to_string(poly) +
                 "` is not a constant; bind every generic with --bind");
        }
        return *value;
    }

    Dims concrete_shape(const Shape& shape) const {
        Dims dims;
        for (const ShapeElem& unit : shape) {
            if (unit.is_pack) {
                fail("shape pack `" + std::string(model_.dims.symbol_name(unit.pack)) +
                     "` is not bound");
            }
            dims.push_back(constant(unit.dim, "dimension"));
        }
        return dims;
    }

    ScalarKind concrete_dtype(DType dtype) const {
        if (dtype.is_var) {
            fail("dtype `" + types_.dtype_var(dtype.var).name + "` is not bound");
        }
        return dtype.scalar;
    }

    // A tensor value (unnamed) with the concrete shape and dtype of a type.
    Val tensor_value(TypeId type, const Substitution& subst) {
        const TypeData& data = types_.get(types_.substitute(type, subst));
        Val value;
        if (data.kind == TypeKind::Scalar || data.kind == TypeKind::CompileInt) {
            value.dtype =
                data.kind == TypeKind::Scalar ? concrete_dtype(data.dtype) : ScalarKind::I64;
        } else if (data.kind == TypeKind::Tensor) {
            value.shape = concrete_shape(data.shape);
            value.dtype = concrete_dtype(data.dtype);
        } else {
            fail("type `" + types_.to_string(type) + "` has no tensor representation");
        }
        return value;
    }

    // --------------------------------------------------------- emission

    std::string fresh() { return "%" + std::to_string(next_++); }

    std::string type_of(const Val& value) const { return tensor_type(value.shape, value.dtype); }

    Val emit(const std::string& op,
             const std::vector<Val>& operands,
             const std::string& attributes,
             Dims shape,
             ScalarKind dtype,
             const std::string& region = "") {
        Val result;
        result.name = fresh();
        result.shape = std::move(shape);
        result.dtype = dtype;
        result.grid_rank = grid_.size();
        std::string line = indent_ + result.name + " = \"stablehlo." + op + "\"(";
        std::string types;
        for (std::size_t i = 0; i < operands.size(); ++i) {
            line += (i == 0 ? "" : ", ") + operands[i].name;
            types += (i == 0 ? "" : ", ") + type_of(operands[i]);
        }
        line += ")";
        if (!region.empty()) {
            line += " (" + region + ")";
        }
        if (!attributes.empty()) {
            line += " {" + attributes + "}";
        }
        line += " : (" + types + ") -> " + type_of(result) + "\n";
        body_ += line;
        return result;
    }

    Val constant_scalar(const std::string& literal, ScalarKind dtype) {
        return emit("constant",
                    {},
                    "value = dense<" + literal + "> : " + tensor_type({}, dtype),
                    {},
                    dtype);
    }

    Val constant_int(std::int64_t value, ScalarKind dtype) {
        if (dtype == ScalarKind::Bool) {
            return constant_scalar(value != 0 ? "true" : "false", dtype);
        }
        if (is_float(dtype)) {
            return constant_scalar(float_text(static_cast<double>(value)), dtype);
        }
        return constant_scalar(std::to_string(value), dtype);
    }

    // Broadcasts `value` to `shape`, mapping its axes to the given result axes.
    Val broadcast(const Val& value, const Dims& shape, const std::vector<std::int64_t>& dims) {
        if (value.shape == shape) {
            return value;
        }
        return emit("broadcast_in_dim",
                    {value},
                    "broadcast_dimensions = " + i64_array(dims),
                    shape,
                    value.dtype);
    }

    // Right-aligned broadcasting, as Linnet's elementwise operators use.
    Val broadcast_trailing(const Val& value, const Dims& shape) {
        std::vector<std::int64_t> dims;
        dims.reserve(value.shape.size());
        for (std::size_t i = 0; i < value.shape.size(); ++i) {
            dims.push_back(static_cast<std::int64_t>(shape.size() - value.shape.size() + i));
        }
        return broadcast(value, shape, dims);
    }

    Val convert(const Val& value, ScalarKind dtype) {
        if (value.dtype == dtype) {
            return value;
        }
        return emit("convert", {value}, "", value.shape, dtype);
    }

    // ------------------------------------------------------------- grid

    bool in_grid() const { return !grid_.empty(); }

    // The position along grid axis `axis`, as an i64 tensor over the grid.
    Val iota(std::size_t axis) {
        const Val positions =
            emit("iota", {}, "iota_dimension = 0 : i64", {grid_[axis]}, ScalarKind::I64);
        return broadcast(positions, grid_, {static_cast<std::int64_t>(axis)});
    }

    // A value as a tensor over the current grid: outer-grid values and
    // scalars are broadcast, index variables materialized.
    Val to_grid(const Val& value) {
        if (value.kind == Val::Kind::Index) {
            return iota(value.axes.front());
        }
        if (value.kind != Val::Kind::Tensor) {
            fail("a compile-time value was used where a tensor is needed");
        }
        if (value.shape == grid_) {
            return value;
        }
        std::vector<std::int64_t> dims;
        for (std::size_t i = 0; i < value.shape.size(); ++i) {
            if (i >= value.grid_rank || i >= grid_.size() || value.shape[i] != grid_[i]) {
                fail("a tensor value of rank " + std::to_string(value.shape.size()) +
                     " was used as a scalar inside index notation");
            }
            dims.push_back(static_cast<std::int64_t>(i));
        }
        return broadcast(value, grid_, dims);
    }

    // Operands of an elementwise operation, all with the shape of the result.
    std::vector<Val> aligned(const std::vector<Val>& operands, const Dims& shape) {
        std::vector<Val> out;
        out.reserve(operands.size());
        for (const Val& operand : operands) {
            out.push_back(in_grid() ? to_grid(operand) : broadcast_trailing(operand, shape));
        }
        return out;
    }

    // ---------------------------------------------------------- running

    Frame& frame() { return frames_.back(); }

    const Val& value(ir::ValueId id) {
        const auto found = frame().values.find(id);
        if (found == frame().values.end()) {
            fail("internal: value used before definition");
        }
        return found->second;
    }

    // The result shape of an operation from its IR type, or the grid.
    Dims result_shape(const ir::Operation& op) {
        if (in_grid()) {
            return grid_;
        }
        return tensor_value(module_.value(op.results.front()).type, frame().subst).shape;
    }

    ScalarKind result_dtype(const ir::Operation& op) {
        return tensor_value(module_.value(op.results.front()).type, frame().subst).dtype;
    }

    // Runs a block's operations and returns its terminator's operands.
    std::vector<Val> run_block(const ir::Block& block) {
        for (const ir::OpId id : block.ops) {
            const ir::Operation& op = module_.op(id);
            if (op.kind == ir::OpKind::Return || op.kind == ir::OpKind::Yield) {
                std::vector<Val> results;
                results.reserve(op.operands.size());
                for (const ir::ValueId operand : op.operands) {
                    results.push_back(value(operand));
                }
                return results;
            }
            run_op(op);
        }
        return {};
    }

    // Runs a region with the given block arguments.
    std::vector<Val> run_region(ir::RegionId region, const std::vector<Val>& arguments) {
        const ir::Block& block = module_.block(module_.region(region).blocks.front());
        for (std::size_t i = 0; i < arguments.size() && i < block.arguments.size(); ++i) {
            frame().values[block.arguments[i]] = arguments[i];
        }
        return run_block(block);
    }

    void define(const ir::Operation& op, Val value) {
        frame().values[op.results.front()] = std::move(value);
    }

    // Inside index notation most operations are scalar and evaluate over the
    // grid, but a tensor-valued expression there (`iota(N)[i]`, a call whose
    // result is then indexed) is a whole tensor computed once.
    void run_op(const ir::Operation& op) {
        if (in_grid() && op.results.size() == 1 && op.kind != ir::OpKind::Reduce) {
            const TypeData& data = types_.get(
                types_.substitute(module_.value(op.results.front()).type, frame().subst));
            if (data.kind == TypeKind::Tensor) {
                const Dims saved = grid_;
                grid_.clear();
                run_scalar_or_tensor_op(op);
                grid_ = saved;
                return;
            }
        }
        run_scalar_or_tensor_op(op);
    }

    void run_scalar_or_tensor_op(const ir::Operation& op) {
        const ir::Attributes& a = op.attributes;
        const auto operand = [&](std::size_t i) -> const Val& { return value(op.operands[i]); };
        const auto elementwise = [&](const char* name, ScalarKind dtype) {
            const Dims shape = result_shape(op);
            std::vector<Val> operands;
            operands.reserve(op.operands.size());
            for (const ir::ValueId id : op.operands) {
                operands.push_back(value(id));
            }
            define(op, emit(name, aligned(operands, shape), "", shape, dtype));
        };
        switch (op.kind) {
        case ir::OpKind::ConstInt:
            define(op, constant_int(a.integer, result_dtype(op)));
            return;
        case ir::OpKind::ConstBool:
            define(op, constant_int(a.integer, ScalarKind::Bool));
            return;
        case ir::OpKind::ConstFloat: {
            const ScalarKind dtype = result_dtype(op);
            define(op, constant_scalar(float_text(a.number), dtype));
            return;
        }
        case ir::OpKind::ConstDim:
            define(op,
                   constant_int(constant(types_.substitute(a.dim, frame().subst), "dimension"),
                                ScalarKind::I64));
            return;
        case ir::OpKind::Add:
            elementwise("add", result_dtype(op));
            return;
        case ir::OpKind::Sub:
            elementwise("subtract", result_dtype(op));
            return;
        case ir::OpKind::Mul:
            elementwise("multiply", result_dtype(op));
            return;
        case ir::OpKind::Div:
            elementwise("divide", result_dtype(op));
            return;
        case ir::OpKind::Rem:
            elementwise("remainder", result_dtype(op));
            return;
        case ir::OpKind::Min:
            elementwise("minimum", result_dtype(op));
            return;
        case ir::OpKind::Max:
            elementwise("maximum", result_dtype(op));
            return;
        case ir::OpKind::And:
            elementwise("and", ScalarKind::Bool);
            return;
        case ir::OpKind::Or:
            elementwise("or", ScalarKind::Bool);
            return;
        case ir::OpKind::Not:
            elementwise("not", ScalarKind::Bool);
            return;
        case ir::OpKind::Neg:
            elementwise("negate", result_dtype(op));
            return;
        case ir::OpKind::Exp:
            elementwise("exponential", result_dtype(op));
            return;
        case ir::OpKind::Log:
            elementwise("log", result_dtype(op));
            return;
        case ir::OpKind::Sqrt:
            elementwise("sqrt", result_dtype(op));
            return;
        case ir::OpKind::Rsqrt:
            elementwise("rsqrt", result_dtype(op));
            return;
        case ir::OpKind::Sin:
            elementwise("sine", result_dtype(op));
            return;
        case ir::OpKind::Cos:
            elementwise("cosine", result_dtype(op));
            return;
        case ir::OpKind::Tanh:
            elementwise("tanh", result_dtype(op));
            return;
        case ir::OpKind::Abs:
            elementwise("abs", result_dtype(op));
            return;
        case ir::OpKind::Compare: {
            const Dims shape = result_shape(op);
            const std::vector<Val> operands = aligned({operand(0), operand(1)}, shape);
            const char* direction = a.compare == ir::CompareKind::Eq   ? "EQ"
                                    : a.compare == ir::CompareKind::Ne ? "NE"
                                    : a.compare == ir::CompareKind::Lt ? "LT"
                                    : a.compare == ir::CompareKind::Le ? "LE"
                                    : a.compare == ir::CompareKind::Gt ? "GT"
                                                                       : "GE";
            define(op,
                   emit("compare",
                        operands,
                        std::string("comparison_direction = #stablehlo<comparison_direction ") +
                            direction + ">",
                        shape,
                        ScalarKind::Bool));
            return;
        }
        case ir::OpKind::Select: {
            const Dims shape = result_shape(op);
            const std::vector<Val> operands = aligned({operand(0), operand(1), operand(2)}, shape);
            define(op, emit("select", operands, "", shape, result_dtype(op)));
            return;
        }
        case ir::OpKind::Cast: {
            const Val source = in_grid() ? to_grid(operand(0)) : operand(0);
            define(op, convert(source, result_dtype(op)));
            return;
        }
        case ir::OpKind::Reshape: {
            const Val& source = operand(0);
            define(op, emit("reshape", {source}, "", result_shape(op), source.dtype));
            return;
        }
        case ir::OpKind::Broadcast:
            define(op, broadcast_trailing(operand(0), result_shape(op)));
            return;
        case ir::OpKind::Permute: {
            std::vector<std::int64_t> permutation;
            for (const ShapeElem& axis : a.shape) {
                permutation.push_back(constant(axis.dim, "axis"));
            }
            const Val& source = operand(0);
            define(op,
                   emit("transpose",
                        {source},
                        "permutation = " + i64_array(permutation),
                        result_shape(op),
                        source.dtype));
            return;
        }
        case ir::OpKind::Slice: {
            const Val& source = operand(0);
            std::vector<std::int64_t> starts;
            std::vector<std::int64_t> limits;
            std::vector<std::int64_t> strides;
            Dims sliced;
            std::size_t axis = 0;
            for (std::size_t i = 0; i < a.starts.size(); ++i) {
                // A whole shape-pack entry covers every axis the pack binds to.
                const std::size_t count =
                    a.whole[i] && a.pack_units[i].is_pack
                        ? concrete_shape(types_.substitute(Shape{a.pack_units[i]}, frame().subst))
                              .size()
                        : 1;
                for (std::size_t k = 0; k < count; ++k, ++axis) {
                    const std::int64_t size = source.shape[axis];
                    const std::int64_t start =
                        a.whole[i] ? 0
                                   : constant(types_.substitute(a.starts[i], frame().subst),
                                              "slice start");
                    const std::int64_t stop =
                        a.whole[i]
                            ? size
                            : constant(types_.substitute(a.stops[i], frame().subst), "slice stop");
                    const std::int64_t step = a.whole[i] ? 1 : a.steps[i];
                    starts.push_back(start);
                    limits.push_back(stop);
                    strides.push_back(step);
                    sliced.push_back((stop - start + step - 1) / step);
                }
            }
            Val result = emit("slice",
                              {source},
                              "start_indices = " + i64_array(starts) + ", limit_indices = " +
                                  i64_array(limits) + ", strides = " + i64_array(strides),
                              sliced,
                              source.dtype);
            const Dims shape = result_shape(op);
            if (shape != sliced) {
                result = emit("reshape", {result}, "", shape, source.dtype);
            }
            define(op, result);
            return;
        }
        case ir::OpKind::Concat: {
            std::vector<Val> parts;
            parts.reserve(op.operands.size());
            for (const ir::ValueId id : op.operands) {
                parts.push_back(value(id));
            }
            const std::int64_t rank = static_cast<std::int64_t>(parts.front().shape.size());
            const std::int64_t axis = a.axis < 0 ? a.axis + rank : a.axis;
            define(op,
                   emit("concatenate",
                        parts,
                        "dimension = " + std::to_string(axis) + " : i64",
                        result_shape(op),
                        parts.front().dtype));
            return;
        }
        case ir::OpKind::Fill: {
            const Val& fill = operand(0);
            define(op, broadcast(fill, result_shape(op), {}));
            return;
        }
        case ir::OpKind::Iota: {
            const Dims shape = result_shape(op);
            const Val positions =
                emit("iota", {}, "iota_dimension = 0 : i64", shape, ScalarKind::I64);
            define(op, convert(positions, result_dtype(op)));
            return;
        }
        case ir::OpKind::Element:
            define(op, element(op));
            return;
        case ir::OpKind::Comprehension:
            define(op, comprehension(op));
            return;
        case ir::OpKind::Reduce:
            define(op, reduce(op));
            return;
        case ir::OpKind::TupleMake: {
            Val tuple;
            tuple.kind = Val::Kind::Tuple;
            for (const ir::ValueId id : op.operands) {
                tuple.elements.push_back(value(id));
            }
            define(op, tuple);
            return;
        }
        case ir::OpKind::TupleGet: {
            const Val& tuple = operand(0);
            if (tuple.kind != Val::Kind::Tuple ||
                static_cast<std::size_t>(a.integer) >= tuple.elements.size()) {
                fail("internal: tuple.get on a non-tuple");
            }
            define(op, tuple.elements[static_cast<std::size_t>(a.integer)]);
            return;
        }
        case ir::OpKind::OptionSome: {
            Val some;
            some.kind = Val::Kind::Some;
            some.elements.push_back(operand(0));
            define(op, some);
            return;
        }
        case ir::OpKind::OptionNone: {
            Val none;
            none.kind = Val::Kind::None;
            define(op, none);
            return;
        }
        case ir::OpKind::OptionMatch: {
            const Val& subject = operand(0);
            const std::vector<Val> results =
                subject.kind == Val::Kind::Some
                    ? run_region(op.regions[0], {subject.elements.front()})
                    : run_region(op.regions[1], {});
            define(op, results.front());
            return;
        }
        case ir::OpKind::If: {
            // Both branches are pure; select between them.
            const Val& condition = operand(0);
            const Val then_value = run_region(op.regions[0], {}).front();
            const Val else_value = run_region(op.regions[1], {}).front();
            if (then_value.kind != Val::Kind::Tensor) {
                fail("`if` over compile-time values is not supported");
            }
            const Dims shape = result_shape(op);
            define(op,
                   emit("select",
                        aligned({condition, then_value, else_value}, shape),
                        "",
                        shape,
                        then_value.dtype));
            return;
        }
        case ir::OpKind::Call:
        case ir::OpKind::SemanticCall:
            call(op);
            return;
        case ir::OpKind::BlockParam: {
            const Val& block = operand(0);
            const std::string path = block.path + a.name;
            const TypeId member_type = member(block, a.name);
            const TypeData& data = types_.get(types_.substitute(member_type, block.subst));
            const Val* found = parameter(path);
            if (data.kind == TypeKind::Optional) {
                Val optional;
                optional.kind = found == nullptr ? Val::Kind::None : Val::Kind::Some;
                if (found != nullptr) {
                    optional.elements.push_back(*found);
                }
                define(op, optional);
                return;
            }
            if (found == nullptr) {
                fail("internal: parameter `" + path + "` was not collected");
            }
            define(op, *found);
            return;
        }
        case ir::OpKind::BlockSub: {
            const Val& block = operand(0);
            const TypeId member_type = member(block, a.name);
            const TypeData& data = types_.get(types_.substitute(member_type, block.subst));
            define(op, sub_value(data, block.path + a.name));
            return;
        }
        case ir::OpKind::ArrayGet: {
            const Val& array = operand(0);
            const std::int64_t position = constant_of(op.operands[1]);
            if (array.kind != Val::Kind::Array || position < 0 ||
                static_cast<std::size_t>(position) >= array.elements.size()) {
                fail("array index out of range while unrolling");
            }
            define(op, array.elements[static_cast<std::size_t>(position)]);
            return;
        }
        case ir::OpKind::StaticFor: {
            const Val& array = operand(0);
            if (array.kind != Val::Kind::Array) {
                fail("`static for` over a non-array");
            }
            std::vector<Val> carried;
            for (std::size_t i = 1; i < op.operands.size(); ++i) {
                carried.push_back(operand(i));
            }
            for (const Val& element : array.elements) {
                std::vector<Val> arguments{element};
                arguments.insert(arguments.end(), carried.begin(), carried.end());
                carried = run_region(op.regions.front(), arguments);
            }
            for (std::size_t i = 0; i < op.results.size(); ++i) {
                frame().values[op.results[i]] = carried[i];
            }
            return;
        }
        case ir::OpKind::EnumConst: {
            // Enum values are compile-time; a match on one picks its arm.
            Val variant;
            variant.kind = Val::Kind::Enum;
            variant.path = a.name;
            define(op, variant);
            return;
        }
        case ir::OpKind::EnumMatch: {
            const Val& subject = operand(0);
            if (subject.kind != Val::Kind::Enum) {
                fail("`match` on a runtime enum value is not supported");
            }
            // Arms are named by variant; a catch-all arm is `_`.
            for (std::size_t i = 0; i < a.names.size() && i < op.regions.size(); ++i) {
                if (a.names[i] == subject.path || a.names[i] == "_") {
                    define(op, run_region(op.regions[i], {subject}).front());
                    return;
                }
            }
            fail("no arm matches enum variant `" + subject.path + "`");
        }
        case ir::OpKind::StructMake:
        case ir::OpKind::StructGet:
            fail(std::string("`") + std::string(ir::op_spelling(op.kind)) +
                 "` is not supported by the StableHLO exporter");
        case ir::OpKind::Yield:
        case ir::OpKind::Return:
            return;
        }
    }

    // The compile-time integer an index value stands for.
    std::int64_t constant_of(ir::ValueId id) {
        const ir::OpId producer = module_.value(id).producer;
        if (producer != ir::no_id) {
            const ir::Operation& op = module_.op(producer);
            if (op.kind == ir::OpKind::ConstInt) {
                return op.attributes.integer;
            }
            if (op.kind == ir::OpKind::ConstDim) {
                return constant(types_.substitute(op.attributes.dim, frame().subst), "index");
            }
        }
        fail("array indices must be compile-time constants");
    }

    TypeId member(const Val& block, const std::string& name) const {
        if (block.kind != Val::Kind::Block) {
            fail("internal: member access on a non-block value");
        }
        const DeclInfo& info = model_.decls.at(block.block);
        const auto found = info.scope.find(name);
        if (found == info.scope.end()) {
            fail("internal: no member `" + name + "`");
        }
        return model_.entities[found->second].type;
    }

    Val sub_value(const TypeData& data, const std::string& path) {
        if (data.kind == TypeKind::Block) {
            Val block;
            block.kind = Val::Kind::Block;
            block.block = data.decl;
            block.subst = block_substitution(data);
            block.path = path + ".";
            return block;
        }
        if (data.kind == TypeKind::Array) {
            Val array;
            array.kind = Val::Kind::Array;
            const std::int64_t length = constant(data.value, "array length");
            for (std::int64_t i = 0; i < length; ++i) {
                array.elements.push_back(
                    sub_value(types_.get(data.elements.front()), path + "." + std::to_string(i)));
            }
            return array;
        }
        fail("internal: `sub` of a non-block type");
    }

    // ------------------------------------------------------------ calls

    void call(const ir::Operation& op) {
        const auto found = functions_.find(op.attributes.name);
        if (found == functions_.end()) {
            fail("callee `" + op.attributes.name + "` is not defined in this program");
        }
        const ir::Function& callee = *found->second;
        Frame inner;
        inner.subst = op.attributes.substitution;
        for (auto& [symbol, dim] : inner.subst.dims) {
            dim = types_.substitute(dim, frame().subst);
        }
        for (auto& [symbol, shape] : inner.subst.packs) {
            shape = types_.substitute(shape, frame().subst);
        }
        for (auto& [var, dtype] : inner.subst.dtypes) {
            dtype = types_.substitute(dtype, frame().subst);
        }
        std::vector<Val> arguments;
        arguments.reserve(op.operands.size());
        for (const ir::ValueId id : op.operands) {
            arguments.push_back(value(id));
        }
        // A method's generic bindings include its block's.
        if (!arguments.empty() && arguments.front().kind == Val::Kind::Block) {
            for (const auto& [symbol, dim] : arguments.front().subst.dims) {
                inner.subst.dims.emplace(symbol, dim);
            }
            for (const auto& [symbol, shape] : arguments.front().subst.packs) {
                inner.subst.packs.emplace(symbol, shape);
            }
            for (const auto& [var, dtype] : arguments.front().subst.dtypes) {
                inner.subst.dtypes.emplace(var, dtype);
            }
        }
        const Dims saved_grid = grid_;
        grid_.clear();
        frames_.push_back(std::move(inner));
        const std::vector<Val> results = run_region(callee.body, arguments);
        frames_.pop_back();
        grid_ = saved_grid;
        if (!op.results.empty()) {
            if (results.empty()) {
                fail("internal: call without a result");
            }
            Val result = results.front();
            result.grid_rank = grid_.size();
            define(op, result);
        }
    }

    // ------------------------------------------------- index notation

    // Block arguments for the index variables of a comprehension or
    // reduction, extending the grid by their domains.
    std::vector<Val> index_arguments(const ir::Operation& op) {
        const ir::Block& body = module_.block(module_.region(op.regions.front()).blocks.front());
        std::vector<Val> arguments;
        for (std::size_t i = 0; i < body.arguments.size(); ++i) {
            const TypeData& data =
                types_.get(types_.substitute(module_.value(body.arguments[i]).type, frame().subst));
            Val index;
            if (data.kind == TypeKind::ShapeValue) {
                index.kind = Val::Kind::Pack;
                for (const std::int64_t dim : concrete_shape(data.shape)) {
                    index.axes.push_back(grid_.size());
                    grid_.push_back(dim);
                }
            } else {
                index.kind = Val::Kind::Index;
                const ShapeElem& domain = op.attributes.shape[i];
                const Shape unit = types_.substitute(Shape{domain}, frame().subst);
                const Dims dims = concrete_shape(unit);
                if (dims.size() != 1) {
                    fail("an index domain must be one dimension");
                }
                index.axes.push_back(grid_.size());
                grid_.push_back(dims.front());
            }
            arguments.push_back(index);
        }
        return arguments;
    }

    Val comprehension(const ir::Operation& op) {
        if (in_grid()) {
            fail("internal: comprehension inside a region");
        }
        const std::vector<Val> arguments = index_arguments(op);
        Val result = run_region(op.regions.front(), arguments).front();
        result = to_grid(result);
        result = convert(result, result_dtype_outside(op));
        grid_.clear();
        result.grid_rank = 0;
        return result;
    }

    ScalarKind result_dtype_outside(const ir::Operation& op) {
        return tensor_value(module_.value(op.results.front()).type, frame().subst).dtype;
    }

    Val reduce(const ir::Operation& op) {
        const Dims outer = grid_;
        const std::vector<Val> arguments = index_arguments(op);
        Val body = to_grid(run_region(op.regions.front(), arguments).front());
        const ScalarKind dtype = result_dtype_outside(op);
        body = convert(body, dtype);
        std::vector<std::int64_t> dimensions;
        for (std::size_t i = outer.size(); i < grid_.size(); ++i) {
            dimensions.push_back(static_cast<std::int64_t>(i));
        }
        grid_ = outer;
        const char* combine = "add";
        std::string init;
        switch (op.attributes.reduce) {
        case ir::ReduceKind::Sum:
            init = is_float(dtype) ? "0.0" : "0";
            break;
        case ir::ReduceKind::Prod:
            combine = "multiply";
            init = is_float(dtype) ? "1.0" : "1";
            break;
        case ir::ReduceKind::Max:
            combine = "maximum";
            init = lowest(dtype);
            break;
        case ir::ReduceKind::Min:
            combine = "minimum";
            init = highest(dtype);
            break;
        case ir::ReduceKind::Any:
            combine = "or";
            init = "false";
            break;
        case ir::ReduceKind::All:
            combine = "and";
            init = "true";
            break;
        }
        const Val initial = constant_scalar(init, dtype);
        const std::string a = fresh();
        const std::string b = fresh();
        const std::string scalar = tensor_type({}, dtype);
        const std::string combined = fresh();
        const std::string region =
            "{\n" + indent_ + "  ^bb0(" + a + ": " + scalar + ", " + b + ": " + scalar + "):\n" +
            indent_ + "    " + combined + " = \"stablehlo." + combine + "\"(" + a + ", " + b +
            ") : (" + scalar + ", " + scalar + ") -> " + scalar + "\n" + indent_ +
            "    \"stablehlo.return\"(" + combined + ") : (" + scalar + ") -> ()\n" + indent_ + "}";
        return emit("reduce",
                    {body, initial},
                    "dimensions = " + i64_array(dimensions),
                    outer,
                    dtype,
                    region);
    }

    static std::string lowest(ScalarKind dtype) {
        switch (dtype) {
        case ScalarKind::F16:
            return "0xFC00";
        case ScalarKind::BF16:
            return "0xFF80";
        case ScalarKind::F32:
            return "0xFF800000";
        case ScalarKind::F64:
            return "0xFFF0000000000000";
        case ScalarKind::Bool:
            return "false";
        case ScalarKind::I8:
            return "-128";
        case ScalarKind::I16:
            return "-32768";
        case ScalarKind::I32:
            return "-2147483648";
        case ScalarKind::I64:
            return "-9223372036854775808";
        default:
            return "0";
        }
    }

    static std::string highest(ScalarKind dtype) {
        switch (dtype) {
        case ScalarKind::F16:
            return "0x7C00";
        case ScalarKind::BF16:
            return "0x7F80";
        case ScalarKind::F32:
            return "0x7F800000";
        case ScalarKind::F64:
            return "0x7FF0000000000000";
        case ScalarKind::Bool:
            return "true";
        case ScalarKind::I8:
            return "127";
        case ScalarKind::I16:
            return "32767";
        case ScalarKind::I32:
            return "2147483647";
        case ScalarKind::I64:
            return "9223372036854775807";
        case ScalarKind::U8:
            return "255";
        case ScalarKind::U16:
            return "65535";
        case ScalarKind::U32:
            return "4294967295";
        case ScalarKind::U64:
            return "18446744073709551615";
        }
        return "0";
    }

    // `x[i, j]` over the grid: a broadcast when every index is a distinct
    // grid position, a gather otherwise.
    Val element(const ir::Operation& op) {
        const Val& source = value(op.operands.front());
        if (source.kind != Val::Kind::Tensor) {
            fail("internal: element access on a non-tensor");
        }
        std::vector<std::int64_t> axes;
        bool is_positional = true;
        for (std::size_t i = 1; i < op.operands.size(); ++i) {
            const Val& index = value(op.operands[i]);
            if (index.kind == Val::Kind::Index || index.kind == Val::Kind::Pack) {
                for (const std::size_t axis : index.axes) {
                    axes.push_back(static_cast<std::int64_t>(axis));
                }
            } else {
                is_positional = false;
            }
        }
        std::vector<std::int64_t> sorted = axes;
        std::sort(sorted.begin(), sorted.end());
        is_positional = is_positional && axes.size() == source.shape.size() &&
                        std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
        if (is_positional) {
            bool fits = true;
            for (std::size_t i = 0; i < axes.size(); ++i) {
                fits = fits && grid_[static_cast<std::size_t>(axes[i])] == source.shape[i];
            }
            if (fits) {
                return broadcast(source, grid_, axes);
            }
        }
        // General case: one i64 position per operand axis, gathered.
        std::vector<Val> columns;
        for (std::size_t i = 1; i < op.operands.size(); ++i) {
            const Val& index = value(op.operands[i]);
            std::vector<Val> positions;
            if (index.kind == Val::Kind::Pack) {
                for (const std::size_t axis : index.axes) {
                    positions.push_back(iota(axis));
                }
            } else {
                positions.push_back(convert(to_grid(index), ScalarKind::I64));
            }
            for (const Val& position : positions) {
                Dims column_shape = grid_;
                column_shape.push_back(1);
                columns.push_back(emit("reshape", {position}, "", column_shape, ScalarKind::I64));
            }
        }
        if (columns.size() != source.shape.size()) {
            fail("internal: element access with the wrong number of indices");
        }
        Dims indices_shape = grid_;
        indices_shape.push_back(static_cast<std::int64_t>(columns.size()));
        const Val indices = columns.size() == 1
                                ? columns.front()
                                : emit("concatenate",
                                       columns,
                                       "dimension = " + std::to_string(grid_.size()) + " : i64",
                                       indices_shape,
                                       ScalarKind::I64);
        std::vector<std::int64_t> all_axes;
        std::vector<std::int64_t> ones;
        for (std::size_t i = 0; i < source.shape.size(); ++i) {
            all_axes.push_back(static_cast<std::int64_t>(i));
            ones.push_back(1);
        }
        return emit(
            "gather",
            {source, indices},
            "dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = " +
                index_list(all_axes) + ", start_index_map = " + index_list(all_axes) +
                ", index_vector_dim = " + std::to_string(grid_.size()) +
                ">, indices_are_sorted = false, slice_sizes = " + i64_array(ones),
            grid_,
            source.dtype);
    }

    ir::Module& module_;
    const Model& model_;
    TypeStore& types_;
    const StableHloOptions& options_;
    std::map<std::string, const ir::Function*> functions_;
    std::vector<Parameter> parameters_;
    std::deque<Frame> frames_; // deque: frames stay put while calls push new ones
    Dims grid_;
    std::string body_;
    std::string indent_ = "    ";
    std::size_t next_ = 0;
};

} // namespace

std::expected<std::string, std::string> export_stablehlo(ir::Module& module,
                                                         const StableHloOptions& options) {
    try {
        return Exporter(module, options).run();
    } catch (const Unsupported& error) {
        return std::unexpected(error.what());
    }
}

} // namespace linnet::backend
