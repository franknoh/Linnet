#include "linnet/backend/stablehlo.hpp"

#include "linnet/backend/graph_export.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace linnet::backend {

namespace {

using sema::ScalarKind;

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

std::string tensor_type(const TensorInfo& info) {
    return tensor_type(info.shape, info.dtype);
}

std::string i64_array(const Dims& values) {
    std::string text = "array<i64";
    for (std::size_t i = 0; i < values.size(); ++i) {
        text += (i == 0 ? ": " : ", ") + std::to_string(values[i]);
    }
    return text + ">";
}

std::string index_list(const Dims& values) {
    std::string text = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        text += (i == 0 ? "" : ", ") + std::to_string(values[i]);
    }
    return text + "]";
}

// MLIR wants a decimal point in the mantissa: `1.0e30`, not `1e30`.
std::string float_text(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    std::string text = buffer;
    const std::size_t exponent = text.find_first_of("eE");
    if (text.find('.') == std::string::npos) {
        text.insert(exponent == std::string::npos ? text.size() : exponent, ".0");
    }
    return text;
}

// Bit patterns of -inf and +inf, and integer extremes, as MLIR literals.
std::string lowest(ScalarKind dtype) {
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

std::string highest(ScalarKind dtype) {
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

const char* elementwise_name(Elementwise kind) {
    switch (kind) {
    case Elementwise::Add:
        return "add";
    case Elementwise::Sub:
        return "subtract";
    case Elementwise::Mul:
        return "multiply";
    case Elementwise::Div:
        return "divide";
    case Elementwise::Rem:
        return "remainder";
    case Elementwise::Min:
        return "minimum";
    case Elementwise::Max:
        return "maximum";
    case Elementwise::And:
        return "and";
    case Elementwise::Or:
        return "or";
    case Elementwise::Not:
        return "not";
    case Elementwise::Neg:
        return "negate";
    case Elementwise::Exp:
        return "exponential";
    case Elementwise::Log:
        return "log";
    case Elementwise::Sqrt:
        return "sqrt";
    case Elementwise::Rsqrt:
        return "rsqrt";
    case Elementwise::Sin:
        return "sine";
    case Elementwise::Cos:
        return "cosine";
    case Elementwise::Tanh:
        return "tanh";
    case Elementwise::Abs:
        return "abs";
    }
    return "add";
}

// StableHLO in MLIR's generic operation form, which every version parses.
class StableHloTarget : public GraphTarget {
public:
    std::string input(const std::string& name, const Dims& shape, ScalarKind dtype) override {
        const std::string tensor = "%" + name;
        arguments_.push_back(tensor + ": " + tensor_type(shape, dtype));
        return tensor;
    }

    std::string parameter(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        const std::string tensor = "%param" + std::to_string(parameters_++);
        arguments_.push_back(tensor + ": " + tensor_type(shape, dtype) + " {linnet.path = \"" +
                             path + "\"}");
        return tensor;
    }

    std::string state(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        const std::string tensor = "%state" + std::to_string(states_++);
        arguments_.push_back(tensor + ": " + tensor_type(shape, dtype) + " {linnet.state = \"" +
                             path + "\"}");
        return tensor;
    }

    std::string constant(const Literal& literal, ScalarKind dtype) override {
        std::string text;
        switch (literal.kind) {
        case Literal::Kind::Integer:
            text = std::to_string(literal.integer);
            break;
        case Literal::Kind::Real:
            text = float_text(literal.real);
            break;
        case Literal::Kind::Boolean:
            text = literal.integer != 0 ? "true" : "false";
            break;
        case Literal::Kind::Lowest:
            text = lowest(dtype);
            break;
        case Literal::Kind::Highest:
            text = highest(dtype);
            break;
        }
        return emit(
            "constant", {}, "value = dense<" + text + "> : " + tensor_type({}, dtype), {}, dtype);
    }

    std::string elementwise(Elementwise kind,
                            const std::vector<TensorInfo>& operands,
                            const Dims& shape,
                            ScalarKind dtype) override {
        return emit(elementwise_name(kind), operands, "", shape, dtype);
    }

    std::string compare(ir::CompareKind kind,
                        const TensorInfo& a,
                        const TensorInfo& b,
                        const Dims& shape) override {
        const char* direction = kind == ir::CompareKind::Eq   ? "EQ"
                                : kind == ir::CompareKind::Ne ? "NE"
                                : kind == ir::CompareKind::Lt ? "LT"
                                : kind == ir::CompareKind::Le ? "LE"
                                : kind == ir::CompareKind::Gt ? "GT"
                                                              : "GE";
        return emit("compare",
                    {a, b},
                    std::string("comparison_direction = #stablehlo<comparison_direction ") +
                        direction + ">",
                    shape,
                    ScalarKind::Bool);
    }

    std::string select(const TensorInfo& condition,
                       const TensorInfo& on_true,
                       const TensorInfo& on_false,
                       const Dims& shape,
                       ScalarKind dtype) override {
        return emit("select", {condition, on_true, on_false}, "", shape, dtype);
    }

    std::string convert(const TensorInfo& value, ScalarKind dtype) override {
        return emit("convert", {value}, "", value.shape, dtype);
    }

    std::string reshape(const TensorInfo& value, const Dims& shape) override {
        return emit("reshape", {value}, "", shape, value.dtype);
    }

    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override {
        return emit(
            "transpose", {value}, "permutation = " + i64_array(permutation), shape, value.dtype);
    }

    std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) override {
        return emit("broadcast_in_dim",
                    {value},
                    "broadcast_dimensions = " + i64_array(dims),
                    shape,
                    value.dtype);
    }

    std::string slice(const TensorInfo& value,
                      const Dims& starts,
                      const Dims& limits,
                      const Dims& strides,
                      const Dims& shape) override {
        return emit("slice",
                    {value},
                    "start_indices = " + i64_array(starts) + ", limit_indices = " +
                        i64_array(limits) + ", strides = " + i64_array(strides),
                    shape,
                    value.dtype);
    }

    std::string
    concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) override {
        return emit("concatenate",
                    parts,
                    "dimension = " + std::to_string(axis) + " : i64",
                    shape,
                    parts.front().dtype);
    }

    std::string iota(std::int64_t length) override {
        return emit("iota", {}, "iota_dimension = 0 : i64", {length}, ScalarKind::I64);
    }

    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override {
        Dims all_axes;
        Dims ones;
        for (std::size_t i = 0; i < source.shape.size(); ++i) {
            all_axes.push_back(static_cast<std::int64_t>(i));
            ones.push_back(1);
        }
        return emit(
            "gather",
            {source, indices},
            "dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = " +
                index_list(all_axes) + ", start_index_map = " + index_list(all_axes) +
                ", index_vector_dim = " + std::to_string(shape.size()) +
                ">, indices_are_sorted = false, slice_sizes = " + i64_array(ones),
            shape,
            source.dtype);
    }

    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        const char* combine = "add";
        Literal init;
        const bool is_real = sema::is_float(body.dtype);
        switch (kind) {
        case Reduction::Sum:
            init.kind = is_real ? Literal::Kind::Real : Literal::Kind::Integer;
            break;
        case Reduction::Prod:
            combine = "multiply";
            init.kind = is_real ? Literal::Kind::Real : Literal::Kind::Integer;
            init.integer = 1;
            init.real = 1.0;
            break;
        case Reduction::Max:
            combine = "maximum";
            init.kind = Literal::Kind::Lowest;
            break;
        case Reduction::Min:
            combine = "minimum";
            init.kind = Literal::Kind::Highest;
            break;
        case Reduction::Any:
            combine = "or";
            init.kind = Literal::Kind::Boolean;
            break;
        case Reduction::All:
            combine = "and";
            init.kind = Literal::Kind::Boolean;
            init.integer = 1;
            break;
        }
        const TensorInfo initial{constant(init, body.dtype), {}, body.dtype};
        const std::string a = fresh();
        const std::string b = fresh();
        const std::string scalar = tensor_type({}, body.dtype);
        const std::string combined = fresh();
        const std::string region =
            "{\n" + indent_ + "  ^bb0(" + a + ": " + scalar + ", " + b + ": " + scalar + "):\n" +
            indent_ + "    " + combined + " = \"stablehlo." + combine + "\"(" + a + ", " + b +
            ") : (" + scalar + ", " + scalar + ") -> " + scalar + "\n" + indent_ +
            "    \"stablehlo.return\"(" + combined + ") : (" + scalar + ") -> ()\n" + indent_ + "}";
        return emit("reduce",
                    {body, initial},
                    "dimensions = " + i64_array(dims),
                    shape,
                    body.dtype,
                    region);
    }

    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        std::vector<TensorInfo> outputs = results;
        for (const auto& [path, value] : states) {
            outputs.push_back(value);
        }
        std::string names;
        std::string types;
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            names += (i == 0 ? "" : ", ") + outputs[i].name;
            types += (i == 0 ? "" : ", ") + tensor_type(outputs[i]);
        }
        body_ += indent_ + "\"func.return\"(" + names + ") : (" + types + ") -> ()\n";
        std::string out =
            "// " + block_name + "." + entry_name + " from module " + module_path +
            ". Arguments after the inputs are the\n"
            "// parameters of the block hierarchy, named by `linnet.path`, then the\n"
            "// state members read before the call, named by `linnet.state`. Results\n"
            "// after the entry's own are the assigned state members, listed in\n"
            "// `linnet.states`.\n";
        std::string module_name;
        for (const char c : module_path) {
            module_name += c == '.' ? '_' : c;
        }
        out += "module @" + module_name + " {\n";
        out += "  func.func @main(";
        for (std::size_t i = 0; i < arguments_.size(); ++i) {
            out += (i == 0 ? "" : ", ") + arguments_[i];
        }
        out += ") -> " + (outputs.size() == 1 ? types : "(" + types + ")");
        if (!states.empty()) {
            out += " attributes {linnet.states = [";
            for (std::size_t i = 0; i < states.size(); ++i) {
                out += (i == 0 ? "\"" : ", \"") + states[i].first + "\"";
            }
            out += "]}";
        }
        out += " {\n";
        out += body_;
        out += "  }\n}\n";
        return out;
    }

private:
    std::string fresh() { return "%" + std::to_string(next_++); }

    std::string emit(const std::string& op,
                     const std::vector<TensorInfo>& operands,
                     const std::string& attributes,
                     const Dims& shape,
                     ScalarKind dtype,
                     const std::string& region = "") {
        const std::string name = fresh();
        std::string line = indent_ + name + " = \"stablehlo." + op + "\"(";
        std::string types;
        for (std::size_t i = 0; i < operands.size(); ++i) {
            line += (i == 0 ? "" : ", ") + operands[i].name;
            types += (i == 0 ? "" : ", ") + tensor_type(operands[i]);
        }
        line += ")";
        if (!region.empty()) {
            line += " (" + region + ")";
        }
        if (!attributes.empty()) {
            line += " {" + attributes + "}";
        }
        line += " : (" + types + ") -> " + tensor_type(shape, dtype) + "\n";
        body_ += line;
        return name;
    }

    std::vector<std::string> arguments_;
    std::string body_;
    std::string indent_ = "    ";
    std::size_t next_ = 0;
    std::size_t parameters_ = 0;
    std::size_t states_ = 0;
};

} // namespace

std::expected<std::string, std::string> export_stablehlo(ir::Module& module,
                                                         const StableHloOptions& options) {
    StableHloTarget target;
    return export_graph(module, options, target);
}

} // namespace linnet::backend
