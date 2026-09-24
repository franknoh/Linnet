#include "linnet/backend/stablehlo.hpp"

#include "linnet/backend/graph_export.hpp"

#include <cstdio>
#include <optional>
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
    case Elementwise::BitAnd:
        return "and";
    case Elementwise::Or:
    case Elementwise::BitOr:
        return "or";
    case Elementwise::BitXor:
        return "xor";
    case Elementwise::Shl:
        return "shift_left";
    case Elementwise::Shr:
        return "shift_right_arithmetic"; // unsigned dtypes use the logical form
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
        const bool is_unsigned = dtype == ScalarKind::U8 || dtype == ScalarKind::U16 ||
                                 dtype == ScalarKind::U32 || dtype == ScalarKind::U64;
        const char* name = kind == Elementwise::Shr && is_unsigned ? "shift_right_logical"
                                                                   : elementwise_name(kind);
        return emit(name, operands, "", shape, dtype);
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

    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           ScalarKind dtype) override {
        // Contractions become `dot_general`; everything else keeps its
        // canonical body, which XLA fuses well on its own. `(input dtype)`
        // attention skips the f32 accumulation.
        const std::string suffix = "(input dtype)";
        const bool fast = implementation.ends_with(suffix);
        const std::string implementation_base =
            fast ? implementation.substr(0, implementation.size() - suffix.size()) : implementation;
        std::vector<const TensorInfo*> at;
        at.reserve(operands.size());
        for (const std::optional<TensorInfo>& operand : operands) {
            at.push_back(operand.has_value() ? &*operand : nullptr);
        }
        if (implementation == "torch.nn.functional.embedding" && operands.size() == 2 &&
            at[0] != nullptr && at[1] != nullptr) {
            // Rows of the table by id: a gather along axis 0 with the ids as
            // the index vectors, keeping the row axis as the offset dimension.
            const TensorInfo& ids = *at[0];
            const TensorInfo& table = *at[1];
            Dims index_shape = ids.shape;
            index_shape.push_back(1);
            const TensorInfo indices{reshape(ids, index_shape), index_shape, ids.dtype};
            return emit("gather",
                        {table, indices},
                        "dimension_numbers = #stablehlo.gather<offset_dims = [" +
                            std::to_string(ids.shape.size()) +
                            "], collapsed_slice_dims = [0], start_index_map = [0], "
                            "index_vector_dim = " +
                            std::to_string(ids.shape.size()) +
                            ">, indices_are_sorted = false, slice_sizes = " +
                            i64_array({1, table.shape[1]}),
                        shape,
                        table.dtype);
        }
        if (implementation == "torch.Tensor.index_copy" && operands.size() == 3 &&
            at[0] != nullptr && at[1] != nullptr && at[2] != nullptr) {
            // `dynamic_update_slice` takes one start index per dimension.
            Literal zero;
            zero.kind = Literal::Kind::Integer;
            zero.integer = 0;
            const TensorInfo origin{constant(zero, ScalarKind::I32), {}, ScalarKind::I32};
            const std::vector<TensorInfo> arguments{*at[0], *at[1], origin, origin, *at[2], origin};
            return emit("dynamic_update_slice", arguments, "", shape, dtype);
        }
        if (implementation == "torch.matmul" && operands.size() == 2 && at[0] != nullptr &&
            at[1] != nullptr) {
            const TensorInfo& a = *at[0];
            const TensorInfo& b = *at[1];
            const std::int64_t rank = static_cast<std::int64_t>(a.shape.size());
            Dims batch;
            for (std::int64_t i = 0; i + 2 < rank; ++i) {
                batch.push_back(i);
            }
            return dot_general(a, b, batch, batch, {rank - 1}, {rank - 2}, shape, dtype);
        }
        if (implementation == "torch.nn.functional.linear" && operands.size() == 3 &&
            at[0] != nullptr && at[1] != nullptr) {
            const TensorInfo& x = *at[0];
            const TensorInfo& weight = *at[1];
            const std::int64_t last = static_cast<std::int64_t>(x.shape.size()) - 1;
            std::string out = dot_general(x, weight, {}, {}, {last}, {1}, shape, dtype);
            if (at[2] != nullptr) {
                const TensorInfo bias = *at[2];
                const std::string spread = broadcast(bias, {last}, shape);
                out = elementwise(
                    Elementwise::Add, {{out, shape, dtype}, {spread, shape, dtype}}, shape, dtype);
            }
            return out;
        }
        if (implementation_base == "torch.nn.functional.scaled_dot_product_attention" &&
            operands.size() == 5 && at[0] != nullptr && at[1] != nullptr && at[2] != nullptr &&
            at[3] != nullptr) {
            return attention(*at[0], *at[1], *at[2], *at[3], at[4], shape, dtype, fast);
        }
        return std::nullopt;
    }

    // `stablehlo.while` with the condition and body as regions: the outer
    // body text is saved while each region is written into its own buffer.
    bool supports_while() const override { return true; }

    std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) override {
        Loop loop;
        loop.initial = initial;
        loop.outer_body = std::move(body_);
        loop.outer_indent = indent_;
        loops_.push_back(std::move(loop));
        return open_region(initial);
    }

    std::vector<std::string> while_condition(const TensorInfo& predicate) override {
        body_ += indent_ + "\"stablehlo.return\"(" + predicate.name + ") : (" +
                 tensor_type(predicate) + ") -> ()\n";
        loops_.back().condition = close_region();
        return open_region(loops_.back().initial);
    }

    std::vector<std::string> end_while(const std::vector<TensorInfo>& next) override {
        std::string names;
        std::string types;
        for (std::size_t i = 0; i < next.size(); ++i) {
            names += (i == 0 ? "" : ", ") + next[i].name;
            types += (i == 0 ? "" : ", ") + tensor_type(next[i]);
        }
        body_ += indent_ + "\"stablehlo.return\"(" + names + ") : (" + types + ") -> ()\n";
        const std::string body_region = close_region();
        Loop loop = std::move(loops_.back());
        loops_.pop_back();
        body_ = std::move(loop.outer_body);
        indent_ = loop.outer_indent;
        std::string inits;
        std::string init_types;
        for (std::size_t i = 0; i < loop.initial.size(); ++i) {
            inits += (i == 0 ? "" : ", ") + loop.initial[i].name;
            init_types += (i == 0 ? "" : ", ") + tensor_type(loop.initial[i]);
        }
        const std::string result = fresh();
        const std::string count = std::to_string(next.size());
        body_ += indent_ + result + (next.size() == 1 ? "" : ":" + count) +
                 " = \"stablehlo.while\"(" + inits + ") (" + loop.condition + ", " + body_region +
                 ") : (" + init_types + ") -> (" + types + ")\n";
        std::vector<std::string> finals;
        finals.reserve(next.size());
        for (std::size_t i = 0; i < next.size(); ++i) {
            finals.push_back(next.size() == 1 ? result : result + "#" + std::to_string(i));
        }
        return finals;
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

    struct Loop {
        std::vector<TensorInfo> initial;
        std::string outer_body;
        std::string outer_indent;
        std::string condition;
    };

    // Starts a region whose block takes `arguments`; ops now write into it.
    std::vector<std::string> open_region(const std::vector<TensorInfo>& arguments) {
        std::vector<std::string> names;
        std::string header = "{\n" + indent_ + "  ^bb0(";
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            names.push_back(fresh());
            header += (i == 0 ? "" : ", ") + names.back() + ": " + tensor_type(arguments[i]);
        }
        header += "):\n";
        body_ = header;
        indent_ += "    ";
        return names;
    }

    // Ends the region opened last, returning its text.
    std::string close_region() {
        indent_.resize(indent_.size() - 4);
        std::string text = std::move(body_) + indent_ + "}";
        body_.clear();
        return text;
    }

    std::vector<Loop> loops_;

    std::string dot_general(const TensorInfo& lhs,
                            const TensorInfo& rhs,
                            const Dims& lhs_batch,
                            const Dims& rhs_batch,
                            const Dims& lhs_contract,
                            const Dims& rhs_contract,
                            const Dims& shape,
                            ScalarKind dtype) {
        const std::string numbers =
            "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = " +
            i64_list(lhs_batch) + ", rhs_batching_dimensions = " + i64_list(rhs_batch) +
            ", lhs_contracting_dimensions = " + i64_list(lhs_contract) +
            ", rhs_contracting_dimensions = " + i64_list(rhs_contract) + ">";
        return emit("dot_general", {lhs, rhs}, numbers, shape, dtype);
    }

    static std::string i64_list(const Dims& dims) {
        std::string out = "[";
        for (std::size_t i = 0; i < dims.size(); ++i) {
            out += (i == 0 ? "" : ", ") + std::to_string(dims[i]);
        }
        return out + "]";
    }

    // `std.nn.attention::attention` as two `dot_general`s around a softmax
    // in f32, exactly the canonical body's arithmetic; `fast` keeps the
    // input dtype throughout instead.
    std::string attention(const TensorInfo& query,
                          const TensorInfo& key,
                          const TensorInfo& value,
                          const TensorInfo& scale,
                          const TensorInfo* mask,
                          const Dims& shape,
                          ScalarKind dtype,
                          bool fast) {
        const ScalarKind f32 = fast ? dtype : ScalarKind::F32;
        const auto as_f32 = [&](const TensorInfo& t) -> TensorInfo {
            return t.dtype == f32 ? t : TensorInfo{convert(t, f32), t.shape, f32};
        };
        const TensorInfo q = as_f32(query);
        const TensorInfo k = as_f32(key);
        const TensorInfo v = as_f32(value);
        const Dims batch{0, 1};
        const Dims scores_shape{q.shape[0], q.shape[1], q.shape[2], k.shape[2]};
        TensorInfo scores{
            dot_general(q, k, batch, batch, {3}, {3}, scores_shape, f32), scores_shape, f32};
        const TensorInfo spread_scale{
            broadcast(as_f32(scale), {}, scores_shape), scores_shape, f32};
        scores = {elementwise(Elementwise::Mul, {scores, spread_scale}, scores_shape, f32),
                  scores_shape,
                  f32};
        if (mask != nullptr) {
            Literal lowest;
            lowest.kind = Literal::Kind::Real;
            lowest.real = -1e30;
            const TensorInfo fill{constant(lowest, f32), {}, f32};
            const TensorInfo spread_mask{
                broadcast(*mask, {2, 3}, scores_shape), scores_shape, ScalarKind::Bool};
            const TensorInfo spread_fill{broadcast(fill, {}, scores_shape), scores_shape, f32};
            scores = {
                select(spread_mask, scores, spread_fill, scores_shape, f32), scores_shape, f32};
        }
        const Dims rows{q.shape[0], q.shape[1], q.shape[2]};
        const TensorInfo peak{reduce(Reduction::Max, scores, {3}, rows), rows, f32};
        const TensorInfo spread_peak{broadcast(peak, {0, 1, 2}, scores_shape), scores_shape, f32};
        const TensorInfo shifted{
            elementwise(Elementwise::Sub, {scores, spread_peak}, scores_shape, f32),
            scores_shape,
            f32};
        const TensorInfo exps{
            elementwise(Elementwise::Exp, {shifted}, scores_shape, f32), scores_shape, f32};
        const TensorInfo total{reduce(Reduction::Sum, exps, {3}, rows), rows, f32};
        const TensorInfo spread_total{broadcast(total, {0, 1, 2}, scores_shape), scores_shape, f32};
        const TensorInfo weights{
            elementwise(Elementwise::Div, {exps, spread_total}, scores_shape, f32),
            scores_shape,
            f32};
        const std::string mixed = dot_general(weights, v, batch, batch, {3}, {2}, shape, f32);
        return dtype == f32 ? mixed : convert({mixed, shape, f32}, dtype);
    }

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
