#include "linnet/backend/onnx.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace linnet::backend {

namespace {

using sema::ScalarKind;

// ONNX element type names as the text format spells them, and the
// `TensorProto.DataType` numbers `Cast` takes.
const char* onnx_dtype(ScalarKind kind) {
    switch (kind) {
    case ScalarKind::Bool:
        return "bool";
    case ScalarKind::I8:
        return "int8";
    case ScalarKind::I16:
        return "int16";
    case ScalarKind::I32:
        return "int32";
    case ScalarKind::I64:
        return "int64";
    case ScalarKind::U8:
        return "uint8";
    case ScalarKind::U16:
        return "uint16";
    case ScalarKind::U32:
        return "uint32";
    case ScalarKind::U64:
        return "uint64";
    case ScalarKind::F16:
        return "float16";
    case ScalarKind::BF16:
        return "bfloat16";
    case ScalarKind::F32:
        return "float";
    case ScalarKind::F64:
        return "double";
    }
    return "float";
}

int onnx_dtype_code(ScalarKind kind) {
    switch (kind) {
    case ScalarKind::Bool:
        return 9;
    case ScalarKind::I8:
        return 3;
    case ScalarKind::I16:
        return 5;
    case ScalarKind::I32:
        return 6;
    case ScalarKind::I64:
        return 7;
    case ScalarKind::U8:
        return 2;
    case ScalarKind::U16:
        return 4;
    case ScalarKind::U32:
        return 12;
    case ScalarKind::U64:
        return 13;
    case ScalarKind::F16:
        return 10;
    case ScalarKind::BF16:
        return 16;
    case ScalarKind::F32:
        return 1;
    case ScalarKind::F64:
        return 11;
    }
    return 1;
}

std::string tensor_type(const Dims& shape, ScalarKind dtype) {
    std::string text = onnx_dtype(dtype);
    if (shape.empty()) {
        return text;
    }
    text += "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        text += (i == 0 ? "" : ",") + std::to_string(shape[i]);
    }
    return text + "]";
}

std::string float_text(double value) {
    if (value != value) {
        return "NaN";
    }
    if (value == std::numeric_limits<double>::infinity()) {
        return "Infinity";
    }
    if (value == -std::numeric_limits<double>::infinity()) {
        return "-Infinity";
    }
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    std::string text = buffer;
    if (text.find_first_of(".eE") == std::string::npos) {
        text += ".0";
    }
    return text;
}

std::string int_list(const Dims& values) {
    std::string text;
    for (std::size_t i = 0; i < values.size(); ++i) {
        text += (i == 0 ? "" : ", ") + std::to_string(values[i]);
    }
    return text;
}

// The ONNX text format (`onnx.parser`): one node per line, attributes in
// angle brackets, constants as `Constant` nodes.
class OnnxTarget : public GraphTarget {
public:
    std::string input(const std::string& name, const Dims& shape, ScalarKind dtype) override {
        std::string clean;
        for (const char c : name) {
            clean += std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' ? c : '_';
        }
        if (clean.empty() || std::isdigit(static_cast<unsigned char>(clean.front())) != 0) {
            clean = "input_" + clean;
        }
        inputs_.push_back(tensor_type(shape, dtype) + " " + clean);
        return clean;
    }

    std::string parameter(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        const std::string name = "param" + std::to_string(parameters_++);
        inputs_.push_back(tensor_type(shape, dtype) + " " + name);
        metadata_.push_back("\"linnet.path." + name + "\": \"" + path + "\"");
        return name;
    }

    std::string state(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        const std::string name = "state" + std::to_string(states_++);
        inputs_.push_back(tensor_type(shape, dtype) + " " + name);
        metadata_.push_back("\"linnet.state." + name + "\": \"" + path + "\"");
        return name;
    }

    std::string constant(const Literal& literal, ScalarKind dtype) override {
        const std::string name = constant_text(literal_text(literal, dtype), {}, dtype);
        literals_[name] = literal_text(literal, dtype);
        return name;
    }

    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           ScalarKind dtype) override {
        // Library operations with an ONNX operator of the same meaning;
        // `(input dtype)` variants skip the f32 casts.
        const std::string suffix = "(input dtype)";
        const bool fast = implementation.ends_with(suffix);
        const std::string implementation_base =
            fast ? implementation.substr(0, implementation.size() - suffix.size()) : implementation;
        const ScalarKind acc = fast ? dtype : ScalarKind::F32;
        std::vector<const TensorInfo*> at;
        at.reserve(operands.size());
        for (const std::optional<TensorInfo>& operand : operands) {
            at.push_back(operand.has_value() ? &*operand : nullptr);
        }
        const auto f32 = [&](const TensorInfo& t) -> TensorInfo {
            return t.dtype == acc ? t : TensorInfo{convert(t, acc), t.shape, acc};
        };
        const auto back = [&](const std::string& name, const Dims& s) -> std::string {
            return dtype == acc ? name : convert({name, s, acc}, dtype);
        };
        if (implementation_base == "torch.matmul" && operands.size() == 2 && at[0] != nullptr &&
            at[1] != nullptr) {
            return node("MatMul", {*at[0], *at[1]}, "", shape, dtype);
        }
        if (implementation_base == "torch.nn.functional.linear" && operands.size() == 3 &&
            at[0] != nullptr && at[1] != nullptr) {
            const TensorInfo& weight = *at[1];
            const Dims transposed{weight.shape[1], weight.shape[0]};
            const TensorInfo wt{transpose(weight, {1, 0}, transposed), transposed, weight.dtype};
            std::string out = node("MatMul", {*at[0], wt}, "", shape, dtype);
            if (at[2] != nullptr) {
                out = node("Add", {{out, shape, dtype}, *at[2]}, "", shape, dtype);
            }
            return out;
        }
        if (implementation_base == "torch.softmax" && operands.size() == 1 && at[0] != nullptr) {
            const TensorInfo x = f32(*at[0]);
            return back(node("Softmax", {x}, "axis = -1", shape, acc), shape);
        }
        if (implementation_base == "torch.nn.functional.layer_norm" && operands.size() == 4 &&
            at[0] != nullptr && at[1] != nullptr) {
            const TensorInfo x = f32(*at[0]);
            const TensorInfo weight = f32(*at[1]);
            const std::string epsilon = at[3] != nullptr ? literal_of(at[3]->name) : "";
            if (epsilon.empty()) {
                return std::nullopt;
            }
            std::vector<TensorInfo> inputs{x, weight};
            if (at[2] != nullptr) {
                inputs.push_back(f32(*at[2]));
            }
            const std::string normalized =
                node("LayerNormalization", inputs, "axis = -1, epsilon = " + epsilon, shape, acc);
            return back(normalized, shape);
        }
        if (implementation_base == "torch.nn.functional.gelu(tanh)" && operands.size() == 1 &&
            at[0] != nullptr) {
            return node("Gelu", {*at[0]}, "approximate = \"tanh\"", shape, dtype);
        }
        if (implementation_base == "torch.sigmoid" && operands.size() == 1 && at[0] != nullptr) {
            return node("Sigmoid", {*at[0]}, "", shape, dtype);
        }
        if (implementation_base == "torch.relu" && operands.size() == 1 && at[0] != nullptr) {
            return node("Relu", {*at[0]}, "", shape, dtype);
        }
        if (implementation_base == "torch.nn.functional.scaled_dot_product_attention" &&
            operands.size() == 5) {
            const TensorInfo* query = at[0];
            const TensorInfo* key = at[1];
            const TensorInfo* value = at[2];
            const TensorInfo* scale = at[3];
            const TensorInfo* mask = at[4];
            if (query == nullptr || key == nullptr || value == nullptr || scale == nullptr) {
                return std::nullopt;
            }
            // q·kᵀ in f32, scaled, masked, Softmax, ·v: the canonical arithmetic.
            const TensorInfo q = f32(*query);
            const TensorInfo k = f32(*key);
            const TensorInfo v = f32(*value);
            const Dims kt_shape{k.shape[0], k.shape[1], k.shape[3], k.shape[2]};
            const TensorInfo kt{transpose(k, {0, 1, 3, 2}, kt_shape), kt_shape, acc};
            const Dims scores_shape{q.shape[0], q.shape[1], q.shape[2], k.shape[2]};
            TensorInfo scores{node("MatMul", {q, kt}, "", scores_shape, acc), scores_shape, acc};
            scores = {node("Mul", {scores, f32(*scale)}, "", scores_shape, acc), scores_shape, acc};
            if (mask != nullptr) {
                Literal lowest;
                lowest.kind = Literal::Kind::Real;
                lowest.real = -1e30;
                const TensorInfo fill{constant(lowest, acc), {}, acc};
                scores = {
                    node("Where", {*mask, scores, fill}, "", scores_shape, acc), scores_shape, acc};
            }
            const TensorInfo weights{
                node("Softmax", {scores}, "axis = -1", scores_shape, acc), scores_shape, acc};
            return back(node("MatMul", {weights, v}, "", shape, acc), shape);
        }
        return std::nullopt;
    }

    std::string literal_of(const std::string& name) const {
        const auto found = literals_.find(name);
        return found == literals_.end() ? "" : found->second;
    }

    std::string elementwise(Elementwise kind,
                            const std::vector<TensorInfo>& operands,
                            const Dims& shape,
                            ScalarKind dtype) override {
        switch (kind) {
        case Elementwise::Add:
            return node("Add", operands, "", shape, dtype);
        case Elementwise::Sub:
            return node("Sub", operands, "", shape, dtype);
        case Elementwise::Mul:
            return node("Mul", operands, "", shape, dtype);
        case Elementwise::Div:
            return node("Div", operands, "", shape, dtype);
        case Elementwise::Rem:
            return node("Mod", operands, sema::is_float(dtype) ? "fmod = 1" : "", shape, dtype);
        case Elementwise::Min:
            return node("Min", operands, "", shape, dtype);
        case Elementwise::Max:
            return node("Max", operands, "", shape, dtype);
        case Elementwise::BitAnd:
            return node(
                dtype == ScalarKind::Bool ? "And" : "BitwiseAnd", operands, "", shape, dtype);
        case Elementwise::BitOr:
            return node(dtype == ScalarKind::Bool ? "Or" : "BitwiseOr", operands, "", shape, dtype);
        case Elementwise::BitXor:
            return node(
                dtype == ScalarKind::Bool ? "Xor" : "BitwiseXor", operands, "", shape, dtype);
        case Elementwise::Shl:
        case Elementwise::Shr: {
            // `BitShift` takes unsigned operands: signed values go through
            // the unsigned dtype of the same width, so a right shift is
            // logical (it differs from the arithmetic one for negatives).
            const ScalarKind wide = dtype == ScalarKind::I8    ? ScalarKind::U8
                                    : dtype == ScalarKind::I16 ? ScalarKind::U16
                                    : dtype == ScalarKind::I32 ? ScalarKind::U32
                                    : dtype == ScalarKind::I64 ? ScalarKind::U64
                                                               : dtype;
            std::vector<TensorInfo> shifted;
            shifted.reserve(operands.size());
            for (const TensorInfo& operand : operands) {
                shifted.push_back(wide == dtype
                                      ? operand
                                      : TensorInfo{convert(operand, wide), operand.shape, wide});
            }
            const std::string out =
                node("BitShift",
                     shifted,
                     kind == Elementwise::Shl ? "direction = \"LEFT\"" : "direction = \"RIGHT\"",
                     shape,
                     wide);
            return wide == dtype ? out : convert({out, shape, wide}, dtype);
        }
        case Elementwise::And:
            return node("And", operands, "", shape, dtype);
        case Elementwise::Or:
            return node("Or", operands, "", shape, dtype);
        case Elementwise::Not:
            return node("Not", operands, "", shape, dtype);
        case Elementwise::Neg:
            return node("Neg", operands, "", shape, dtype);
        case Elementwise::Exp:
            return node("Exp", operands, "", shape, dtype);
        case Elementwise::Log:
            return node("Log", operands, "", shape, dtype);
        case Elementwise::Sqrt:
            return node("Sqrt", operands, "", shape, dtype);
        case Elementwise::Rsqrt: {
            // ONNX has no Rsqrt.
            const TensorInfo root{node("Sqrt", operands, "", shape, dtype), shape, dtype};
            return node("Reciprocal", {root}, "", shape, dtype);
        }
        case Elementwise::Sin:
            return node("Sin", operands, "", shape, dtype);
        case Elementwise::Cos:
            return node("Cos", operands, "", shape, dtype);
        case Elementwise::Tanh:
            return node("Tanh", operands, "", shape, dtype);
        case Elementwise::Abs:
            return node("Abs", operands, "", shape, dtype);
        }
        return node("Add", operands, "", shape, dtype);
    }

    std::string compare(ir::CompareKind kind,
                        const TensorInfo& a,
                        const TensorInfo& b,
                        const Dims& shape) override {
        switch (kind) {
        case ir::CompareKind::Eq:
            return node("Equal", {a, b}, "", shape, ScalarKind::Bool);
        case ir::CompareKind::Ne: {
            const TensorInfo equal{
                node("Equal", {a, b}, "", shape, ScalarKind::Bool), shape, ScalarKind::Bool};
            return node("Not", {equal}, "", shape, ScalarKind::Bool);
        }
        case ir::CompareKind::Lt:
            return node("Less", {a, b}, "", shape, ScalarKind::Bool);
        case ir::CompareKind::Le:
            return node("LessOrEqual", {a, b}, "", shape, ScalarKind::Bool);
        case ir::CompareKind::Gt:
            return node("Greater", {a, b}, "", shape, ScalarKind::Bool);
        case ir::CompareKind::Ge:
            return node("GreaterOrEqual", {a, b}, "", shape, ScalarKind::Bool);
        }
        return node("Equal", {a, b}, "", shape, ScalarKind::Bool);
    }

    std::string select(const TensorInfo& condition,
                       const TensorInfo& on_true,
                       const TensorInfo& on_false,
                       const Dims& shape,
                       ScalarKind dtype) override {
        return node("Where", {condition, on_true, on_false}, "", shape, dtype);
    }

    std::string convert(const TensorInfo& value, ScalarKind dtype) override {
        return node(
            "Cast", {value}, "to = " + std::to_string(onnx_dtype_code(dtype)), value.shape, dtype);
    }

    std::string reshape(const TensorInfo& value, const Dims& shape) override {
        const TensorInfo target = int64_vector(shape.empty() ? Dims{} : shape);
        return node("Reshape", {value, target}, "", shape, value.dtype);
    }

    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override {
        return node(
            "Transpose", {value}, "perm = [" + int_list(permutation) + "]", shape, value.dtype);
    }

    std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) override {
        // A reshape only inserts axes, so the operand's axes are first put
        // in the order they take in the result; then the rest expands.
        Dims order(dims.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = static_cast<std::int64_t>(i);
        }
        std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
            return dims[static_cast<std::size_t>(a)] < dims[static_cast<std::size_t>(b)];
        });
        TensorInfo source = value;
        Dims sorted_dims = dims;
        bool is_identity = true;
        for (std::size_t i = 0; i < order.size(); ++i) {
            is_identity = is_identity && order[i] == static_cast<std::int64_t>(i);
        }
        if (!is_identity) {
            Dims permuted;
            for (const std::int64_t axis : order) {
                permuted.push_back(value.shape[static_cast<std::size_t>(axis)]);
                sorted_dims[permuted.size() - 1] = dims[static_cast<std::size_t>(axis)];
            }
            source = {transpose(value, order, permuted), permuted, value.dtype};
        }
        Dims placed(shape.size(), 1);
        for (std::size_t i = 0; i < sorted_dims.size(); ++i) {
            placed[static_cast<std::size_t>(sorted_dims[i])] = source.shape[i];
        }
        if (placed != source.shape) {
            source = {reshape(source, placed), placed, source.dtype};
        }
        if (placed == shape) {
            return source.name;
        }
        const TensorInfo target = int64_vector(shape);
        return node("Expand", {source, target}, "", shape, value.dtype);
    }

    std::string slice(const TensorInfo& value,
                      const Dims& starts,
                      const Dims& limits,
                      const Dims& strides,
                      const Dims& shape) override {
        Dims axes;
        for (std::size_t i = 0; i < starts.size(); ++i) {
            axes.push_back(static_cast<std::int64_t>(i));
        }
        const TensorInfo start = int64_vector(starts);
        const TensorInfo end = int64_vector(limits);
        const TensorInfo axis = int64_vector(axes);
        const TensorInfo step = int64_vector(strides);
        return node("Slice", {value, start, end, axis, step}, "", shape, value.dtype);
    }

    std::string
    concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) override {
        return node("Concat", parts, "axis = " + std::to_string(axis), shape, parts.front().dtype);
    }

    std::string iota(std::int64_t length) override {
        const Literal zero;
        Literal limit;
        limit.integer = length;
        Literal one;
        one.integer = 1;
        const TensorInfo start{constant(zero, ScalarKind::I64), {}, ScalarKind::I64};
        const TensorInfo stop{constant(limit, ScalarKind::I64), {}, ScalarKind::I64};
        const TensorInfo delta{constant(one, ScalarKind::I64), {}, ScalarKind::I64};
        return node("Range", {start, stop, delta}, "", {length}, ScalarKind::I64);
    }

    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override {
        return node("GatherND", {source, indices}, "batch_dims = 0", shape, source.dtype);
    }

    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        const TensorInfo axes = int64_vector(dims);
        const bool is_logical = kind == Reduction::Any || kind == Reduction::All;
        TensorInfo source = body;
        if (is_logical) {
            // Boolean reductions go through integers: ONNX reduces numbers.
            source = {convert(body, ScalarKind::I32), body.shape, ScalarKind::I32};
        }
        const char* op = kind == Reduction::Sum    ? "ReduceSum"
                         : kind == Reduction::Prod ? "ReduceProd"
                         : kind == Reduction::Max  ? "ReduceMax"
                         : kind == Reduction::Min  ? "ReduceMin"
                         : kind == Reduction::Any  ? "ReduceMax"
                                                   : "ReduceMin";
        std::string reduced = node(op, {source, axes}, "keepdims = 0", shape, source.dtype);
        if (is_logical) {
            return convert({reduced, shape, ScalarKind::I32}, ScalarKind::Bool);
        }
        return reduced;
    }

    // ONNX `Loop`: the condition is checked before each iteration from a
    // value the previous one produced, so the predicate is evaluated once
    // before the loop and again at the end of each body.
    bool supports_while() const override { return true; }
    bool while_needs_initial_condition() const override { return true; }
    bool while_needs_trailing_condition() const override { return true; }

    void while_initial_condition(const TensorInfo& predicate) override {
        pending_condition_ = predicate.name;
    }

    std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) override {
        Loop loop;
        loop.initial = initial;
        loop.initial_condition = pending_condition_;
        loop.outer_body = std::move(body_);
        loop.outer_indent = indent_;
        loop.graph = "loop_body" + std::to_string(loops_.size() + next_);
        body_.clear();
        indent_ += "  ";
        std::vector<std::string> names;
        names.reserve(initial.size());
        for (std::size_t i = 0; i < initial.size(); ++i) {
            names.push_back(loop.graph + "_in" + std::to_string(i));
        }
        loop.inputs = names;
        loops_.push_back(std::move(loop));
        return names;
    }

    std::vector<std::string> while_condition(const TensorInfo& predicate) override {
        (void)predicate; // the body graph checks the condition it computes itself
        return loops_.back().inputs;
    }

    void while_trailing_condition(const TensorInfo& predicate) override {
        loops_.back().trailing_condition = predicate.name;
    }

    std::vector<std::string> end_while(const std::vector<TensorInfo>& next) override {
        Loop loop = std::move(loops_.back());
        loops_.pop_back();
        std::string body_text = std::move(body_);
        // Outputs are named copies of the final values.
        std::string outputs;
        outputs += "bool " + loop.graph + "_cond_out";
        body_text +=
            indent_ + loop.graph + "_cond_out = Identity(" + loop.trailing_condition + ")\n";
        for (std::size_t i = 0; i < next.size(); ++i) {
            const std::string name = loop.graph + "_out" + std::to_string(i);
            body_text += indent_ + name + " = Identity(" + next[i].name + ")\n";
            outputs += ", " + tensor_type(next[i].shape, next[i].dtype) + " " + name;
        }
        indent_ = loop.outer_indent;
        body_ = std::move(loop.outer_body);
        std::string inputs = "int64 " + loop.graph + "_iter, bool " + loop.graph + "_cond_in";
        for (std::size_t i = 0; i < loop.initial.size(); ++i) {
            inputs += ", " + tensor_type(loop.initial[i].shape, loop.initial[i].dtype) + " " +
                      loop.inputs[i];
        }
        std::vector<std::string> finals;
        finals.reserve(next.size());
        std::string results;
        std::string inits;
        for (std::size_t i = 0; i < next.size(); ++i) {
            finals.push_back(fresh());
            results += (i == 0 ? "" : ", ") + finals.back();
            inits += ", " + loop.initial[i].name;
        }
        body_ += indent_ + results + " = Loop <body = " + loop.graph + " (" + inputs + ") => (" +
                 outputs + ") {\n" + body_text + indent_ + "}> (\"\", " + loop.initial_condition +
                 inits + ")\n";
        return finals;
    }

    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        // Outputs need names of their own: `Identity` gives the results
        // stable ones, and `next_state<N>` names each assigned state member,
        // mapped to its path by `linnet.next_state.<name>`.
        std::vector<std::string> outputs;
        for (std::size_t i = 0; i < results.size(); ++i) {
            const std::string name = "output" + std::to_string(i);
            body_ += "    " + name + " = Identity(" + results[i].name + ")\n";
            outputs.push_back(tensor_type(results[i].shape, results[i].dtype) + " " + name);
        }
        for (std::size_t i = 0; i < states.size(); ++i) {
            const std::string name = "next_state" + std::to_string(i);
            body_ += "    " + name + " = Identity(" + states[i].second.name + ")\n";
            outputs.push_back(tensor_type(states[i].second.shape, states[i].second.dtype) + " " +
                              name);
            metadata_.push_back("\"linnet.next_state." + name + "\": \"" + states[i].first + "\"");
        }
        std::string out = "<ir_version: 10, opset_import: [\"\" : 20], producer_name: \"linnet\", "
                          "doc_string: \"" +
                          block_name + "." + entry_name + " from module " + module_path + "\"";
        if (!metadata_.empty()) {
            out += ", metadata_props: [";
            for (std::size_t i = 0; i < metadata_.size(); ++i) {
                out += (i == 0 ? "" : ", ") + metadata_[i];
            }
            out += "]";
        }
        out += ">\nmain (";
        for (std::size_t i = 0; i < inputs_.size(); ++i) {
            out += (i == 0 ? "" : ", ") + inputs_[i];
        }
        out += ") => (";
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            out += (i == 0 ? "" : ", ") + outputs[i];
        }
        out += ") {\n";
        out += body_;
        out += "}\n";
        return out;
    }

private:
    std::string fresh() { return "v" + std::to_string(next_++); }

    static std::string literal_text(const Literal& literal, ScalarKind dtype) {
        const bool is_real = sema::is_float(dtype);
        switch (literal.kind) {
        case Literal::Kind::Integer:
            return is_real ? float_text(static_cast<double>(literal.integer))
                           : std::to_string(literal.integer);
        case Literal::Kind::Real:
            return is_real ? float_text(literal.real)
                           : std::to_string(static_cast<std::int64_t>(literal.real));
        case Literal::Kind::Boolean:
            return literal.integer != 0 ? "1" : "0";
        case Literal::Kind::Lowest:
            if (is_real) {
                return "-Infinity";
            }
            return dtype == ScalarKind::Bool ? "0" : std::to_string(lowest_integer(dtype));
        case Literal::Kind::Highest:
            if (is_real) {
                return "Infinity";
            }
            return dtype == ScalarKind::Bool ? "1" : std::to_string(highest_integer(dtype));
        }
        return "0";
    }

    static std::int64_t lowest_integer(ScalarKind dtype) {
        switch (dtype) {
        case ScalarKind::I8:
            return std::numeric_limits<std::int8_t>::min();
        case ScalarKind::I16:
            return std::numeric_limits<std::int16_t>::min();
        case ScalarKind::I32:
            return std::numeric_limits<std::int32_t>::min();
        case ScalarKind::I64:
            return std::numeric_limits<std::int64_t>::min();
        default:
            return 0;
        }
    }

    static std::int64_t highest_integer(ScalarKind dtype) {
        switch (dtype) {
        case ScalarKind::I8:
            return std::numeric_limits<std::int8_t>::max();
        case ScalarKind::I16:
            return std::numeric_limits<std::int16_t>::max();
        case ScalarKind::I32:
            return std::numeric_limits<std::int32_t>::max();
        case ScalarKind::U8:
            return std::numeric_limits<std::uint8_t>::max();
        case ScalarKind::U16:
            return std::numeric_limits<std::uint16_t>::max();
        case ScalarKind::U32:
            return std::numeric_limits<std::uint32_t>::max();
        default:
            return std::numeric_limits<std::int64_t>::max();
        }
    }

    // `Constant <value = float[2] {1.0, 2.0}> ()`; a 0-d tensor omits the
    // shape.
    std::string constant_text(const std::string& values, const Dims& shape, ScalarKind dtype) {
        const std::string name = fresh();
        body_ += indent_ + name + " = Constant <value = " + tensor_type(shape, dtype) + " {" +
                 values + "}> ()\n";
        return name;
    }

    TensorInfo int64_vector(const Dims& values) {
        const Dims shape{static_cast<std::int64_t>(values.size())};
        return {constant_text(int_list(values), shape, ScalarKind::I64), shape, ScalarKind::I64};
    }

    std::string node(const std::string& op,
                     const std::vector<TensorInfo>& operands,
                     const std::string& attributes,
                     const Dims& shape,
                     ScalarKind dtype) {
        (void)shape;
        (void)dtype;
        const std::string name = fresh();
        std::string line = indent_ + name + " = " + op;
        if (!attributes.empty()) {
            line += " <" + attributes + ">";
        }
        line += " (";
        for (std::size_t i = 0; i < operands.size(); ++i) {
            line += (i == 0 ? "" : ", ") + operands[i].name;
        }
        body_ += line + ")\n";
        return name;
    }

    struct Loop {
        std::vector<TensorInfo> initial;
        std::vector<std::string> inputs;
        std::string initial_condition;
        std::string trailing_condition;
        std::string outer_body;
        std::string outer_indent;
        std::string graph;
    };

    std::vector<Loop> loops_;
    std::string pending_condition_;
    std::vector<std::string> inputs_;
    std::vector<std::string> metadata_;
    std::map<std::string, std::string> literals_; // constant name -> literal text
    std::string body_;
    std::string indent_ = "  ";
    std::size_t next_ = 0;
    std::size_t parameters_ = 0;
    std::size_t states_ = 0;
};

} // namespace

std::expected<std::string, std::string> export_onnx(ir::Module& module,
                                                    const OnnxOptions& options) {
    OnnxTarget target;
    return export_graph(module, options, target);
}

} // namespace linnet::backend
