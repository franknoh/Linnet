#include "linnet/backend/torch_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

std::string torch_dtype(ScalarKind dtype) {
    switch (dtype) {
    case ScalarKind::Bool:
        return "torch.bool";
    case ScalarKind::I8:
        return "torch.int8";
    case ScalarKind::I16:
        return "torch.int16";
    case ScalarKind::I32:
        return "torch.int32";
    case ScalarKind::I64:
        return "torch.int64";
    case ScalarKind::U8:
        return "torch.uint8";
    case ScalarKind::U16:
        return "torch.uint16";
    case ScalarKind::U32:
        return "torch.uint32";
    case ScalarKind::U64:
        return "torch.uint64";
    case ScalarKind::F16:
        return "torch.float16";
    case ScalarKind::BF16:
        return "torch.bfloat16";
    case ScalarKind::F32:
        return "torch.float32";
    case ScalarKind::F64:
        return "torch.float64";
    }
    return "torch.float32";
}

bool is_real(ScalarKind dtype) {
    return dtype == ScalarKind::F16 || dtype == ScalarKind::BF16 || dtype == ScalarKind::F32 ||
           dtype == ScalarKind::F64;
}

std::string dims_text(const Dims& dims) {
    std::string out = "(";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        out += (i == 0 ? "" : ", ") + std::to_string(dims[i]);
    }
    return out + (dims.size() == 1 ? ",)" : ")");
}

std::string real_text(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    std::string text = buffer;
    if (text.find_first_of(".einEIN") == std::string::npos) {
        text += ".0";
    }
    return text;
}

// The generated module: straight-line PyTorch over static shapes.
class TorchTarget : public GraphTarget {
public:
    std::string input(const std::string& name, const Dims& shape, ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        std::string clean;
        for (const char c : name) {
            clean += std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' ? c : '_';
        }
        const std::string argument = "in_" + clean;
        arguments_.push_back(argument);
        return argument;
    }

    std::string parameter(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        const std::string argument = "p" + std::to_string(parameters_.size());
        parameters_.push_back(path);
        arguments_.push_back(argument);
        return argument;
    }

    std::string state(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        const std::string argument = "s" + std::to_string(states_.size());
        states_.push_back(path);
        arguments_.push_back(argument);
        return argument;
    }

    std::string constant(const Literal& literal, ScalarKind dtype) override {
        const std::string text = literal_text(literal, dtype);
        const std::string name =
            define("torch.tensor(" + text + ", dtype=" + torch_dtype(dtype) + ", device=_device)");
        literals_[name] = text;
        if (literal.kind == Literal::Kind::Integer || literal.kind == Literal::Kind::Real) {
            values_[name] = literal.kind == Literal::Kind::Real
                                ? literal.real
                                : static_cast<double>(literal.integer);
        }
        return name;
    }

    // Scalar arithmetic on constants is folded to a Python literal as well,
    // so a computed `rsqrt(cast<f32>(D))` reaches a kernel as `scale=...`
    // rather than as a tensor `torch.compile` has to read back.
    void fold(const std::string& name,
              Elementwise kind,
              const std::vector<TensorInfo>& operands,
              ScalarKind dtype) {
        if (dtype == ScalarKind::Bool || operands.empty() || !operands[0].shape.empty()) {
            return;
        }
        std::vector<double> values;
        for (const TensorInfo& operand : operands) {
            const auto found = values_.find(operand.name);
            if (found == values_.end()) {
                return;
            }
            values.push_back(found->second);
        }
        double result = 0.0;
        const double a = values[0];
        const double b = values.size() > 1 ? values[1] : 0.0;
        switch (kind) {
        case Elementwise::Add:
            result = a + b;
            break;
        case Elementwise::Sub:
            result = a - b;
            break;
        case Elementwise::Mul:
            result = a * b;
            break;
        case Elementwise::Div:
            if (b == 0.0) {
                return;
            }
            result = is_real(dtype) ? a / b : std::trunc(a / b);
            break;
        case Elementwise::Neg:
            result = -a;
            break;
        case Elementwise::Sqrt:
            result = std::sqrt(a);
            break;
        case Elementwise::Rsqrt:
            result = 1.0 / std::sqrt(a);
            break;
        case Elementwise::Exp:
            result = std::exp(a);
            break;
        case Elementwise::Log:
            result = std::log(a);
            break;
        default:
            return;
        }
        if (!is_real(dtype)) {
            if (kind != Elementwise::Add && kind != Elementwise::Sub && kind != Elementwise::Mul &&
                kind != Elementwise::Div && kind != Elementwise::Neg) {
                return;
            }
            values_[name] = result;
            literals_[name] = std::to_string(static_cast<long long>(result));
            return;
        }
        if (dtype == ScalarKind::F32) {
            result = static_cast<double>(static_cast<float>(result));
        }
        values_[name] = result;
        literals_[name] = real_text(result);
    }

    std::string elementwise(Elementwise kind,
                            const std::vector<TensorInfo>& operands,
                            const Dims& shape,
                            ScalarKind dtype) override {
        (void)shape;
        const std::string name = spell(kind, operands);
        fold(name, kind, operands, dtype);
        return name;
    }

    std::string spell(Elementwise kind, const std::vector<TensorInfo>& operands) {
        const std::string& a = operands[0].name;
        const std::string b = operands.size() > 1 ? operands[1].name : "";
        switch (kind) {
        case Elementwise::Add:
            return define(a + " + " + b);
        case Elementwise::Sub:
            return define(a + " - " + b);
        case Elementwise::Mul:
            return define(a + " * " + b);
        case Elementwise::Div:
            return define(is_real(operands[0].dtype)
                              ? a + " / " + b
                              : "torch.div(" + a + ", " + b + ", rounding_mode=\"trunc\")");
        case Elementwise::Rem:
            return define("torch.fmod(" + a + ", " + b + ")");
        case Elementwise::Min:
            return define("torch.minimum(" + a + ", " + b + ")");
        case Elementwise::Max:
            return define("torch.maximum(" + a + ", " + b + ")");
        case Elementwise::And:
        case Elementwise::BitAnd:
            return define(a + " & " + b);
        case Elementwise::Or:
        case Elementwise::BitOr:
            return define(a + " | " + b);
        case Elementwise::BitXor:
            return define(a + " ^ " + b);
        case Elementwise::Shl:
            return define(a + " << " + b);
        case Elementwise::Shr:
            return define(a + " >> " + b);
        case Elementwise::Not:
            return define("~" + a);
        case Elementwise::Neg:
            return define("-" + a);
        case Elementwise::Exp:
            return define("torch.exp(" + a + ")");
        case Elementwise::Log:
            return define("torch.log(" + a + ")");
        case Elementwise::Sqrt:
            return define("torch.sqrt(" + a + ")");
        case Elementwise::Rsqrt:
            return define("torch.rsqrt(" + a + ")");
        case Elementwise::Sin:
            return define("torch.sin(" + a + ")");
        case Elementwise::Cos:
            return define("torch.cos(" + a + ")");
        case Elementwise::Tanh:
            return define("torch.tanh(" + a + ")");
        case Elementwise::Abs:
            return define("torch.abs(" + a + ")");
        }
        return define(a);
    }

    std::string compare(ir::CompareKind kind,
                        const TensorInfo& a,
                        const TensorInfo& b,
                        const Dims& shape) override {
        (void)shape;
        const char* op = kind == ir::CompareKind::Eq   ? "eq"
                         : kind == ir::CompareKind::Ne ? "ne"
                         : kind == ir::CompareKind::Lt ? "lt"
                         : kind == ir::CompareKind::Le ? "le"
                         : kind == ir::CompareKind::Gt ? "gt"
                                                       : "ge";
        return define(std::string("torch.") + op + "(" + a.name + ", " + b.name + ")");
    }

    std::string select(const TensorInfo& condition,
                       const TensorInfo& on_true,
                       const TensorInfo& on_false,
                       const Dims& shape,
                       ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        return define("torch.where(" + condition.name + ", " + on_true.name + ", " + on_false.name +
                      ")");
    }

    std::string convert(const TensorInfo& value, ScalarKind dtype) override {
        const std::string name = define(value.name + ".to(" + torch_dtype(dtype) + ")");
        const auto found = values_.find(value.name);
        if (found != values_.end() && is_real(dtype)) {
            fold(name,
                 Elementwise::Add,
                 {{value.name, {}, dtype}, {zero_name(dtype), {}, dtype}},
                 dtype);
        }
        return name;
    }

    // A zero literal to fold conversions through (`x + 0` in the target dtype).
    std::string zero_name(ScalarKind dtype) {
        auto& cached = zeros_[dtype];
        if (cached.empty()) {
            Literal zero;
            zero.kind = Literal::Kind::Real;
            cached = constant(zero, dtype);
        }
        return cached;
    }

    std::string reshape(const TensorInfo& value, const Dims& shape) override {
        return define(value.name + ".reshape(" + dims_text(shape) + ")");
    }

    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override {
        (void)shape;
        return define(value.name + ".permute(" + dims_text(permutation) + ")");
    }

    std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) override {
        // Axes first in the order they take in the result, then size-one
        // axes inserted, then the expansion.
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
        return define(source.name + ".expand(" + dims_text(shape) + ")");
    }

    std::string slice(const TensorInfo& value,
                      const Dims& starts,
                      const Dims& limits,
                      const Dims& strides,
                      const Dims& shape) override {
        (void)shape;
        std::string index;
        for (std::size_t i = 0; i < starts.size(); ++i) {
            index += (i == 0 ? "" : ", ") + std::to_string(starts[i]) + ":" +
                     std::to_string(limits[i]) + ":" + std::to_string(strides[i]);
        }
        return define(value.name + "[" + index + "]");
    }

    std::string
    concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) override {
        (void)shape;
        std::string list;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            list += (i == 0 ? "" : ", ") + parts[i].name;
        }
        return define("torch.cat([" + list + "], dim=" + std::to_string(axis) + ")");
    }

    std::string iota(std::int64_t length) override {
        return define("torch.arange(" + std::to_string(length) +
                      ", dtype=torch.int64, device=_device)");
    }

    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override {
        (void)shape;
        // `indices[..., k]` selects along axis k of the source: advanced
        // indexing with one index tensor per source axis.
        std::string index;
        for (std::size_t k = 0; k < source.shape.size(); ++k) {
            index += (k == 0 ? "" : ", ") + indices.name + "[..., " + std::to_string(k) + "]";
        }
        return define(source.name + "[" + index + "]");
    }

    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        (void)shape;
        const std::string axes = dims_text(dims);
        switch (kind) {
        case Reduction::Sum:
            return define(body.name + ".sum(dim=" + axes + ")");
        case Reduction::Prod:
            // `prod` takes one axis: the trailing reduced axes are flattened.
            return define(body.name + ".flatten(" +
                          std::to_string(body.shape.size() - dims.size()) + ").prod(dim=-1)");
        case Reduction::Max:
            return define(body.name + ".amax(dim=" + axes + ")");
        case Reduction::Min:
            return define(body.name + ".amin(dim=" + axes + ")");
        case Reduction::Any:
            return define(body.name + ".any(dim=" + axes + ")");
        case Reduction::All:
            return define(body.name + ".all(dim=" + axes + ")");
        }
        return body.name;
    }

    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        const auto name = [&](std::size_t i) -> std::string {
            return i < operands.size() && operands[i] ? operands[i]->name : "None";
        };
        const auto scalar = [&](std::size_t i) -> std::string {
            // A constant scalar is spelled as a literal; anything else is
            // read from its tensor.
            if (i < operands.size() && operands[i]) {
                const auto found = literals_.find(operands[i]->name);
                if (found != literals_.end()) {
                    return found->second;
                }
                return "float(" + operands[i]->name + ")";
            }
            return "None";
        };
        if (implementation == "torch.matmul" && operands.size() == 2) {
            return define("torch.matmul(" + name(0) + ", " + name(1) + ")");
        }
        if (implementation == "torch.nn.functional.linear" && operands.size() == 3) {
            return define("F.linear(" + name(0) + ", " + name(1) + ", " + name(2) + ")");
        }
        if (implementation == "torch.softmax" && operands.size() == 1) {
            return define("torch.softmax(" + name(0) + ".float(), dim=-1).to(" + name(0) +
                          ".dtype)");
        }
        if (implementation == "torch.relu" && operands.size() == 1) {
            return define("torch.relu(" + name(0) + ")");
        }
        if (implementation == "torch.sigmoid" && operands.size() == 1) {
            return define("torch.sigmoid(" + name(0) + ")");
        }
        if (implementation == "torch.nn.functional.silu" && operands.size() == 1) {
            return define("F.silu(" + name(0) + ")");
        }
        if (implementation == "torch.nn.functional.gelu(tanh)" && operands.size() == 1) {
            return define("F.gelu(" + name(0) + ", approximate=\"tanh\")");
        }
        // The normalized width is the operand's last axis.
        const auto width_of = [&](std::size_t i) -> std::string {
            const std::optional<TensorInfo>& operand = operands[i];
            return operand.has_value() && !operand->shape.empty()
                       ? std::to_string(operand->shape.back())
                       : "1";
        };
        if (implementation == "torch.rms_norm" && operands.size() == 3 && operands[0]) {
            const std::string width = width_of(0);
            return define("torch.rms_norm(" + name(0) + ".float(), [" + width +
                          "], eps=" + scalar(2) + ").to(" + name(0) + ".dtype) * " + name(1));
        }
        if (implementation == "torch.nn.functional.layer_norm" && operands.size() == 4 &&
            operands[0]) {
            const std::string width = width_of(0);
            const std::string scaled =
                define("F.layer_norm(" + name(0) + ".float(), [" + width + "], eps=" + scalar(3) +
                       ").to(" + name(0) + ".dtype) * " + name(1));
            return operands[2] ? define(scaled + " + " + name(2)) : scaled;
        }
        if (implementation == "torch.nn.functional.scaled_dot_product_attention" &&
            operands.size() == 5) {
            return define("F.scaled_dot_product_attention(" + name(0) + ".float(), " + name(1) +
                          ".float(), " + name(2) + ".float(), attn_mask=" + name(4) +
                          ", scale=" + scalar(3) + ").to(" + name(0) + ".dtype)");
        }
        return std::nullopt;
    }

    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        std::string out = "# " + block_name + "." + entry_name + " from module " + module_path +
                          ", generated by `linnet torch` for one shape\n"
                          "# binding. `main` takes the entry's inputs, then the parameters in\n"
                          "# PARAMETERS order, then the states in STATES order; it returns the\n"
                          "# entry's RESULTS results followed by the states in NEXT_STATES\n"
                          "# order.\n"
                          "import torch\n"
                          "import torch.nn.functional as F\n\n";
        out += "PARAMETERS = " + string_list(parameters_) + "\n";
        out += "STATES = " + string_list(states_) + "\n";
        std::vector<std::string> next_states;
        next_states.reserve(states.size());
        for (const auto& [path, value] : states) {
            next_states.push_back(path);
        }
        out += "NEXT_STATES = " + string_list(next_states) + "\n";
        out += "RESULTS = " + std::to_string(results.size()) + "\n\n\n";
        out += "def main(";
        for (std::size_t i = 0; i < arguments_.size(); ++i) {
            out += (i == 0 ? "" : ", ") + arguments_[i];
        }
        out += "):\n";
        out += "    _device = " +
               (arguments_.empty() ? std::string("torch.device(\"cpu\")")
                                   : arguments_.front() + ".device") +
               "\n";
        out += body_;
        out += "    return (";
        std::vector<std::string> outputs;
        outputs.reserve(results.size() + states.size());
        for (const TensorInfo& result : results) {
            outputs.push_back(result.name);
        }
        for (const auto& [path, value] : states) {
            outputs.push_back(value.name);
        }
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            out += (i == 0 ? "" : ", ") + outputs[i];
        }
        out += outputs.size() == 1 ? ",)\n" : ")\n";
        return out;
    }

    // A Python `while True:` over loop variables, broken out of when the
    // condition fails.
    bool supports_while() const override { return true; }

    std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) override {
        std::vector<std::string> names;
        const std::string prefix = "w" + std::to_string(loops_++) + "_";
        for (std::size_t i = 0; i < initial.size(); ++i) {
            names.push_back(prefix + std::to_string(i));
            body_ += indent_ + names.back() + " = " + initial[i].name + "\n";
        }
        body_ += indent_ + "while True:\n";
        indent_ += "    ";
        loop_names_.push_back(names);
        return names;
    }

    std::vector<std::string> while_condition(const TensorInfo& predicate) override {
        body_ += indent_ + "if not bool(" + predicate.name + "):\n" + indent_ + "    break\n";
        return loop_names_.back();
    }

    std::vector<std::string> end_while(const std::vector<TensorInfo>& next) override {
        const std::vector<std::string> names = std::move(loop_names_.back());
        loop_names_.pop_back();
        std::string targets;
        std::string values;
        for (std::size_t i = 0; i < next.size(); ++i) {
            targets += (i == 0 ? "" : ", ") + names[i];
            values += (i == 0 ? "" : ", ") + next[i].name;
        }
        body_ += indent_ + targets + (next.size() == 1 ? "," : "") + " = " + values +
                 (next.size() == 1 ? "," : "") + "\n";
        indent_.resize(indent_.size() - 4);
        return names;
    }

private:
    std::string define(const std::string& expression) {
        const std::string name = "v" + std::to_string(next_++);
        body_ += indent_ + name + " = " + expression + "\n";
        return name;
    }

    static std::string string_list(const std::vector<std::string>& items) {
        std::string out = "[";
        for (std::size_t i = 0; i < items.size(); ++i) {
            out += (i == 0 ? "\"" : ", \"") + items[i] + "\"";
        }
        return out + "]";
    }

    static std::string literal_text(const Literal& literal, ScalarKind dtype) {
        switch (literal.kind) {
        case Literal::Kind::Integer:
            return is_real(dtype) ? real_text(static_cast<double>(literal.integer))
                                  : std::to_string(literal.integer);
        case Literal::Kind::Real:
            return real_text(literal.real);
        case Literal::Kind::Boolean:
            return literal.integer != 0 ? "True" : "False";
        case Literal::Kind::Lowest:
            if (is_real(dtype)) {
                return "float(\"-inf\")";
            }
            return dtype == ScalarKind::Bool ? "False"
                                             : "torch.iinfo(" + torch_dtype(dtype) + ").min";
        case Literal::Kind::Highest:
            if (is_real(dtype)) {
                return "float(\"inf\")";
            }
            return dtype == ScalarKind::Bool ? "True"
                                             : "torch.iinfo(" + torch_dtype(dtype) + ").max";
        }
        return "0";
    }

    std::vector<std::string> arguments_;
    std::vector<std::string> parameters_;         // paths, in argument order
    std::vector<std::string> states_;             // paths, in argument order
    std::map<std::string, std::string> literals_; // constant name -> Python literal
    std::map<std::string, double> values_;        // constant name -> folded scalar value
    std::map<ScalarKind, std::string> zeros_;     // per-dtype zero constants
    std::string body_;
    std::string indent_ = "    ";
    std::vector<std::vector<std::string>> loop_names_;
    std::size_t loops_ = 0;
    std::size_t next_ = 0;
};

} // namespace

std::expected<std::string, std::string> export_torch_source(ir::Module& module,
                                                            const TorchSourceOptions& options) {
    TorchTarget target;
    return export_graph(module, options, target);
}

} // namespace linnet::backend
