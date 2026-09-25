#include "linnet/backend/jax_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

std::string jnp_dtype(ScalarKind dtype) {
    switch (dtype) {
    case ScalarKind::Bool:
        return "jnp.bool_";
    case ScalarKind::I8:
        return "jnp.int8";
    case ScalarKind::I16:
        return "jnp.int16";
    case ScalarKind::I32:
        return "jnp.int32";
    case ScalarKind::I64:
        return "jnp.int64";
    case ScalarKind::U8:
        return "jnp.uint8";
    case ScalarKind::U16:
        return "jnp.uint16";
    case ScalarKind::U32:
        return "jnp.uint32";
    case ScalarKind::U64:
        return "jnp.uint64";
    case ScalarKind::F16:
        return "jnp.float16";
    case ScalarKind::BF16:
        return "jnp.bfloat16";
    case ScalarKind::F32:
        return "jnp.float32";
    case ScalarKind::F64:
        return "jnp.float64";
    }
    return "jnp.float32";
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

// The generated module: straight-line `jax.numpy` over static shapes.
class JaxTarget : public GraphTarget {
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
            define("jnp.asarray(" + text + ", dtype=" + jnp_dtype(dtype) + ")");
        literals_[name] = text;
        if (literal.kind == Literal::Kind::Integer || literal.kind == Literal::Kind::Real) {
            values_[name] = literal.kind == Literal::Kind::Real
                                ? literal.real
                                : static_cast<double>(literal.integer);
        }
        return name;
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

    std::string compare(ir::CompareKind kind,
                        const TensorInfo& a,
                        const TensorInfo& b,
                        const Dims& shape) override {
        (void)shape;
        const char* op = kind == ir::CompareKind::Eq   ? "=="
                         : kind == ir::CompareKind::Ne ? "!="
                         : kind == ir::CompareKind::Lt ? "<"
                         : kind == ir::CompareKind::Le ? "<="
                         : kind == ir::CompareKind::Gt ? ">"
                                                       : ">=";
        return define(a.name + " " + op + " " + b.name);
    }

    std::string select(const TensorInfo& condition,
                       const TensorInfo& on_true,
                       const TensorInfo& on_false,
                       const Dims& shape,
                       ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        return define("jnp.where(" + condition.name + ", " + on_true.name + ", " + on_false.name +
                      ")");
    }

    std::string convert(const TensorInfo& value, ScalarKind dtype) override {
        const std::string name = define(value.name + ".astype(" + jnp_dtype(dtype) + ")");
        const auto found = values_.find(value.name);
        if (found != values_.end()) {
            values_[name] = found->second;
            literals_[name] = is_real(dtype)
                                  ? real_text(found->second)
                                  : std::to_string(static_cast<long long>(found->second));
        }
        return name;
    }

    std::string reshape(const TensorInfo& value, const Dims& shape) override {
        return define(value.name + ".reshape(" + dims_text(shape) + ")");
    }

    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override {
        (void)shape;
        return define("jnp.transpose(" + value.name + ", " + dims_text(permutation) + ")");
    }

    std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) override {
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
        return define("jnp.broadcast_to(" + source.name + ", " + dims_text(shape) + ")");
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
        return define("jnp.concatenate([" + list + "], axis=" + std::to_string(axis) + ")");
    }

    std::string iota(std::int64_t length) override {
        return define("jnp.arange(" + std::to_string(length) + ", dtype=jnp.int64)");
    }

    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override {
        (void)shape;
        std::string index;
        for (std::size_t k = 0; k < source.shape.size(); ++k) {
            index += (k == 0 ? "" : ", ") + indices.name + "[..., " + std::to_string(k) + "]";
        }
        return define(source.name + "[" + index + "]");
    }

    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        (void)shape;
        const std::string axes = ", axis=" + dims_text(dims) + ")";
        switch (kind) {
        case Reduction::Sum:
            return define("jnp.sum(" + body.name + axes);
        case Reduction::Prod:
            return define("jnp.prod(" + body.name + axes);
        case Reduction::Max:
            return define("jnp.max(" + body.name + axes);
        case Reduction::Min:
            return define("jnp.min(" + body.name + axes);
        case Reduction::Any:
            return define("jnp.any(" + body.name + axes);
        case Reduction::All:
            return define("jnp.all(" + body.name + axes);
        }
        return body.name;
    }

    bool broadcasts_elementwise() const override { return true; }

    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        ScalarKind dtype) override {
        (void)shape, (void)dtype;
        return define("jnp.einsum(\"" + einsum_equation(lhs_axes, rhs_axes, out_axes) + "\", " +
                      lhs.name + ", " + rhs.name + ")");
    }

    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        std::vector<const TensorInfo*> at;
        at.reserve(operands.size());
        for (const std::optional<TensorInfo>& operand : operands) {
            at.push_back(operand.has_value() ? &*operand : nullptr);
        }
        const std::string suffix = "(input dtype)";
        const bool fast = implementation.ends_with(suffix);
        std::string implementation_base =
            fast ? implementation.substr(0, implementation.size() - suffix.size()) : implementation;
        const std::string gqa = "(enable_gqa)";
        if (implementation_base.ends_with(gqa)) {
            implementation_base.resize(implementation_base.size() - gqa.size());
        }
        const auto name = [&](std::size_t i) -> std::string {
            return i < at.size() && at[i] != nullptr ? at[i]->name : "None";
        };
        const auto scalar = [&](std::size_t i) -> std::string {
            if (i < at.size() && at[i] != nullptr) {
                const auto found = literals_.find(at[i]->name);
                return found != literals_.end() ? found->second : "float(" + at[i]->name + ")";
            }
            return "None";
        };
        // Casts to f32 and back around a library call, as the canonical
        // bodies compute.
        const auto f32 = [&](std::size_t i) {
            return fast ? name(i) : name(i) + ".astype(jnp.float32)";
        };
        const auto back = [&](const std::string& expression, std::size_t like) {
            return fast ? expression : expression + ".astype(" + name(like) + ".dtype)";
        };
        if (implementation_base == "torch.nn.functional.embedding" && at.size() == 2 &&
            at[0] != nullptr && at[1] != nullptr) {
            return define("jnp.take(" + name(1) + ", " + name(0) + ", axis=0)");
        }
        if (implementation_base == "torch.tril" && at.empty() && shape.size() == 2) {
            std::string mask = define("jnp.tril(jnp.ones((" + std::to_string(shape[0]) + ", " +
                                      std::to_string(shape[1]) + "), dtype=bool), " +
                                      std::to_string(shape[1] - shape[0]) + ")");
            if (shape[0] == shape[1]) {
                causal_masks_.insert(mask);
            }
            return mask;
        }
        if (implementation_base == "torch.Tensor.index_copy" && at.size() == 3 &&
            at[0] != nullptr && at[1] != nullptr && at[2] != nullptr) {
            return define("jax.lax.dynamic_update_slice_in_dim(" + name(0) + ", " + name(1) + ", " +
                          name(2) + ".astype(jnp.int32), 2)");
        }
        if (implementation_base == "torch.matmul" && at.size() == 2) {
            return define("jnp.matmul(" + name(0) + ", " + name(1) + ")");
        }
        if (implementation_base == "torch.nn.functional.conv2d" && at.size() == 3 &&
            at[0] != nullptr && at[1] != nullptr) {
            const auto stride = call_generic("Stride");
            const auto pad = call_generic("Pad");
            if (!stride || !pad) {
                return std::nullopt;
            }
            const std::string s = std::to_string(*stride);
            const std::string p = std::to_string(*pad);
            const std::string mixed =
                define("jax.lax.conv_general_dilated(" + name(0) + ", " + name(1) + ", (" + s +
                       ", " + s + "), ((" + p + ", " + p + "), (" + p + ", " + p +
                       ")), dimension_numbers=(\"NCHW\", \"OIHW\", \"NCHW\"))");
            return at[2] != nullptr ? define(mixed + " + " + name(2) + ".reshape((1, -1, 1, 1))")
                                    : mixed;
        }
        if (implementation_base == "torch.nn.functional.linear" && at.size() == 3 &&
            at[0] != nullptr && at[1] != nullptr) {
            const std::string product = define(name(0) + " @ " + name(1) + ".T");
            return at[2] != nullptr ? define(product + " + " + name(2)) : product;
        }
        if (implementation_base == "torch.softmax" && at.size() == 1 && at[0] != nullptr) {
            return define(back("jax.nn.softmax(" + f32(0) + ", axis=-1)", 0));
        }
        if (implementation_base == "torch.relu" && at.size() == 1) {
            return define("jax.nn.relu(" + name(0) + ")");
        }
        if (implementation_base == "torch.sigmoid" && at.size() == 1) {
            return define("jax.nn.sigmoid(" + name(0) + ")");
        }
        if (implementation_base == "torch.nn.functional.silu" && at.size() == 1) {
            return define("jax.nn.silu(" + name(0) + ")");
        }
        if (implementation_base == "torch.nn.functional.gelu" && at.size() == 1) {
            return define("jax.nn.gelu(" + name(0) + ", approximate=False)");
        }
        if (implementation_base == "torch.nn.functional.gelu(tanh)" && at.size() == 1) {
            return define("jax.nn.gelu(" + name(0) + ", approximate=True)");
        }
        if (implementation_base == "torch.rms_norm" && at.size() == 3 && at[0] != nullptr) {
            const std::string x = define(f32(0));
            const std::string scale = define("jax.lax.rsqrt(jnp.mean(" + x + " * " + x +
                                             ", axis=-1, keepdims=True) + " + scalar(2) + ")");
            return define(back("(" + x + " * " + scale + ")", 0) + " * " + name(1));
        }
        if (implementation_base == "torch.nn.functional.layer_norm" && at.size() == 4 &&
            at[0] != nullptr) {
            const std::string x = define(f32(0));
            const std::string centered =
                define(x + " - jnp.mean(" + x + ", axis=-1, keepdims=True)");
            const std::string scale =
                define("jax.lax.rsqrt(jnp.mean(" + centered + " * " + centered +
                       ", axis=-1, keepdims=True) + " + scalar(3) + ")");
            const std::string scaled =
                define(back("(" + centered + " * " + scale + ")", 0) + " * " + name(1));
            return at[2] != nullptr ? define(scaled + " + " + name(2)) : scaled;
        }
        if (implementation_base == "torch.nn.functional.scaled_dot_product_attention" &&
            at.size() == 5 && at[0] != nullptr && at[1] != nullptr && at[2] != nullptr) {
            // `jax.nn.dot_product_attention` takes [B, S, N, D], groups
            // key/value heads natively, and has a causal mode, so a square
            // `causal_mask` disappears into `is_causal=True`.
            const std::string q = define("jnp.swapaxes(" + f32(0) + ", 1, 2)");
            const std::string k = define("jnp.swapaxes(" + f32(1) + ", 1, 2)");
            const std::string v = define("jnp.swapaxes(" + f32(2) + ", 1, 2)");
            std::string mask;
            if (at[4] != nullptr && causal_masks_.contains(at[4]->name)) {
                mask = ", is_causal=True";
            } else if (at[4] != nullptr) {
                mask = ", mask=" + name(4) + "[None, None]";
            }
            const std::string mixed = define("jax.nn.dot_product_attention(" + q + ", " + k + ", " +
                                             v + ", scale=" + scalar(3) + mask + ")");
            return define(back("jnp.swapaxes(" + mixed + ", 1, 2)", 0));
        }
        return std::nullopt;
    }

    // `jax.lax.while_loop` over a tuple of carried values, its condition and
    // body as nested functions written while the evaluator emits them.
    bool supports_while() const override { return true; }

    std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) override {
        Loop loop;
        loop.id = loops_made_++;
        loop.initial = initial;
        const std::string prefix = "w" + std::to_string(loop.id) + "_";
        for (std::size_t i = 0; i < initial.size(); ++i) {
            loop.names.push_back(prefix + std::to_string(i));
        }
        body_ += indent_ + "def _cond" + std::to_string(loop.id) + "(carried):\n";
        indent_ += "    ";
        body_ += indent_ + unpack(loop.names) + " = carried\n";
        loops_.push_back(loop);
        cse_.emplace_back(); // the condition's values live in its own function
        return loop.names;
    }

    std::vector<std::string> while_condition(const TensorInfo& predicate) override {
        const Loop& loop = loops_.back();
        body_ += indent_ + "return " + predicate.name + "\n";
        indent_.resize(indent_.size() - 4);
        cse_.pop_back();
        cse_.emplace_back(); // and the body's in its own
        body_ += indent_ + "def _body" + std::to_string(loop.id) + "(carried):\n";
        indent_ += "    ";
        body_ += indent_ + unpack(loop.names) + " = carried\n";
        return loop.names;
    }

    std::vector<std::string> end_while(const std::vector<TensorInfo>& next) override {
        const Loop loop = loops_.back();
        loops_.pop_back();
        std::string values;
        std::string inits;
        for (std::size_t i = 0; i < next.size(); ++i) {
            values += (i == 0 ? "" : ", ") + next[i].name;
            inits += (i == 0 ? "" : ", ") + loop.initial[i].name;
        }
        body_ += indent_ + "return (" + values + (next.size() == 1 ? ",)" : ")") + "\n";
        indent_.resize(indent_.size() - 4);
        cse_.pop_back();
        const std::string result = "w" + std::to_string(loop.id);
        body_ += indent_ + result + " = jax.lax.while_loop(_cond" + std::to_string(loop.id) +
                 ", _body" + std::to_string(loop.id) + ", (" + inits +
                 (next.size() == 1 ? ",)" : ")") + ")\n";
        std::vector<std::string> finals;
        finals.reserve(next.size());
        for (std::size_t i = 0; i < next.size(); ++i) {
            finals.push_back(result + "[" + std::to_string(i) + "]");
        }
        return finals;
    }

    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        std::string out = "# " + block_name + "." + entry_name + " from module " + module_path +
                          ", generated by `linnet jax` for one shape\n"
                          "# binding. `main` takes the entry's inputs, then the parameters in\n"
                          "# PARAMETERS order, then the states in STATES order; it returns the\n"
                          "# entry's RESULTS results followed by the states in NEXT_STATES\n"
                          "# order. Linnet's `i64` needs 64-bit integers enabled.\n"
                          "import jax\n"
                          "import jax.numpy as jnp\n\n"
                          "jax.config.update(\"jax_enable_x64\", True)\n\n";
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
        std::string tail = "    return (";
        std::vector<std::string> outputs;
        outputs.reserve(results.size() + states.size());
        for (const TensorInfo& result : results) {
            outputs.push_back(result.name);
        }
        for (const auto& [path, value] : states) {
            outputs.push_back(value.name);
        }
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            tail += (i == 0 ? "" : ", ") + outputs[i];
        }
        out += prune_python_assignments(body_, tail);
        out += tail + (outputs.size() == 1 ? ",)\n" : ")\n");
        return out;
    }

private:
    struct Loop {
        std::size_t id = 0;
        std::vector<TensorInfo> initial;
        std::vector<std::string> names;
    };

    // Values are immutable, so an expression already computed in this scope
    // or an enclosing one names the same array (see the torch target).
    std::string define(const std::string& expression) {
        for (auto scope = cse_.rbegin(); scope != cse_.rend(); ++scope) {
            const auto found = scope->find(expression);
            if (found != scope->end()) {
                return found->second;
            }
        }
        const std::string name = "v" + std::to_string(next_++);
        body_ += indent_ + name + " = " + expression + "\n";
        cse_.back()[expression] = name;
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
            return define(is_real(operands[0].dtype) ? a + " / " + b
                                                     : "jax.lax.div(" + a + ", " + b + ")");
        case Elementwise::Rem:
            return define("jnp.fmod(" + a + ", " + b + ")");
        case Elementwise::Min:
            return define("jnp.minimum(" + a + ", " + b + ")");
        case Elementwise::Max:
            return define("jnp.maximum(" + a + ", " + b + ")");
        case Elementwise::And:
        case Elementwise::BitAnd:
            return define(a + " & " + b);
        case Elementwise::Or:
        case Elementwise::BitOr:
            return define(a + " | " + b);
        case Elementwise::BitXor:
            return define(a + " ^ " + b);
        case Elementwise::Shl:
            return define("jnp.left_shift(" + a + ", " + b + ")");
        case Elementwise::Shr:
            return define("jnp.right_shift(" + a + ", " + b + ")");
        case Elementwise::Not:
            return define("~" + a);
        case Elementwise::Neg:
            return define("-" + a);
        case Elementwise::Exp:
            return define("jnp.exp(" + a + ")");
        case Elementwise::Log:
            return define("jnp.log(" + a + ")");
        case Elementwise::Sqrt:
            return define("jnp.sqrt(" + a + ")");
        case Elementwise::Rsqrt:
            return define("jax.lax.rsqrt(" + a + ")");
        case Elementwise::Sin:
            return define("jnp.sin(" + a + ")");
        case Elementwise::Cos:
            return define("jnp.cos(" + a + ")");
        case Elementwise::Tanh:
            return define("jnp.tanh(" + a + ")");
        case Elementwise::Abs:
            return define("jnp.abs(" + a + ")");
        }
        return define(a);
    }

    // Scalar arithmetic on constants folds to a literal for kernel keywords.
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
        const double a = values[0];
        const double b = values.size() > 1 ? values[1] : 0.0;
        double result = 0.0;
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
        default:
            return;
        }
        if (!is_real(dtype)) {
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

    static std::string unpack(const std::vector<std::string>& names) {
        std::string out;
        for (std::size_t i = 0; i < names.size(); ++i) {
            out += (i == 0 ? "" : ", ") + names[i];
        }
        return names.size() == 1 ? out + "," : out;
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
                return "-jnp.inf";
            }
            return dtype == ScalarKind::Bool ? "False" : "jnp.iinfo(" + jnp_dtype(dtype) + ").min";
        case Literal::Kind::Highest:
            if (is_real(dtype)) {
                return "jnp.inf";
            }
            return dtype == ScalarKind::Bool ? "True" : "jnp.iinfo(" + jnp_dtype(dtype) + ").max";
        }
        return "0";
    }

    std::vector<std::string> arguments_;
    std::vector<std::string> parameters_;
    std::vector<std::string> states_;
    std::map<std::string, std::string> literals_;
    std::map<std::string, double> values_;
    std::vector<Loop> loops_;
    std::size_t loops_made_ = 0;
    std::string body_;
    std::string indent_ = "    ";
    std::set<std::string> causal_masks_;                     // square masks from `causal_mask`
    std::vector<std::map<std::string, std::string>> cse_{1}; // expression -> name, per scope
    std::size_t next_ = 0;
};

} // namespace

std::expected<std::string, std::string> export_jax_source(ir::Module& module,
                                                          const JaxSourceOptions& options) {
    JaxTarget target;
    return export_graph(module, options, target);
}

} // namespace linnet::backend
