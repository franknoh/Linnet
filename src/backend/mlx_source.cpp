#include "linnet/backend/mlx_source.hpp"

#include "linnet/backend/python_target.hpp"
#include "linnet/backend/recorded_graph.hpp"
#include "linnet/support/text.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

// What every module defines: `mlx.core` has no `fmod` and no integer
// division rounding toward zero, and an array written at an index stays the
// array other names hold.
constexpr const char* prelude = "def _div(a, b):\n"
                                "    # Integer division rounding toward zero.\n"
                                "    q = mx.floor_divide(a, b)\n"
                                "    return mx.where((q * b != a) & ((a < 0) != (b < 0)), q + 1, q)\n"
                                "\n"
                                "\n"
                                "def _rem(a, b):\n"
                                "    # The remainder of a division rounding toward zero.\n"
                                "    if mx.issubdtype(a.dtype, mx.integer):\n"
                                "        return a - b * _div(a, b)\n"
                                "    return a - b * mx.trunc(a / b)\n"
                                "\n"
                                "\n"
                                "def _put(x, index, value):\n"
                                "    # `x` with `x[index]` set to `value`, as a new array: every name\n"
                                "    # bound to `x` keeps its values.\n"
                                "    y = mx.array(x)\n"
                                "    y[index] = value\n"
                                "    return y\n"
                                "\n"
                                "\n";

// What a module converting to or from a 4-bit or an FP8 E5M2 tensor imports.
constexpr const char* formats_import =
    "from linnet.mlx.formats import from_e5m2 as _from_e5m2, to_e5m2 as _to_e5m2\n"
    "from linnet.mlx.formats import pack4 as _pack4, unpack4 as _unpack4\n\n";

// The generated module: straight-line `mlx.core` over static shapes.
class MlxTarget : public PythonTarget {
public:
    explicit MlxTarget(const MlxSourceOptions& options)
        : PythonTarget("mx",
                       "bool_",
                       options.prepare,
                       {options.lora, options.lora_rank, options.lora_alpha}) {}

    std::string state(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        state_types_.emplace_back(shape, dtype);
        return PythonTarget::state(path, shape, dtype);
    }

    std::string compare(ir::CompareKind kind,
                        const TensorInfo& a,
                        const TensorInfo& b,
                        const Dims& shape) override {
        (void)shape;
        return define(a.name + " " + std::string(ir::compare_symbol(kind)) + " " + b.name);
    }

    std::string convert(const TensorInfo& value, ScalarKind dtype) override {
        const ScalarKind from = value.dtype;
        if (from == dtype) {
            return value.name;
        }
        if (sema::is_packed(from) || sema::is_packed(dtype) || from == ScalarKind::F8E5M2 ||
            dtype == ScalarKind::F8E5M2) {
            // Bytes of 4-bit elements, two a byte along the last axis, and
            // FP8 E5M2, the high byte of an f16 (`linnet.mlx.formats`).
            uses_formats_ = true;
            std::string wide = value.name;
            if (sema::is_packed(from)) {
                wide = define("_unpack4(" + wide + ", \"" + std::string(sema::scalar_name(from)) +
                              "\")");
            } else if (from == ScalarKind::F8E5M2) {
                wide = define("_from_e5m2(" + wide + ")");
            }
            if (sema::is_packed(dtype)) {
                return define("_pack4(" + wide + ", \"" + std::string(sema::scalar_name(dtype)) +
                              "\")");
            }
            if (dtype == ScalarKind::F8E5M2) {
                return define("_to_e5m2(" + wide + ")");
            }
            return define(wide + ".astype(" + dtype_name(dtype) + ")");
        }
        if (from == ScalarKind::F8E4M3) {
            const ScalarKind wide = sema::is_float(dtype) ? dtype : ScalarKind::F32;
            std::string name = define("mx.from_fp8(" + value.name + ", dtype=" + dtype_name(wide) +
                                      ")");
            return wide == dtype ? name : define(name + ".astype(" + dtype_name(dtype) + ")");
        }
        if (dtype == ScalarKind::F8E4M3) {
            return define("mx.to_fp8(" + value.name + ")");
        }
        const std::string name = define(value.name + ".astype(" + dtype_name(dtype) + ")");
        fold_convert(name, value, dtype);
        return name;
    }

    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override {
        (void)shape;
        return define("mx.transpose(" + value.name + ", " + python_tuple(permutation) + ")");
    }

    std::string
    concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) override {
        (void)shape;
        std::string list;
        list += join(parts, ", ", [&](const auto& item) { return item.name; });
        return define("mx.concatenate([" + list + "], axis=" + std::to_string(axis) + ")");
    }

    std::string iota(std::int64_t length) override {
        return define("mx.arange(" + std::to_string(length) + ", dtype=mx.int64)");
    }

    std::optional<std::string>
    update_row(const TensorInfo& stack, const TensorInfo& index, const TensorInfo& value) override {
        return define("mx.slice_update(" + stack.name + ", mx.expand_dims(" + value.name +
                      ", 0), " + index.name + ".astype(mx.int32).reshape(1), (0,))");
    }

    // MLX differentiates the body as it is: an op's `grad` is not used, and
    // nothing is detached.
    std::vector<std::string>
    begin_custom_gradient(const std::vector<TensorInfo>& arguments) override {
        return GraphTarget::begin_custom_gradient(arguments);
    }

    std::optional<std::string> scatter_add(const Dims& shape,
                                           ScalarKind dtype,
                                           const TensorInfo& indices,
                                           const TensorInfo& values) override {
        std::vector<std::string> axes;
        const std::int64_t count = indices.shape.empty() ? 0 : indices.shape.back();
        for (std::int64_t k = 0; k < count; ++k) {
            axes.push_back(indices.name + "[..., " + std::to_string(k) + "]");
        }
        return define("mx.zeros(" + python_tuple(shape) + ", dtype=" + dtype_name(dtype) +
                      ").at[" + join(axes, ", ") + (axes.size() == 1 ? "," : "") + "].add(" +
                      values.name + ")");
    }

    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        (void)shape;
        const std::string axes = ", axis=" + python_tuple(dims) + ")";
        switch (kind) {
        case Reduction::Sum:
            return define("mx.sum(" + body.name + axes);
        case Reduction::Prod:
            return define("mx.prod(" + body.name + axes);
        case Reduction::Max:
            return define("mx.max(" + body.name + axes);
        case Reduction::Min:
            return define("mx.min(" + body.name + axes);
        case Reduction::Any:
            return define("mx.any(" + body.name + axes);
        case Reduction::All:
            return define("mx.all(" + body.name + axes);
        }
        return body.name;
    }

    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           ScalarKind dtype) override {
        (void)dtype;
        const std::vector<const TensorInfo*> at = operand_pointers(operands);
        const CallName parsed = call_name(implementation);
        const std::string& base = parsed.base;
        const bool fast = parsed.fast;
        const auto present = [&](std::size_t i) { return i < at.size() && at[i] != nullptr; };
        const auto name = [&](std::size_t i) -> std::string {
            return present(i) ? at[i]->name : "None";
        };
        const auto scalar = [&](std::size_t i) -> std::string {
            if (present(i)) {
                const auto found = literals_.find(at[i]->name);
                return found != literals_.end() ? found->second : at[i]->name + ".item()";
            }
            return "None";
        };
        // Casts to f32 and back around a library call, as the canonical
        // bodies compute.
        const auto f32 = [&](std::size_t i) {
            return fast ? name(i) : name(i) + ".astype(mx.float32)";
        };
        const auto back = [&](const std::string& expression, std::size_t like) {
            return fast ? expression : "(" + expression + ").astype(" + name(like) + ".dtype)";
        };
        const auto i32 = [&](std::size_t i) { return name(i) + ".astype(mx.int32)"; };
        if (base == "torch.nn.functional.embedding" && at.size() == 2 && present(0) &&
            present(1)) {
            return define("mx.take(" + name(1) + ", " + name(0) + ", axis=0)");
        }
        if (base == "torch.index_select" && at.size() == 2 && present(0) && present(1)) {
            return define("mx.take(" + name(0) + ", " + name(1) + ", axis=-1)");
        }
        if (base == "linnet.decode_fp8" && at.size() == 1 && present(0)) {
            return define("mx.from_fp8(" + name(0) + ", dtype=mx.float32)");
        }
        if (base == "torch.tril" && at.empty() && shape.size() == 2) {
            std::string mask = define("mx.tril(mx.ones(" + python_tuple(shape) +
                                      ", dtype=mx.bool_), " +
                                      std::to_string(shape[1] - shape[0]) + ")");
            if (shape[0] == shape[1]) {
                causal_masks_.insert(mask);
            }
            return mask;
        }
        if (base == "torch.Tensor.index_copy" && at.size() == 3 && present(0) && present(1) &&
            present(2)) {
            return define("mx.slice_update(" + name(0) + ", " + name(1) + ", " + i32(2) +
                          ".reshape(1), (2,))");
        }
        if (base == "torch.Tensor.index_put(tokens)" && at.size() == 4 && present(0) &&
            present(1) && present(2) && present(3) && at[0]->shape.size() == 4) {
            // `write_tokens`: token p at (rows[p], positions[p]), a scatter of
            // P x H vectors.
            return define("_put(" + name(0) + ", (" + i32(2) + "[:, None], mx.arange(" +
                          std::to_string(at[0]->shape[1]) + ")[None, :], " + i32(3) +
                          "[:, None]), mx.transpose(" + name(1) + "[0], (1, 0, 2)))");
        }
        if (base == "torch.Tensor.index_put" && (at.size() == 3 || at.size() == 4) && present(0) &&
            present(1) && at[0]->shape.size() == 4) {
            const Dims& cache = at[0]->shape;
            if (at.size() == 3) {
                // `write_rows`: row b at at[b], a scatter of B x H vectors.
                return define("_put(" + name(0) + ", (mx.arange(" + std::to_string(cache[0]) +
                              ")[:, None], mx.arange(" + std::to_string(cache[1]) +
                              ")[None, :], " + i32(2) + "[:, None]), " + name(1) + "[:, :, 0])");
            }
            if (present(2) && !at[2]->shape.empty()) {
                // `write_slots`: a span in each of several rows, a scatter.
                return define("_put(" + name(0) + ", (" + i32(2) + "[:, None, None], mx.arange(" +
                              std::to_string(cache[1]) + ")[None, :, None], (" + i32(3) +
                              " + mx.arange(" + std::to_string(at[1]->shape[2]) +
                              "))[None, None, :]), " + name(1) + ")");
            }
            // `write_slot`: one row's span, a slice write.
            return define("mx.slice_update(" + name(0) + ", " + name(1) + ", mx.stack([" + i32(2) +
                          ", " + i32(3) + "]), (0, 2))");
        }
        if (base == "torch.matmul" && at.size() == 2) {
            return define("mx.matmul(" + name(0) + ", " + name(1) + ")");
        }
        if (is_convolution(base) && at.size() == 3 && present(0) && present(1)) {
            const auto window = conv_window(base);
            if (!window) {
                return std::nullopt;
            }
            // MLX convolves channels last: [N, ..., C] by [O, ..., I].
            const bool flat = window->strides.size() == 1;
            const std::string last = flat ? "(0, 2, 1)" : "(0, 2, 3, 1)";
            const std::string first = flat ? "(0, 2, 1)" : "(0, 3, 1, 2)";
            const std::string x = define("mx.transpose(" + name(0) + ", " + last + ")");
            const std::string w = define("mx.transpose(" + name(1) + ", " + last + ")");
            std::string call;
            if (flat) {
                call = "mx.conv1d(" + x + ", " + w + ", stride=" +
                       std::to_string(window->strides[0]) +
                       ", padding=" + std::to_string(window->pads[0]) + ")";
            } else {
                call = "mx.conv2d(" + x + ", " + w + ", stride=" + python_tuple(window->strides) +
                       ", padding=" + python_tuple(window->pads) + ")";
            }
            const std::string mixed = define("mx.transpose(" + call + ", " + first + ")");
            return present(2) ? define(mixed + " + " + name(2) +
                                       (flat ? ".reshape((1, -1, 1))" : ".reshape((1, -1, 1, 1))"))
                              : mixed;
        }
        if (base == "torch.nn.functional.linear" && at.size() == 3 && present(0) && present(1)) {
            std::string product = define(name(0) + " @ " + name(1) + ".T");
            if (present(2)) {
                product = define(product + " + " + name(2));
            }
            const auto path = lora_target(name(1));
            if (!path || at[1]->shape.size() != 2) {
                return product;
            }
            // A low-rank adapter beside the weight (LoRA): `x @ A.T @ B.T`,
            // scaled by alpha / rank.
            const auto [a, b] = adapters(*path, at[1]->shape, at[1]->dtype);
            const std::string low = define("(" + name(0) + " @ " + a + ".T) @ " + b + ".T");
            return define(product + " + " + low + " * " +
                          python_float(lora_.alpha / static_cast<double>(lora_.rank)));
        }
        if (base == "torch.softmax" && at.size() == 1 && present(0)) {
            return define(back("mx.softmax(" + f32(0) + ", axis=-1)", 0));
        }
        if (base == "torch.relu" && at.size() == 1) {
            return define("mx.maximum(" + name(0) + ", 0)");
        }
        if (base == "torch.sigmoid" && at.size() == 1) {
            return define("mx.sigmoid(" + name(0) + ")");
        }
        if (base == "torch.nn.functional.silu" && at.size() == 1) {
            return define(name(0) + " * mx.sigmoid(" + name(0) + ")");
        }
        if (base == "torch.nn.functional.gelu" && at.size() == 1) {
            return define("0.5 * " + name(0) + " * (1 + mx.erf(" + name(0) +
                          " * 0.7071067811865476))");
        }
        if (base == "torch.nn.functional.gelu(tanh)" && at.size() == 1) {
            const std::string& x = name(0);
            return define("0.5 * " + x + " * (1 + mx.tanh(0.7978845608028654 * (" + x +
                          " + 0.044715 * " + x + " * " + x + " * " + x + ")))");
        }
        if (base == "torch.rms_norm" && at.size() == 3 && present(0)) {
            if (fast && present(1)) {
                return define("mx.fast.rms_norm(" + name(0) + ", " + name(1) + ", " + scalar(2) +
                              ")");
            }
            const std::string x = define(f32(0));
            const std::string scale = define("mx.rsqrt(mx.mean(" + x + " * " + x +
                                             ", axis=-1, keepdims=True) + " + scalar(2) + ")");
            const std::string normed = define(back(x + " * " + scale, 0));
            return present(1) ? define(normed + " * " + name(1)) : normed;
        }
        if (base == "torch.nn.functional.layer_norm" && at.size() == 4 && present(0)) {
            if (fast) {
                return define("mx.fast.layer_norm(" + name(0) + ", " + name(1) + ", " + name(2) +
                              ", " + scalar(3) + ")");
            }
            const std::string x = define(f32(0));
            const std::string centered =
                define(x + " - mx.mean(" + x + ", axis=-1, keepdims=True)");
            const std::string scale =
                define("mx.rsqrt(mx.mean(" + centered + " * " + centered +
                       ", axis=-1, keepdims=True) + " + scalar(3) + ")");
            std::string scaled = define(back(centered + " * " + scale, 0));
            if (present(1)) {
                scaled = define(scaled + " * " + name(1));
            }
            return present(2) ? define(scaled + " + " + name(2)) : scaled;
        }
        if (base == "torch.nn.functional.scaled_dot_product_attention" && at.size() == 5 &&
            present(0) && present(1) && present(2)) {
            // MLX's fused attention takes PyTorch's [B, N, S, D], groups
            // key/value heads itself, and has a causal mode, so a square
            // `causal_mask` disappears into `mask="causal"`.
            std::string mask;
            if (present(4) && causal_masks_.contains(at[4]->name)) {
                mask = ", mask=\"causal\"";
            } else if (present(4)) {
                // A shared [Q, K] mask, or one per sequence ([B, Q, K]).
                mask = ", mask=" + name(4) + (at[4]->shape.size() == 3 ? "[:, None]" : "");
            }
            return define(back("mx.fast.scaled_dot_product_attention(" + f32(0) + ", " + f32(1) +
                                   ", " + f32(2) + ", scale=" + scalar(3) + mask + ")",
                               0));
        }
        return std::nullopt;
    }

    // A Python `while`, its condition read back each iteration: such a
    // module runs uncompiled (`DYNAMIC`).
    bool supports_while() const override { return true; }

    std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) override {
        dynamic_ = true;
        std::vector<std::string> names;
        const std::string prefix = "w" + std::to_string(loops_++) + "_";
        for (std::size_t i = 0; i < initial.size(); ++i) {
            names.push_back(prefix + std::to_string(i));
            body_ += indent_ + names.back() + " = " + initial[i].name + "\n";
        }
        body_ += indent_ + "while True:\n";
        indent_ += "    ";
        loop_names_.push_back(names);
        cse_.emplace_back(); // values computed in the body belong to one iteration
        return names;
    }

    std::vector<std::string> while_condition(const TensorInfo& predicate) override {
        body_ += indent_ + "if not " + predicate.name + ".item():\n" + indent_ + "    break\n";
        return loop_names_.back();
    }

    // A Python `for` over the range, unrolled where `mx.compile` traces it.
    // The index enters the body as an array.
    bool supports_counted() const override { return true; }

    std::vector<std::string> begin_counted(std::int64_t start,
                                           std::int64_t stop,
                                           const std::vector<TensorInfo>& initial) override {
        const std::string prefix = "f" + std::to_string(loops_++) + "_";
        std::vector<std::string> names;
        for (std::size_t i = 0; i < initial.size(); ++i) {
            names.push_back(prefix + std::to_string(i));
            body_ += indent_ + names.back() + " = " + initial[i].name + "\n";
        }
        body_ += indent_ + "for " + prefix + "n in range(" + std::to_string(start) + ", " +
                 std::to_string(stop) + "):\n";
        indent_ += "    ";
        body_ += indent_ + prefix + "i = mx.array(" + prefix + "n, dtype=mx.int64)\n";
        loop_names_.push_back(names);
        cse_.emplace_back();
        names.insert(names.begin(), prefix + "i");
        return names;
    }

    std::vector<std::string> end_counted(const std::vector<TensorInfo>& next) override {
        return end_while(next);
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
        cse_.pop_back();
        return names;
    }

    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        std::string out =
            "# " + entry_label(block_name, entry_name) + " from module " + module_path +
            ", generated by `linnet mlx` for one shape\n"
            "# binding. `main` takes the entry's inputs, then the parameters in\n"
            "# PARAMETERS order, then the states in STATES order; it returns the\n"
            "# entry's RESULTS results followed by the states in NEXT_STATES\n"
            "# order. 8-bit and 4-bit floats are held as their bytes.\n"
            "import mlx.core as mx\n\n";
        if (uses_formats_) {
            out += formats_import;
        }
        out += "\n";
        out += prelude;
        out += helpers_;
        out += definitions_;
        out += "PARAMETERS = " + string_list(parameters_) + "\n";
        out += "PARAMETER_TYPES = " + types_list(parameter_types_) + "\n";
        out += "STATES = " + string_list(states_) + "\n";
        out += "STATE_TYPES = " + types_list(state_types_) + "\n";
        std::vector<std::string> next_states;
        next_states.reserve(states.size());
        for (const auto& [path, value] : states) {
            next_states.push_back(path);
        }
        out += "NEXT_STATES = " + string_list(next_states) + "\n";
        out += "RESULTS = " + std::to_string(results.size()) + "\n";
        out += std::string("DYNAMIC = ") + (dynamic_ ? "True" : "False") + "\n";
        std::string tail = "    return (";
        std::vector<std::string> outputs;
        outputs.reserve(results.size() + states.size());
        for (const TensorInfo& result : results) {
            outputs.push_back(result.name);
        }
        for (const auto& [path, value] : states) {
            outputs.push_back(value.name);
        }
        tail += join(outputs, ", ");
        tail += outputs.size() == 1 ? ",)\n" : ")\n";
        // Weight-only work, run once per loaded model (`--prepare`).
        PreparedSplit prepared;
        prepared.body = prune_python_assignments(body_, tail);
        if (prepare_) {
            prepared = split_prepared(prepared.body, tail, parameters_, "");
        }
        if (!prepared.outputs.empty()) {
            out += "PREPARED = " + string_list(prepared.keys) + "\n";
            out += "PREPARE_INPUTS = " + string_list(prepared.inputs) + "\n\n\n";
            out += "def prepare(";
            out += join(prepared.inputs, ", ");
            std::string returned = "    return (";
            returned += join(prepared.outputs, ", ");
            returned += prepared.outputs.size() == 1 ? ",)\n" : ")\n";
            out += "):\n" + prepared.prepare + returned;
        }
        out += "\n\ndef main(";
        std::vector<std::string> arguments = arguments_;
        arguments.insert(arguments.end(), prepared.outputs.begin(), prepared.outputs.end());
        out += join(arguments, ", ");
        out += "):\n" + prepared.body + tail;
        return out;
    }

private:
    // `[((2, 3), "f32"), ...]`: shapes and Linnet dtypes.
    static std::string types_list(const std::vector<std::pair<Dims, ScalarKind>>& types) {
        std::string out = "[";
        for (std::size_t i = 0; i < types.size(); ++i) {
            const auto& [shape, dtype] = types[i];
            out += (i == 0 ? "(" : ", (") + python_tuple(shape) + ", \"" +
                   std::string(sema::scalar_name(dtype)) + "\")";
        }
        return out + "]";
    }

    // The `mx` spellings of what `PythonTarget` writes.
    std::string constant_expression(const std::string& text, ScalarKind dtype) override {
        return "mx.array(" + text + ", dtype=" + dtype_name(dtype) + ")";
    }
    std::string divide_integers(const std::string& a, const std::string& b) override {
        return "_div(" + a + ", " + b + ")";
    }
    std::string shift(const std::string& a, const std::string& b, bool left) override {
        return std::string(left ? "mx.left_shift(" : "mx.right_shift(") + a + ", " + b + ")";
    }
    std::string reciprocal_sqrt(const std::string& a) override { return "mx.rsqrt(" + a + ")"; }
    std::string expand(const std::string& value, const Dims& shape) override {
        return "mx.broadcast_to(" + value + ", " + python_tuple(shape) + ")";
    }
    std::string infinity() const override { return "float(\"inf\")"; }
    std::string detach(const std::string& value) const override {
        return "mx.stop_gradient(" + value + ")";
    }
    std::string truncated_remainder(const std::string& a, const std::string& b) override {
        return "_rem(" + a + ", " + b + ")";
    }

    std::vector<std::pair<Dims, ScalarKind>> state_types_; // shape and dtype, in STATES order
    std::vector<std::vector<std::string>> loop_names_;     // each open loop's carried names
    std::size_t loops_ = 0;
    bool dynamic_ = false;      // a `while` reads its condition back
    bool uses_formats_ = false; // the module imports `linnet.mlx.formats`
};

} // namespace

std::expected<std::unique_ptr<GraphTarget>, std::string>
make_mlx_target(const MlxSourceOptions& options) {
    if (const auto conflict = python_options_conflict(options)) {
        return std::unexpected(*conflict);
    }
    return std::make_unique<MlxTarget>(options);
}

std::expected<std::string, std::string> export_mlx_source(ir::Module& module,
                                                          const MlxSourceOptions& options) {
    if (const auto conflict = python_options_conflict(options)) {
        return std::unexpected(*conflict);
    }
    MlxTarget target(options);
    MlxTarget recorded(options);
    return export_through_graph(module, options, recorded, "mlx", target);
}

} // namespace linnet::backend
