#include "linnet/backend/jax_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
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

// What a module with routed experts imports (`native_call`).
constexpr const char* experts_import = "import linnet.jax.moe as _moe\n\n";
// What a module with a blockwise output head and loss imports.
constexpr const char* loss_import = "import linnet.jax.loss as _loss\n\n";
// What a module whose parameters are gathered from their parts imports.
constexpr const char* gather_import = "from linnet.jax.fsdp import gather as _gather\n\n";

bool identifier_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

// Every identifier `text` mentions.
std::set<std::string> identifiers(std::string_view text) {
    std::set<std::string> out;
    std::size_t i = 0;
    while (i < text.size()) {
        if (identifier_char(text[i]) && std::isdigit(static_cast<unsigned char>(text[i])) == 0 &&
            (i == 0 || !identifier_char(text[i - 1]))) {
            std::size_t end = i;
            while (end < text.size() && identifier_char(text[end])) {
                ++end;
            }
            out.emplace(text.substr(i, end - i));
            i = end;
        } else {
            ++i;
        }
    }
    return out;
}

// `body` with each region between `# remat N begin` and `# remat N end`
// moved into a function the backward pass computes again: `jax.checkpoint`
// keeps what the function reads and returns, and recomputes the rest. What
// it returns is every value it defines that the code after it (or `tail`)
// reads.
std::string outline_remat(const std::string& body, const std::string& tail) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start < body.size()) {
        std::size_t end = body.find('\n', start);
        end = end == std::string::npos ? body.size() : end;
        lines.emplace_back(body.substr(start, end - start));
        start = end + 1;
    }
    const auto marker = [](const std::string& line, std::string_view what) {
        const std::size_t first = line.find_first_not_of(' ');
        return first != std::string::npos && line.compare(first, 8, "# remat ") == 0 &&
               line.ends_with(what);
    };
    std::string out;
    std::size_t i = 0;
    while (i < lines.size()) {
        if (!marker(lines[i], " begin")) {
            out += lines[i++] + "\n";
            continue;
        }
        const std::string indent = lines[i].substr(0, lines[i].find_first_not_of(' '));
        const std::string id = lines[i].substr(
            indent.size() + 8, lines[i].size() - indent.size() - 8 - std::strlen(" begin"));
        std::size_t close = i + 1;
        while (close < lines.size() &&
               !(marker(lines[close], " end") &&
                 lines[close].find("# remat " + id + " end") != std::string::npos)) {
            ++close;
        }
        std::vector<std::string> defined;
        for (std::size_t k = i + 1; k < close; ++k) {
            const std::string& line = lines[k];
            if (!line.starts_with(indent) || line.size() <= indent.size() ||
                line[indent.size()] == ' ') {
                continue;
            }
            const std::size_t equals = line.find(" = ", indent.size());
            const std::string name = equals == std::string::npos
                                         ? std::string()
                                         : line.substr(indent.size(), equals - indent.size());
            if (!name.empty() && std::ranges::all_of(name, identifier_char)) {
                defined.push_back(name);
            }
        }
        std::string after = tail;
        for (std::size_t k = close + 1; k < lines.size(); ++k) {
            after += lines[k] + "\n";
        }
        const std::set<std::string> read_after = identifiers(after);
        std::vector<std::string> outputs;
        for (const std::string& name : defined) {
            if (read_after.contains(name)) {
                outputs.push_back(name);
            }
        }
        if (!outputs.empty()) {
            std::string returned = "(";
            for (std::size_t k = 0; k < outputs.size(); ++k) {
                returned.append(k == 0 ? "" : ", ").append(outputs[k]);
            }
            returned.append(outputs.size() == 1 ? ",)" : ")");
            out.append(indent).append("def _remat").append(id).append("():\n");
            for (std::size_t k = i + 1; k < close; ++k) {
                out.append(lines[k].empty() ? "" : "    ").append(lines[k]).append("\n");
            }
            out.append(indent).append("    return ").append(returned).append("\n");
            out.append(indent).append(returned).append(" = jax.checkpoint(_remat");
            out.append(id).append(")()\n");
        }
        i = close + 1;
    }
    return out;
}

// The generated module: straight-line `jax.numpy` over static shapes.
class JaxTarget : public GraphTarget {
public:
    explicit JaxTarget(const JaxSourceOptions& options)
        : prepare_(options.prepare), full_precision_(options.full_precision), lora_(options.lora),
          lora_rank_(options.lora_rank), lora_alpha_(options.lora_alpha) {}

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
        parameters_end_ = arguments_.size();
        return argument;
    }

    // The path of the weight argument `argument` names when a `--lora`
    // pattern matches it.
    std::optional<std::string> lora_target(const std::string& given) const {
        // A sharded weight is used gathered: its argument is what was gathered.
        const auto gathered = gathered_.find(given);
        const std::string& argument = gathered == gathered_.end() ? given : gathered->second;
        if (lora_.empty() || argument.size() < 2 || argument[0] != 'p' ||
            !std::all_of(argument.begin() + 1, argument.end(), [](char c) {
                return std::isdigit(static_cast<unsigned char>(c)) != 0;
            })) {
            return std::nullopt;
        }
        const std::size_t index = std::stoul(argument.substr(1));
        if (index >= parameters_.size() || !parameters_[index].ends_with(".weight")) {
            return std::nullopt;
        }
        const std::string& path = parameters_[index];
        if (std::ranges::any_of(
                lora_, [&](const std::string& pattern) { return glob_match(pattern, path); })) {
            return path;
        }
        return std::nullopt;
    }

    // The adapter parameters of the weight at `path`: `lora_a` and `lora_b`
    // of its block, made once, after the other parameters among `main`'s
    // arguments (and after them in PARAMETERS).
    std::pair<std::string, std::string> adapters(const std::string& path) {
        if (const auto found = adapters_.find(path); found != adapters_.end()) {
            return found->second;
        }
        const std::string block = path.substr(0, path.size() - std::string_view(".weight").size());
        const auto add = [&](const std::string& adapter) {
            const std::string argument = "p" + std::to_string(parameters_.size());
            parameters_.push_back(block + "." + adapter);
            arguments_.insert(arguments_.begin() + static_cast<std::ptrdiff_t>(parameters_end_),
                              argument);
            ++parameters_end_;
            return argument;
        };
        const std::string a = add("lora_a");
        const std::string b = add("lora_b");
        return adapters_.emplace(path, std::make_pair(a, b)).first->second;
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
            literals_[name] = sema::is_float(dtype)
                                  ? python_float(found->second)
                                  : std::to_string(static_cast<long long>(found->second));
        }
        return name;
    }

    std::string reshape(const TensorInfo& value, const Dims& shape) override {
        return define(value.name + ".reshape(" + python_tuple(shape) + ")");
    }

    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override {
        (void)shape;
        return define("jnp.transpose(" + value.name + ", " + python_tuple(permutation) + ")");
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
        return define("jnp.broadcast_to(" + source.name + ", " + python_tuple(shape) + ")");
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
        for (std::size_t k = 0; k < static_cast<std::size_t>(indices.shape.back()); ++k) {
            index += (k == 0 ? "" : ", ") + indices.name + "[..., " + std::to_string(k) + "]";
        }
        return define(source.name + "[" + index + "]");
    }

    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        (void)shape;
        const std::string axes = ", axis=" + python_tuple(dims) + ")";
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
        if (implementation_base == "torch.index_select" && at.size() == 2 && at[0] != nullptr &&
            at[1] != nullptr) {
            return define("jnp.take(" + name(0) + ", " + name(1) + ", axis=-1)");
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
        if (implementation_base == "torch.Tensor.index_put(tokens)" && at.size() == 4 &&
            at[0] != nullptr && at[1] != nullptr && at[2] != nullptr && at[3] != nullptr &&
            at[0]->shape.size() == 4) {
            // `write_tokens`: token p at (rows[p], positions[p]), a scatter of
            // P x H vectors.
            return define(name(0) + ".at[" + name(2) + ".astype(jnp.int32)[:, None], jnp.arange(" +
                          std::to_string(at[0]->shape[1]) + ")[None, :], " + name(3) +
                          ".astype(jnp.int32)[:, None]].set(" + name(1) +
                          "[0].transpose(1, 0, 2))");
        }
        if (implementation_base == "torch.Tensor.index_put" && (at.size() == 3 || at.size() == 4) &&
            at[0] != nullptr && at[1] != nullptr && at[0]->shape.size() == 4) {
            if (at.size() == 3) {
                // `write_rows`: row b at at[b], a scatter of B x H vectors.
                const Dims& cache = at[0]->shape;
                return define(name(0) + ".at[jnp.arange(" + std::to_string(cache[0]) +
                              ")[:, None], jnp.arange(" + std::to_string(cache[1]) +
                              ")[None, :], " + name(2) + ".astype(jnp.int32)[:, None]].set(" +
                              name(1) + "[:, :, 0])");
            }
            if (at[2] != nullptr && !at[2]->shape.empty()) {
                // `write_slots`: a span in each of several rows, a scatter.
                return define(name(0) + ".at[" + name(2) + ".astype(jnp.int32)[:, None, None], " +
                              "jnp.arange(" + std::to_string(at[0]->shape[1]) +
                              ")[None, :, None], (" + name(3) + ".astype(jnp.int32) + jnp.arange(" +
                              std::to_string(at[1]->shape[2]) + "))[None, None, :]].set(" +
                              name(1) + ")");
            }
            // `write_slot`: one row's span, a slice write.
            return define("jax.lax.dynamic_update_slice(" + name(0) + ", " + name(1) + ", (" +
                          name(2) + ".astype(jnp.int32), jnp.int32(0), " + name(3) +
                          ".astype(jnp.int32), jnp.int32(0)))");
        }
        if (implementation_base == "torch.matmul" && at.size() == 2) {
            return define("jnp.matmul(" + name(0) + ", " + name(1) + ")");
        }
        if (is_convolution(implementation_base) && at.size() == 3 && at[0] != nullptr &&
            at[1] != nullptr) {
            const auto window = conv_window(implementation_base);
            if (!window) {
                return std::nullopt;
            }
            const bool flat = window->strides.size() == 1;
            std::string strides = "(";
            std::string padding = "(";
            for (std::size_t i = 0; i < window->strides.size(); ++i) {
                const std::string p = std::to_string(window->pads[i]);
                strides += std::to_string(window->strides[i]);
                strides += ", ";
                padding += "(";
                padding += p;
                padding += ", ";
                padding += p;
                padding += "), ";
            }
            const std::string layout =
                flat ? "(\"NCH\", \"OIH\", \"NCH\")" : "(\"NCHW\", \"OIHW\", \"NCHW\")";
            const std::string mixed =
                define("jax.lax.conv_general_dilated(" + name(0) + ", " + name(1) + ", " + strides +
                       "), " + padding + "), dimension_numbers=" + layout + ")");
            return at[2] != nullptr
                       ? define(mixed + " + " + name(2) +
                                (flat ? ".reshape((1, -1, 1))" : ".reshape((1, -1, 1, 1))"))
                       : mixed;
        }
        if (implementation_base == "torch.nn.functional.max_pool2d" && at.size() == 1 &&
            at[0] != nullptr) {
            const auto window = call_generic("K");
            const auto stride = call_generic("Stride");
            const auto pad = call_generic("Pad");
            if (!window || !stride || !pad) {
                return std::nullopt;
            }
            // Padding takes the initial value, minus infinity, which no
            // window's maximum is.
            const std::string k = std::to_string(*window);
            const std::string s = std::to_string(*stride);
            const std::string p = std::to_string(*pad);
            return define("jax.lax.reduce_window(" + name(0) + ", jnp.array(-jnp.inf, " + name(0) +
                          ".dtype), jax.lax.max, (1, 1, " + k + ", " + k + "), (1, 1, " + s + ", " +
                          s + "), ((0, 0), (0, 0), (" + p + ", " + p + "), (" + p + ", " + p +
                          ")))");
        }
        if (implementation_base == "torch.nn.functional.linear" && at.size() == 3 &&
            at[0] != nullptr && at[1] != nullptr) {
            std::string product = define(name(0) + " @ " + name(1) + ".T");
            if (at[2] != nullptr) {
                product = define(product + " + " + name(2));
            }
            const auto path = lora_target(name(1));
            if (!path || at[1]->shape.size() != 2) {
                return product;
            }
            // A low-rank adapter beside the weight (LoRA): `x @ A.T @ B.T`,
            // scaled by alpha / rank.
            const auto [a, b] = adapters(*path);
            const std::string low = define("(" + name(0) + " @ " + a + ".T) @ " + b + ".T");
            return define(product + " + " + low + " * " +
                          python_float(lora_alpha_ / static_cast<double>(lora_rank_)));
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
            const bool masked = at[4] != nullptr && !causal_masks_.contains(at[4]->name);
            if (at[4] != nullptr && !masked) {
                mask = ", is_causal=True";
            } else if (masked) {
                // A shared [Q, K] mask, or one per sequence ([B, Q, K]).
                mask =
                    ", mask=" + name(4) + (at[4]->shape.size() == 3 ? "[:, None]" : "[None, None]");
            }
            const std::string kernel =
                ", implementation=_attention_kernel(" + q + (masked ? ", masked=True" : "") + ")";
            const std::string mixed = define("jax.nn.dot_product_attention(" + q + ", " + k + ", " +
                                             v + ", scale=" + scalar(3) + mask + kernel + ")");
            return define(back("jnp.swapaxes(" + mixed + ", 1, 2)", 0));
        }
        // The output head and its loss a block of rows at a time
        // (`linnet.jax.loss`), never the whole [N, V] logits.
        const bool cross_entropy = implementation_base == "linnet.linear_cross_entropy";
        const bool log_probs = implementation_base == "linnet.linear_token_log_probs";
        if ((cross_entropy || log_probs) && at.size() == (cross_entropy ? 4U : 3U) &&
            std::ranges::none_of(at,
                                 [](const TensorInfo* operand) { return operand == nullptr; })) {
            uses_loss_ = true;
            std::string call =
                std::string("_loss.") +
                (cross_entropy ? "linear_cross_entropy(" : "linear_token_log_probs(") + name(0) +
                ", " + name(1) + ", " + name(2);
            if (cross_entropy) {
                call += ", " + name(3);
            }
            return define(call + ")");
        }
        // MXFP4 experts: the chosen experts' products by `linnet.jax.moe`,
        // over a 16-bit copy of the experts made by its own statement, which
        // reads only weights: `prepare` makes it once and every entry shares
        // it.
        const bool grouped = implementation_base == "linnet.mxfp4_grouped(shared)" ||
                             implementation_base == "linnet.mxfp4_grouped(combine)";
        const bool chosen = implementation_base == "linnet.mxfp4_experts" ||
                            implementation_base == "linnet.mxfp4_experts(shared)";
        if ((grouped || chosen) && at.size() >= 4 && at[0] != nullptr && at[1] != nullptr &&
            at[2] != nullptr && at[3] != nullptr) {
            const bool combine = implementation_base == "linnet.mxfp4_grouped(combine)";
            if (combine && (at.size() != 5 || at[4] == nullptr)) {
                return std::nullopt;
            }
            uses_experts_ = true;
            const std::string weight = define("_moe.mxfp4_weight(" + name(1) + ", " + name(2) +
                                              ", " + jnp_dtype(dtype) + ")");
            if (combine) {
                return define("_moe.experts_combined(" + name(0) + ", " + weight + ", " + name(3) +
                              ", " + name(4) + ")");
            }
            // `mxfp4_experts_shared` takes the row's input as [R, 1, In].
            const std::string x = implementation_base == "linnet.mxfp4_experts(shared)"
                                      ? name(0) + "[:, 0]"
                                      : name(0);
            const bool shared = implementation_base.ends_with("(shared)");
            return define("_moe.experts(" + x + ", " + weight + ", " + name(3) + ", " +
                          (shared ? "True" : "False") + ")");
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

    bool supports_fully_shard() const override { return true; }

    // A weight's part (this device's, under `shard_map`) gathered whole in
    // the dtype the entry computes in (`linnet.jax.fsdp.gather`); outside a
    // mesh the weight is already whole and only cast.
    std::string gather(const TensorInfo& value) override {
        uses_gather_ = true;
        const std::string name = define("_gather(" + value.name + ", " + python_tuple(value.shape) +
                                        ", " + jnp_dtype(value.dtype) + ")");
        gathered_.emplace(name, value.name);
        if (value.name.size() > 1 && value.name[0] == 'p') {
            const std::size_t index = std::stoul(value.name.substr(1));
            if (index < parameters_.size() &&
                std::ranges::find(gathered_paths_, parameters_[index]) == gathered_paths_.end()) {
                gathered_paths_.push_back(parameters_[index]);
            }
        }
        return name;
    }

    bool supports_remat() const override { return true; }

    void begin_remat() override {
        body_ += indent_ + "# remat " + std::to_string(remats_) + " begin\n";
    }

    void end_remat() override {
        body_ += indent_ + "# remat " + std::to_string(remats_++) + " end\n";
    }

    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        std::string out =
            "# " + entry_label(block_name, entry_name) + " from module " + module_path +
            ", generated by `linnet jax` for one shape\n"
            "# binding. `main` takes the entry's inputs, then the parameters in\n"
            "# PARAMETERS order, then the states in STATES order; it returns the\n"
            "# entry's RESULTS results followed by the states in NEXT_STATES\n"
            "# order. Linnet's `i64` needs 64-bit integers enabled.\n"
            "import jax\n"
            "import jax.numpy as jnp\n\n"
            "jax.config.update(\"jax_enable_x64\", True)\n\n"
            "\n"
            "def _attention_kernel(query, masked=False):\n"
            "    # cuDNN's fused attention on a GPU for 16-bit inputs and heads\n"
            "    # of at most 256 (a multiple of 8), XLA's own elsewhere (CPU,\n"
            "    # f32, the 512-wide heads of a VAE), where cuDNN has no kernel.\n"
            "    # With an explicit mask, cuDNN only for 16 queries a sequence or\n"
            "    # more (a prompt): XLA fuses a decoding step's one query better,\n"
            "    # except in f16, where XLA's attention rejects a single query\n"
            "    # (`Unsupported dot precision algorithm`).\n"
            "    width = query.shape[-1]\n"
            "    if (\n"
            "        jax.default_backend() == \"gpu\"\n"
            "        and query.dtype in (jnp.bfloat16, jnp.float16)\n"
            "        and width <= 256\n"
            "        and width % 8 == 0\n"
            "        and (not masked or query.shape[1] >= 16 or query.dtype == jnp.float16)\n"
            "    ):\n"
            "        return \"cudnn\"\n"
            "    return None\n\n";
        if (uses_loss_) {
            out += loss_import;
        }
        if (uses_experts_) {
            out += experts_import;
        }
        if (uses_gather_) {
            out += gather_import;
        }
        out += "PARAMETERS = " + string_list(parameters_) + "\n";
        // The parameters the code gathers from their parts itself.
        out += "GATHERED = " + string_list(gathered_paths_) + "\n";
        out += "STATES = " + string_list(states_) + "\n";
        std::vector<std::string> next_states;
        next_states.reserve(states.size());
        for (const auto& [path, value] : states) {
            next_states.push_back(path);
        }
        out += "NEXT_STATES = " + string_list(next_states) + "\n";
        out += "RESULTS = " + std::to_string(results.size()) + "\n";
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
        tail += outputs.size() == 1 ? ",)\n" : ")\n";
        // Weight-only work, run once per loaded model (`--prepare`).
        PreparedSplit prepared;
        prepared.body = prune_python_assignments(body_, tail);
        if (prepare_) {
            prepared = split_prepared(prepared.body, tail, parameters_, "");
        }
        if (remats_ > 0) {
            prepared.body = outline_remat(prepared.body, tail);
        }
        if (!prepared.outputs.empty()) {
            out += "PREPARED = " + string_list(prepared.keys) + "\n";
            out += "PREPARE_INPUTS = " + string_list(prepared.inputs) + "\n\n\n";
            out += "def prepare(";
            for (std::size_t i = 0; i < prepared.inputs.size(); ++i) {
                out += (i == 0 ? "" : ", ") + prepared.inputs[i];
            }
            std::string returned = "    return (";
            for (std::size_t i = 0; i < prepared.outputs.size(); ++i) {
                returned += (i == 0 ? "" : ", ") + prepared.outputs[i];
            }
            returned += prepared.outputs.size() == 1 ? ",)\n" : ")\n";
            out += "):\n" + precise(prepared.prepare + returned);
        }
        out += "\n\ndef main(";
        std::vector<std::string> arguments = arguments_;
        arguments.insert(arguments.end(), prepared.outputs.begin(), prepared.outputs.end());
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            out += (i == 0 ? "" : ", ") + arguments[i];
        }
        out += "):\n" + precise(prepared.body + tail);
        return out;
    }

private:
    // A function body run with f32 products (`@`, `einsum`, convolutions,
    // attention) at full precision, when the numerics are not `fast`: XLA's
    // default on an NVIDIA GPU is TF32. The setting is read while tracing,
    // so it holds for everything the body calls and nothing outside it.
    std::string precise(const std::string& body) const {
        if (!full_precision_) {
            return body;
        }
        std::string out = "    with jax.default_matmul_precision(\"highest\"):\n";
        std::size_t start = 0;
        while (start < body.size()) {
            std::size_t end = body.find('\n', start);
            end = end == std::string::npos ? body.size() : end + 1;
            const std::string_view line{body.data() + start, end - start};
            out += (line == "\n" ? "" : "    ") + std::string(line);
            start = end;
        }
        return out;
    }

    bool prepare_ = false;          // split weight-only work into `prepare`
    bool full_precision_ = false;   // f32 products at full precision (`precise`)
    std::vector<std::string> lora_; // weights given low-rank adapters (`--lora`)
    std::int64_t lora_rank_ = 0;
    double lora_alpha_ = 0.0;
    std::map<std::string, std::pair<std::string, std::string>> adapters_; // weight -> a, b
    bool uses_experts_ = false;                   // the module imports `linnet.jax.moe`
    bool uses_loss_ = false;                      // the module imports `linnet.jax.loss`
    bool uses_gather_ = false;                    // the module imports `linnet.jax.fsdp`
    std::vector<std::string> gathered_paths_;     // parameters gathered (`GATHERED`)
    std::map<std::string, std::string> gathered_; // gathered value -> its argument
    std::size_t remats_ = 0;                      // recomputed regions so far

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
            return define(sema::is_float(operands[0].dtype) ? a + " / " + b
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
            result = sema::is_float(dtype) ? a / b : std::trunc(a / b);
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
        if (!sema::is_float(dtype)) {
            values_[name] = result;
            literals_[name] = std::to_string(static_cast<long long>(result));
            return;
        }
        if (dtype == ScalarKind::F32) {
            result = static_cast<double>(static_cast<float>(result));
        }
        values_[name] = result;
        literals_[name] = python_float(result);
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
            return sema::is_float(dtype) ? python_float(static_cast<double>(literal.integer))
                                         : std::to_string(literal.integer);
        case Literal::Kind::Real:
            return python_float(literal.real);
        case Literal::Kind::Boolean:
            return literal.integer != 0 ? "True" : "False";
        case Literal::Kind::Lowest:
            if (sema::is_float(dtype)) {
                return "-jnp.inf";
            }
            return dtype == ScalarKind::Bool ? "False" : "jnp.iinfo(" + jnp_dtype(dtype) + ").min";
        case Literal::Kind::Highest:
            if (sema::is_float(dtype)) {
                return "jnp.inf";
            }
            return dtype == ScalarKind::Bool ? "True" : "jnp.iinfo(" + jnp_dtype(dtype) + ").max";
        }
        return "0";
    }

    std::vector<std::string> arguments_;
    std::size_t parameters_end_ = 0; // `arguments_` past the last parameter
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
    if (!options.lora.empty() && (options.prepare || options.lora_rank <= 0)) {
        return std::unexpected(options.prepare
                                   ? "adapters need the weights unprepared (no --prepare)"
                                   : "adapters need a positive rank");
    }
    JaxTarget target(options);
    return export_graph(module, options, target);
}

} // namespace linnet::backend
