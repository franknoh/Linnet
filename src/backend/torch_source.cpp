#include "linnet/backend/torch_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
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
    TorchTarget(bool prepare, bool fuse) : prepare_(prepare), fuse_(fuse) {}

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
        const std::string argument = "p" + std::to_string(parameters_.size());
        parameters_.push_back(path);
        parameter_types_.emplace_back(shape, dtype);
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
        const std::string name = define("torch.tensor(" + text + ", dtype=" + torch_dtype(dtype) +
                                        ", device=" + device() + ")");
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
                      ", dtype=torch.int64, device=" + device() + ")");
    }

    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override {
        (void)shape;
        // `indices[..., k]` selects along axis k of the source: advanced
        // indexing with one index tensor per leading source axis, the
        // trailing ones taken whole.
        std::string index;
        for (std::size_t k = 0; k < static_cast<std::size_t>(indices.shape.back()); ++k) {
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

    bool broadcasts_elementwise() const override { return true; }

    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        ScalarKind dtype) override {
        (void)shape, (void)dtype;
        return define("torch.einsum(\"" + einsum_equation(lhs_axes, rhs_axes, out_axes) + "\", " +
                      lhs.name + ", " + rhs.name + ")");
    }

    // Placement: `_dev` is the tuple of devices `main` receives, one per
    // slot. A transfer is never shared through the expression cache, since
    // an offloaded parameter's copy is deleted when its block returns and
    // the same expression later must make a new one.
    bool supports_placement() const override { return true; }
    void enable_placement(int slots) override {
        placed_ = true;
        slots_ = slots;
    }
    void set_slot(int slot) override { slot_ = slot; }
    std::string transfer(const TensorInfo& value, int slot) override {
        const std::string name = "v" + std::to_string(next_++);
        body_ += indent_ + name + " = " + value.name + ".to(_dev[" + std::to_string(slot) +
                 "], non_blocking=True)\n";
        return name;
    }
    void release(const std::vector<std::string>& names) override {
        std::string line = indent_ + "del ";
        for (std::size_t i = 0; i < names.size(); ++i) {
            line += (i == 0 ? "" : ", ") + names[i];
        }
        body_ += line + "\n";
    }

    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        const std::string suffix = "(input dtype)";
        const bool fast = implementation.ends_with(suffix);
        std::string implementation_base =
            fast ? implementation.substr(0, implementation.size() - suffix.size()) : implementation;
        const std::string gqa = "(enable_gqa)";
        const bool grouped = implementation_base.ends_with(gqa);
        if (grouped) {
            implementation_base.resize(implementation_base.size() - gqa.size());
        }
        // `fast`: the kernel runs in the tensor's dtype; otherwise in f32 as
        // the canonical body does.
        const auto up = [&](const std::string& x) { return fast ? x : x + ".float()"; };
        const auto down = [&](const std::string& expression, const std::string& like) {
            return fast ? expression : expression + ".to(" + like + ".dtype)";
        };
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
        if (implementation_base == "torch.matmul" && operands.size() == 2) {
            return define("torch.matmul(" + name(0) + ", " + name(1) + ")");
        }
        if (implementation_base == "torch.nn.functional.embedding" && operands.size() == 2) {
            return define("F.embedding(" + name(0) + ".long(), " + name(1) + ")");
        }
        if (implementation_base == "torch.index_select" && operands.size() == 2) {
            return define(name(0) + ".index_select(-1, " + name(1) + ".long())");
        }
        if (implementation_base == "torch.tril" && operands.empty() && shape.size() == 2) {
            // `keys[k] <= queries[q] + (K - Q)`: ones below the diagonal K - Q.
            std::string mask =
                define("torch.ones(" + dims_text(shape) + ", dtype=torch.bool, device=" + device() +
                       ").tril(" + std::to_string(shape[1] - shape[0]) + ")");
            if (shape[0] == shape[1]) {
                causal_masks_.insert(mask);
            }
            return mask;
        }
        if (implementation_base == "torch.Tensor.index_copy" && operands.size() == 3) {
            std::vector<const TensorInfo*> at;
            at.reserve(operands.size());
            for (const std::optional<TensorInfo>& operand : operands) {
                at.push_back(operand.has_value() ? &*operand : nullptr);
            }
            if (at[0] == nullptr || at[1] == nullptr || at[2] == nullptr) {
                return std::nullopt;
            }
            // A slice write, not a pass over the cache: one position when
            // decoding, the whole prompt's span when prefilling.
            const Dims& value = at[1]->shape;
            const std::int64_t span = value.size() == 4 ? value[2] : 1;
            if (span == 1) {
                return define(name(0) + ".index_copy(2, " + name(2) + ".reshape(1).long(), " +
                              name(1) + ")");
            }
            return define(name(0) + ".index_copy(2, " + name(2) + ".reshape(1).long() + " +
                          "torch.arange(" + std::to_string(span) + ", device=" + name(0) +
                          ".device), " + name(1) + ")");
        }
        if ((implementation_base == "linnet.mxfp4_experts" ||
             implementation_base == "linnet.mxfp4_experts(shared)") &&
            operands.size() == 4 && operands[0] && operands[1] && operands[2] && operands[3] &&
            shape.size() == 3) {
            // A shared input ([R, 1, In]) is the same row for each chosen slot:
            // a view.
            mxfp4_helper_ = true;
            const std::string x =
                implementation_base == "linnet.mxfp4_experts"
                    ? name(0)
                    : name(0) + ".expand(-1, " + std::to_string(shape[1]) + ", -1)";
            return define("_mxfp4_experts(" + x + ", " + name(1) + ", " + name(2) + ", " + name(3) +
                          ")");
        }
        if ((implementation_base == "linnet.mxfp4_grouped(shared)" && operands.size() == 4) ||
            (implementation_base == "linnet.mxfp4_grouped(combine)" && operands.size() == 5)) {
            mxfp4_helper_ = true;
            mxfp4_grouped_helper_ = true;
            std::string call =
                "_mxfp4_grouped(" + name(0) + ", " + name(1) + ", " + name(2) + ", " + name(3);
            if (operands.size() == 5) {
                call += ", " + name(4);
            }
            return define(call + ")");
        }
        if (implementation_base == "linnet.linear_cross_entropy" && operands.size() == 4 &&
            operands[0] && operands[1] && operands[2] && operands[3]) {
            loss_helper_ = true;
            return define("_linear_cross_entropy(" + name(0) + ", " + name(1) + ", " + name(2) +
                          ", " + name(3) + ")");
        }
        if (implementation_base == "linnet.linear_token_log_probs" && operands.size() == 3 &&
            operands[0] && operands[1] && operands[2]) {
            loss_helper_ = true;
            return define("_linear_token_log_probs(" + name(0) + ", " + name(1) + ", " + name(2) +
                          ")");
        }
        if (implementation_base == "torch.distributed.all_reduce" && operands.size() == 1 &&
            operands[0]) {
            shards_helper_ = true;
            return define("_all_reduce(" + name(0) + ")");
        }
        if (implementation_base == "torch.distributed.all_gather" && operands.size() == 1 &&
            operands[0]) {
            shards_helper_ = true;
            return define("_all_gather(" + name(0) + ")");
        }
        if (implementation_base == "torch.Tensor.index_put(tokens)" && operands.size() == 4 &&
            operands[0] && operands[1] && operands[2] && operands[3]) {
            // `write_tokens`: token p at (rows[p], positions[p]), every head;
            // the values [1, H, P, D] as [P, H, D].
            return define("torch.ops.aten.index_put(" + name(0) + ", [" + name(2) +
                          ".long(), None, " + name(3) + ".long()], " + name(1) +
                          "[0].permute(1, 0, 2))");
        }
        if (implementation_base == "torch.Tensor.index_put" &&
            (operands.size() == 3 || operands.size() == 4)) {
            std::vector<const TensorInfo*> at;
            at.reserve(operands.size());
            for (const std::optional<TensorInfo>& operand : operands) {
                at.push_back(operand.has_value() ? &*operand : nullptr);
            }
            if (at[0] == nullptr || at[1] == nullptr || at[2] == nullptr) {
                return std::nullopt;
            }
            // Rows and positions as index tensors, the heads as a full slice
            // (`None`): one kernel over the positions it changes, and a cache
            // split by heads across devices (a DTensor) keeps its split.
            const Dims& cache = at[0]->shape;
            const Dims& value = at[1]->shape;
            if (cache.size() != 4 || value.size() != 4) {
                return std::nullopt;
            }
            const std::string device = name(0) + ".device";
            const std::string put = "torch.ops.aten.index_put(" + name(0) + ", [";
            if (operands.size() == 3) {
                // `write_rows`: row b at at[b]; the values are [B, H, D].
                return define(put + "torch.arange(" + std::to_string(cache[0]) +
                              ", device=" + device + "), None, " + name(2) + ".long()], " +
                              name(1) + "[:, :, 0])");
            }
            const std::string span = "(" + name(3) + ".long() + torch.arange(" +
                                     std::to_string(value[2]) + ", device=" + device +
                                     "))[None, :]";
            // `write_slots` (rows `slots`) or `write_slot` (one row): [M, N]
            // indices, the values [M, N, H, D].
            const std::string rows = at[2]->shape.empty() ? name(2) + ".long().reshape(1, 1)"
                                                          : name(2) + ".long()[:, None]";
            return define(put + rows + ", None, " + span + "], " + name(1) +
                          ".permute(0, 2, 1, 3))");
        }
        if (implementation_base == "torch.nn.functional.group_norm" && operands.size() == 4 &&
            operands[0] && operands[1] && operands[2] && operands[3]) {
            // The group count is the call's own `Groups`.
            const auto groups = call_generic("Groups");
            if (!groups) {
                return std::nullopt;
            }
            return define("F.group_norm(" + name(0) + ", " + std::to_string(*groups) + ", " +
                          name(1) + ", " + name(2) + ", " + scalar(3) + ")");
        }
        if (is_convolution(implementation_base) && operands.size() == 3 && operands[0] &&
            operands[1]) {
            // `F.conv1d`/`F.conv2d` take the window geometry as arguments. It
            // comes from the call's own generics: recovering it from the
            // shapes is ambiguous (a 3x3 window taking 4 positions to 2 fits
            // stride 1 without padding and stride 2 with one), and a wrong
            // guess is a silently wrong answer. Without them, the body runs.
            const auto window = conv_window(implementation_base);
            if (!window) {
                return std::nullopt;
            }
            // One number when every axis agrees, as for a square window.
            const auto tuple = [](const std::vector<std::int64_t>& values) {
                if (std::adjacent_find(values.begin(), values.end(), std::not_equal_to<>()) ==
                    values.end()) {
                    return std::to_string(values.front());
                }
                std::string text = "(";
                for (std::size_t i = 0; i < values.size(); ++i) {
                    text += i == 0 ? "" : ", ";
                    text += std::to_string(values[i]);
                }
                return text + (values.size() == 1 ? ",)" : ")");
            };
            const char* function = window->strides.size() == 1 ? "F.conv1d(" : "F.conv2d(";
            return define(function + name(0) + ", " + name(1) + ", " + name(2) + ", stride=" +
                          tuple(window->strides) + ", padding=" + tuple(window->pads) + ")");
        }
        if (implementation_base == "torch.ops.aten._weight_int4pack_mm" && operands.size() == 5 &&
            operands[0] && operands[1] && operands[2] && operands[3]) {
            // Packing reads only weights, so `--prepare` does it once; the
            // helpers pick tinygemm or Linnet's kernel, by the rows, where
            // they run (CUDA, bf16, tinygemm's group sizes) and the
            // dequantized weight elsewhere.
            int4_helpers_ = true;
            const std::string packed =
                define("_int4_pack(" + name(1) + ", " + name(2) + ", " + name(3) + ")");
            return define("_int4_linear(" + name(0) + ", " + packed + ", " + name(4) + ", " +
                          name(1) + ", " + name(2) + ", " + name(3) + ")");
        }
        if (implementation_base == "torch._grouped_mm" && operands.size() == 3 && operands[0] &&
            operands[1] && operands[2]) {
            experts_helper_ = true;
            return define("_linear_experts(" + name(0) + ", " + name(1) + ", " + name(2) + ")");
        }
        if (implementation_base == "torch._grouped_mm(shared)" && operands.size() == 3 &&
            operands[0] && operands[1] && operands[2]) {
            experts_helper_ = true;
            return define("_linear_experts_shared(" + name(0) + ", " + name(1) + ", " + name(2) +
                          ")");
        }
        if (implementation_base == "torch._grouped_mm(combined)" && operands.size() == 4 &&
            operands[0] && operands[1] && operands[2] && operands[3]) {
            experts_helper_ = true;
            return define("_combine_experts(" + name(0) + ", " + name(1) + ", " + name(2) + ", " +
                          name(3) + ")");
        }
        if (implementation_base == "torch.nn.functional.max_pool2d" && operands.size() == 1 &&
            operands[0]) {
            // The geometry is the call's own, as for the convolution;
            // `F.max_pool2d` pads with minus infinity and takes at most half
            // a window of it.
            const auto window = call_generic("K");
            const auto stride = call_generic("Stride");
            const auto pad = call_generic("Pad");
            if (!window || !stride || !pad || 2 * *pad > *window) {
                return std::nullopt;
            }
            return define("F.max_pool2d(" + name(0) + ", " + std::to_string(*window) + ", stride=" +
                          std::to_string(*stride) + ", padding=" + std::to_string(*pad) + ")");
        }
        if (implementation_base == "torch.nn.functional.interpolate(nearest)" &&
            operands.size() == 1 && shape.size() == 4) {
            // Like the convolution, the scale is recovered from the shapes.
            const TensorInfo* source = nullptr;
            for (const std::optional<TensorInfo>& operand : operands) {
                source = operand.has_value() ? &*operand : nullptr;
            }
            if (source == nullptr || source->shape.size() != 4 || source->shape[2] <= 0 ||
                source->shape[3] <= 0) {
                return std::nullopt;
            }
            const Dims& input = source->shape;
            if (shape[2] % input[2] != 0 || shape[3] % input[3] != 0 ||
                shape[2] / input[2] != shape[3] / input[3]) {
                return std::nullopt;
            }
            const std::int64_t scale = shape[2] / input[2];
            return define("F.interpolate(" + name(0) + ", scale_factor=" + std::to_string(scale) +
                          ", mode=\"nearest\")");
        }
        if (implementation_base == "torch.nn.functional.batch_norm" && operands.size() == 6 &&
            operands[0] && operands[1] && operands[2] && operands[3] && operands[4]) {
            return define("F.batch_norm(" + name(0) + ", " + name(1) + ", " + name(2) + ", " +
                          name(3) + ", " + name(4) + ", False, 0.0, " + scalar(5) + ")");
        }
        if (implementation_base == "torch.Tensor.mean" && operands.size() == 1 && operands[0]) {
            return define(name(0) + ".mean(dim=(2, 3), dtype=torch.float32).to(" + name(0) +
                          ".dtype)");
        }
        if (implementation_base == "torch.nn.functional.linear" && operands.size() == 3) {
            return define("F.linear(" + name(0) + ", " + name(1) + ", " + name(2) + ")");
        }
        // `torch.softmax` and `torch.rms_norm` accumulate in f32 for f16 and
        // bf16 inputs themselves, so their results are bit-identical to the
        // canonical body's explicit casts: no `.float()` in any tier.
        if (implementation_base == "torch.softmax" && operands.size() == 1) {
            return define("torch.softmax(" + name(0) + ", dim=-1)");
        }
        if (implementation_base == "torch.relu" && operands.size() == 1) {
            return define("torch.relu(" + name(0) + ")");
        }
        if (implementation_base == "torch.sigmoid" && operands.size() == 1) {
            return define("torch.sigmoid(" + name(0) + ")");
        }
        if (implementation_base == "torch.nn.functional.silu" && operands.size() == 1) {
            return define("F.silu(" + name(0) + ")");
        }
        if (implementation_base == "torch.nn.functional.gelu" && operands.size() == 1) {
            return define("F.gelu(" + name(0) + ", approximate=\"none\")");
        }
        if (implementation_base == "torch.nn.functional.gelu(tanh)" && operands.size() == 1) {
            return define("F.gelu(" + name(0) + ", approximate=\"tanh\")");
        }
        // The normalized width is the operand's last axis.
        const auto width_of = [&](std::size_t i) -> std::string {
            const std::optional<TensorInfo>& operand = operands[i];
            return operand.has_value() && !operand->shape.empty()
                       ? std::to_string(operand->shape.back())
                       : "1";
        };
        if (implementation_base == "torch.rms_norm" && operands.size() == 3 && operands[0]) {
            const std::string width = width_of(0);
            return define("torch.rms_norm(" + name(0) + ", [" + width + "], eps=" + scalar(2) +
                          ") * " + name(1));
        }
        if (implementation_base == "torch.nn.functional.layer_norm" && operands.size() == 4 &&
            operands[0]) {
            const std::string width = width_of(0);
            const std::string scaled = define(
                down("F.layer_norm(" + up(name(0)) + ", [" + width + "], eps=" + scalar(3) + ")",
                     name(0)) +
                " * " + name(1));
            return operands[2] ? define(scaled + " + " + name(2)) : scaled;
        }
        if (implementation_base == "linnet.sink_attention" && operands.size() == 6 && operands[0] &&
            operands[1] && operands[5]) {
            // FlexAttention under `torch.compile` where its blocks pay off, as
            // for `_attend` -- the sink folded in from its log-sum-exp -- and
            // otherwise two products around the softmax with the sink in it.
            sink_helper_ = true;
            flex_helpers_ = true;
            const std::optional<TensorInfo>& query_operand = operands[0];
            const std::optional<TensorInfo>& key_operand = operands[1];
            const std::optional<TensorInfo>& mask_operand = operands[5];
            if (!query_operand || !key_operand || !mask_operand) {
                return std::nullopt;
            }
            const Dims& query = query_operand->shape;
            const Dims& key = key_operand->shape;
            const Dims& mask = mask_operand->shape;
            std::string blocks = "None";
            const std::int64_t block_q = 128;
            std::int64_t block_k = 128;
            if (query.size() == 4 && key.size() == 4 && mask.size() == 3 && key[1] > 0 &&
                query[1] % key[1] == 0 && query[3] >= 16 && query[3] <= 256) {
                const std::int64_t queries = query[2];
                const std::int64_t keys = key[2];
                block_k = queries == 1 ? 64 : 128;
                const std::int64_t group = query[1] / key[1];
                const bool whole_group = (group & (group - 1)) == 0;
                // As for `_attend`: many queries only under one mask for the
                // whole batch.
                if (keys % block_k == 0 && (queries <= block_q || queries % block_q == 0) &&
                    (queries > 1 ? mask[0] == 1 : mask[0] > 1 && whole_group)) {
                    const std::string sizes =
                        std::to_string(queries) + ", " + std::to_string(keys) + ", " +
                        std::to_string(block_q) + ", " + std::to_string(block_k);
                    const std::string key_of = name(5) + "/" + sizes;
                    auto found = flex_blocks_.find(key_of);
                    if (found == flex_blocks_.end()) {
                        found = flex_blocks_
                                    .emplace(key_of,
                                             define("_flex_blocks(" + name(5) + ", " + sizes + ")"))
                                    .first;
                    }
                    blocks = found->second;
                }
            }
            return define("_sink_attend(" + name(0) + ", " + name(1) + ", " + name(2) + ", " +
                          name(3) + ", " + name(5) + ", " + scalar(4) + ", " + blocks + ", " +
                          std::to_string(block_q) + ", " + std::to_string(block_k) + ")");
        }
        if (implementation_base == "torch.nn.functional.scaled_dot_product_attention" &&
            operands.size() == 5) {
            // A square causal mask becomes `is_causal=True`, which lets PyTorch
            // pick its fused causal kernels; other masks are passed as they are.
            std::vector<const TensorInfo*> at;
            at.reserve(operands.size());
            for (const std::optional<TensorInfo>& operand : operands) {
                at.push_back(operand.has_value() ? &*operand : nullptr);
            }
            const TensorInfo* query = at[0];
            const TensorInfo* key = at[1];
            const TensorInfo* attn_mask = at[4];
            // A mask per sequence ([B, Q, K]) broadcasts over the heads.
            std::string mask = ", attn_mask=" + name(4);
            if (attn_mask != nullptr && attn_mask->shape.size() == 3) {
                mask += ".unsqueeze(1)";
            }
            if (attn_mask != nullptr && causal_masks_.contains(attn_mask->name)) {
                mask = ", is_causal=True";
            } else if (attn_mask == nullptr) {
                mask = "";
            }
            const bool heads_differ = query != nullptr && key != nullptr &&
                                      query->shape.size() > 1 && key->shape.size() > 1 &&
                                      key->shape[1] != query->shape[1];
            // Fast numerics with an explicit mask: `_attend`, which under
            // `torch.compile` on CUDA runs FlexAttention over only the key
            // blocks the mask reaches -- a serving step's rows at their own
            // lengths, packed prompts that each see only themselves, a
            // sliding window -- and otherwise the arithmetic below. The
            // blocks are worked out once per mask and shared by every layer.
            // FlexAttention takes heads 16 to 256 wide.
            if (fast && query != nullptr && key != nullptr && attn_mask != nullptr &&
                !causal_masks_.contains(attn_mask->name) && query->shape.size() == 4 &&
                key->shape.size() == 4 && key->shape[1] > 0 &&
                query->shape[1] % key->shape[1] == 0 && query->shape[3] >= 16 &&
                query->shape[3] <= 256 &&
                (attn_mask->shape.size() == 2 || attn_mask->shape.size() == 3)) {
                const std::int64_t queries = query->shape[2];
                const std::int64_t keys = key->shape[2];
                const std::int64_t block_q = 128;
                const std::int64_t block_k = queries == 1 ? 64 : 128;
                // One query per row pays off across many rows at their own
                // lengths (a serving step); one request's step has no blocks
                // to skip, and the two products below launch less.
                const bool rows_differ = attn_mask->shape.size() == 3 && attn_mask->shape[0] > 1;
                // FlexAttention's decoding kernel is wrong (errors near 0.4 for
                // Qwen2.5's seven query heads a key head, PyTorch 2.14) unless
                // the query heads of a key head are a power of two.
                const std::int64_t group = query->shape[1] / key->shape[1];
                const bool whole_group = (group & (group - 1)) == 0;
                // Many queries take FlexAttention only under one mask for the
                // whole batch: with a mask per sequence, block lists computed
                // in the same compiled graph as the kernel give wrong results
                // (PyTorch 2.14), where FlexAttention's decoding kernel is right.
                if (keys % block_k == 0 && (queries <= block_q || queries % block_q == 0) &&
                    (queries > 1 ? !rows_differ : rows_differ && whole_group)) {
                    flex_helpers_ = true;
                    const std::string sizes =
                        std::to_string(queries) + ", " + std::to_string(keys) + ", " +
                        std::to_string(block_q) + ", " + std::to_string(block_k);
                    const std::string key_of = attn_mask->name + "/" + sizes;
                    auto found = flex_blocks_.find(key_of);
                    if (found == flex_blocks_.end()) {
                        found = flex_blocks_
                                    .emplace(key_of,
                                             define("_flex_blocks(" + attn_mask->name + ", " +
                                                    sizes + ")"))
                                    .first;
                    }
                    return define("_attend(" + name(0) + ", " + name(1) + ", " + name(2) + ", " +
                                  attn_mask->name + ", " + scalar(3) + ", " + found->second + ", " +
                                  std::to_string(block_q) + ", " + std::to_string(block_k) + ")");
                }
            }
            // One query per sequence against a masked cache -- a decoding
            // step -- as two batched matrix products around a softmax. PyTorch
            // sends a masked single query to its memory-efficient kernel, which
            // does not split the keys across blocks: 5x slower for 64 rows of
            // 640 cached positions. Grouped key/value heads are a reshape of
            // the query, never a copy of the cache.
            if (query != nullptr && key != nullptr && attn_mask != nullptr &&
                !causal_masks_.contains(attn_mask->name) && query->shape.size() == 4 &&
                key->shape.size() == 4 && query->shape[2] == 1 && key->shape[1] > 0 &&
                query->shape[1] % key->shape[1] == 0) {
                const Dims& q = query->shape;
                const Dims& k = key->shape;
                const std::string heads = std::to_string(k[1]);
                const std::string group = std::to_string(q[1] / k[1]);
                const std::string width = std::to_string(q[3]);
                const std::string rows = attn_mask->shape.size() == 3
                                             ? std::to_string(attn_mask->shape[0])
                                             : std::string("1");
                const std::string keys = std::to_string(k[2]);
                const std::string grouped_query =
                    define(up(name(0)) + ".reshape(" + std::to_string(q[0]) + ", " + heads + ", " +
                           group + ", " + width + ")");
                const std::string scores =
                    define("torch.matmul(" + grouped_query + ", " + up(name(1)) +
                           ".transpose(-1, -2)).float() * " + scalar(3));
                const std::string masked =
                    define(scores + ".masked_fill(~" + name(4) + ".reshape(" + rows + ", 1, 1, " +
                           keys + "), -1e30)");
                const std::string weights =
                    define("torch.softmax(" + masked + ", dim=-1).to(" + up(name(2)) + ".dtype)");
                return define(down("torch.matmul(" + weights + ", " + up(name(2)) + ").reshape(" +
                                       std::to_string(q[0]) + ", " + std::to_string(q[1]) +
                                       ", 1, " + width + ")",
                                   name(0)));
            }
            const std::string groups = grouped && heads_differ ? ", enable_gqa=True" : "";
            return define(down("F.scaled_dot_product_attention(" + up(name(0)) + ", " +
                                   up(name(1)) + ", " + up(name(2)) + mask +
                                   ", scale=" + scalar(3) + groups + ")",
                               name(0)));
        }
        return std::nullopt;
    }

    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        std::string out = "# " + entry_label(block_name, entry_name) + " from module " +
                          module_path +
                          ", generated by `linnet torch` for one shape\n"
                          "# binding. `main` takes the entry's inputs, then the parameters in\n"
                          "# PARAMETERS order, then the states in STATES order, then the values\n"
                          "# `constants(device)` returns (input-independent tensors such as\n"
                          "# rotary tables, computed once per shape); it returns the entry's\n"
                          "# RESULTS results followed by the states in NEXT_STATES order.\n"
                          "import torch\n"
                          "import torch.nn.functional as F\n\n";
        if (int4_helpers_) {
            out += int4_helpers();
        }
        if (experts_helper_) {
            out += experts_helper();
        }
        if (flex_helpers_) {
            out += flex_helpers();
        }
        if (sink_helper_) {
            out += sink_helper();
        }
        if (shards_helper_) {
            out += shards_helper();
        }
        if (mxfp4_helper_) {
            out += mxfp4_helper();
        }
        if (mxfp4_grouped_helper_) {
            out += mxfp4_grouped_helper();
        }
        if (loss_helper_) {
            out += loss_helper();
        }
        out += "PARAMETERS = " + string_list(parameters_) + "\n";
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
        Hoisted hoisted = hoist(prune_python_assignments(body_, tail), tail);
        // Sibling linear layers become one product only where their joined
        // weight is prepared once, not rebuilt every call.
        std::vector<std::vector<std::string>> fused;
        if (prepare_ && fuse_ && !placed_) {
            hoisted.body = fuse_linears(hoisted.body, fused);
        }
        if (!fused.empty()) {
            out += adjacent_helper();
        }
        out += "CONSTANTS = " + string_list(hoisted.names) + "\n";
        // Weight-only work, run once per loaded model rather than per call.
        // Streamed (offloaded) weights are not resident, so placement keeps
        // everything in `main`.
        PreparedSplit prepared;
        prepared.body = hoisted.body;
        if (prepare_ && !placed_) {
            prepared = split_prepared(hoisted.body, tail, parameters_, hoisted.constants);
        }
        if (!prepared.outputs.empty()) {
            out += "PREPARED = " + string_list(prepared.keys) + "\n";
            out += "PREPARE_INPUTS = " + string_list(prepared.inputs) + "\n";
        }
        if (!fused.empty()) {
            // Parameters `_adjacent` joins, which the runtime lays out one
            // after another so that the joined weight is a view of them.
            out += "FUSED = [";
            for (std::size_t i = 0; i < fused.size(); ++i) {
                out += (i == 0 ? "" : ", ") + string_list(fused[i]);
            }
            out += "]\n";
        }
        // The states `main` writes into the tensor it was given: CUDA graphs
        // must be told their addresses are fixed, or they are skipped.
        std::vector<std::string> in_place;
        const std::string body =
            write_states_in_place(release_dead_values(prepared.body, tail), tail, in_place);
        std::vector<std::string> in_place_paths;
        in_place_paths.reserve(in_place.size());
        for (const std::string& argument : in_place) {
            in_place_paths.push_back(states_.at(std::stoul(argument.substr(1))));
        }
        out += "IN_PLACE = " + string_list(in_place_paths) + "\n";
        if (placed_) {
            out += "SLOTS = " + std::to_string(slots_) + "\n";
        }
        out += "\n\n";
        if (placed_) {
            out += "def constants(_dev):\n";
            out += "    _device = _dev[0]\n";
        } else {
            out += "def constants(_device):\n";
        }
        out += hoisted.constants;
        out += "    return (";
        for (std::size_t i = 0; i < hoisted.names.size(); ++i) {
            out += (i == 0 ? "" : ", ") + hoisted.names[i];
        }
        out += hoisted.names.size() == 1 ? ",)\n\n\n" : ")\n\n\n";
        if (!prepared.outputs.empty()) {
            // `PREPARE_INPUTS` names parameters (`pN`, PARAMETERS order) and
            // constants (`vN`, from `constants`); `PREPARED` keys each result
            // by its computation, so entries share what they compute alike.
            out += "def prepare(";
            for (const std::string& input : prepared.inputs) {
                out += input + ", ";
            }
            std::string returned = "    return (";
            for (std::size_t i = 0; i < prepared.outputs.size(); ++i) {
                returned += (i == 0 ? "" : ", ") + prepared.outputs[i];
            }
            returned += prepared.outputs.size() == 1 ? ",)\n" : ")\n";
            // Intermediates are released as soon as they are dead: a model's
            // worth of dequantization scratch at once would not fit.
            out +=
                "_device):\n" + release_dead_values(prepared.prepare, returned) + returned + "\n\n";
        }
        out += "def main(";
        std::vector<std::string> arguments = arguments_;
        arguments.insert(arguments.end(), hoisted.names.begin(), hoisted.names.end());
        arguments.insert(arguments.end(), prepared.outputs.begin(), prepared.outputs.end());
        if (placed_) {
            arguments.emplace_back("_dev");
        }
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            out += (i == 0 ? "" : ", ") + arguments[i];
        }
        out += "):\n";
        if (placed_) {
            out += "    _device = _dev[0]\n";
        } else {
            out += "    _device = " +
                   (arguments_.empty() ? std::string("torch.device(\"cpu\")")
                                       : arguments_.front() + ".device") +
                   "\n";
        }
        out += body;
        out += tail;
        return out;
    }

    // A cache write whose cache is a state argument nothing else reads
    // happens in place (`index_copy_`, `index_put_`): the
    // functional form copies the whole cache to change one position per
    // sequence, every layer, every step -- for a batch of long sequences,
    // more memory traffic than the weights. The runtime sees the state
    // come back as the same tensor and keeps it.
    static std::string write_states_in_place(const std::string& body,
                                             const std::string& tail,
                                             std::vector<std::string>& written) {
        std::vector<std::string> lines;
        std::string current;
        for (const char c : body) {
            if (c == '\n') {
                lines.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        lines.push_back(tail);
        const auto mentions = [](const std::string& line, const std::string& name) {
            for (std::size_t at = line.find(name); at != std::string::npos;
                 at = line.find(name, at + 1)) {
                const std::size_t end = at + name.size();
                const bool starts =
                    at == 0 || (!std::isalnum(static_cast<unsigned char>(line[at - 1])) &&
                                line[at - 1] != '_');
                const bool ends =
                    end == line.size() ||
                    (!std::isalnum(static_cast<unsigned char>(line[end])) && line[end] != '_');
                if (starts && ends) {
                    return true;
                }
            }
            return false;
        };
        std::string out;
        for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
            std::string line = lines[i];
            const std::size_t equals = line.find(" = s");
            if (line.starts_with("    v") && equals != std::string::npos) {
                const std::size_t start = equals + 3;
                const std::size_t dot = line.find('.', start);
                const std::string state = line.substr(start, dot - start);
                const bool digits = state.size() > 1 &&
                                    state.find_first_not_of("0123456789", 1) == std::string::npos;
                // The write must be the state's only use: a read after it
                // would see the new value, and one before it may be a view.
                bool read_later = false;
                for (std::size_t j = 0; j < lines.size() && !read_later; ++j) {
                    read_later = j != i && mentions(lines[j], state);
                }
                for (const char* write : {".index_copy(", ".index_put("}) {
                    const std::string call(write);
                    if (digits && !read_later && line.compare(dot, call.size(), call) == 0) {
                        line.insert(dot + call.size() - 1, "_");
                        written.push_back(state);
                    }
                }
            }
            // `vN = torch.ops.aten.index_put(sK, [...], ...)`: the same, spelled
            // as an ATen call (it takes `None` for a full slice).
            const std::string aten = " = torch.ops.aten.index_put(s";
            const std::size_t call = line.find(aten);
            if (line.starts_with("    v") && call != std::string::npos) {
                const std::size_t start = call + aten.size() - 1;
                const std::size_t comma = line.find(',', start);
                const std::string state = line.substr(start, comma - start);
                const bool digits = state.size() > 1 &&
                                    state.find_first_not_of("0123456789", 1) == std::string::npos;
                bool read_later = false;
                for (std::size_t j = 0; j < lines.size() && !read_later; ++j) {
                    read_later = j != i && mentions(lines[j], state);
                }
                if (digits && !read_later) {
                    line.insert(start - 1, "_");
                    written.push_back(state);
                }
            }
            out += line + "\n";
        }
        return out;
    }

    struct Hoisted {
        std::string constants;          // the input-independent assignments
        std::vector<std::string> names; // what they define, in order
        std::string body;               // the rest of the entry
    };

    // Splits the entry into what depends only on constants (rotary tables,
    // masks, folded scalars: computed once per shape by `constants`) and
    // what depends on the inputs, parameters, or states. Only top-level
    // assignments move; loop bodies stay where they are.
    Hoisted hoist(const std::string& body, const std::string& tail) const {
        Hoisted out;
        std::set<std::string> constant_names;
        std::vector<std::string> hoisted_names;
        std::string line;
        std::vector<std::string> lines;
        for (const char c : body) {
            if (c == '\n') {
                lines.push_back(line);
                line.clear();
            } else {
                line += c;
            }
        }
        for (const std::string& text : lines) {
            const std::size_t start = text.find_first_not_of(' ');
            const bool top_level = start == 4 && text[start] == 'v';
            const std::size_t end = top_level ? text.find(" = ", start) : std::string::npos;
            const bool assignment =
                end != std::string::npos && text.find_first_not_of("0123456789", start + 1) == end;
            if (assignment && depends_only_on(text.substr(end + 3), constant_names)) {
                const std::string name = text.substr(start, end - start);
                constant_names.insert(name);
                hoisted_names.push_back(name);
                // A mask that is the same every call has its blocks listed
                // once, eagerly, and can say there is nothing to skip.
                if (text.find(" = _flex_blocks(") != std::string::npos && text.ends_with(")")) {
                    out.constants += text.substr(0, text.size() - 1) + ", once=True)\n";
                } else {
                    out.constants += text + "\n";
                }
            } else {
                out.body += text + "\n";
            }
        }
        // Only the constants the entry reads cross the boundary; the rest
        // are intermediates of `constants` itself.
        const std::set<std::string> used = mentioned_values_in(out.body + tail);
        for (const std::string& name : hoisted_names) {
            if (used.contains(name)) {
                out.names.push_back(name);
            }
        }
        return out;
    }

    static std::set<std::string> mentioned_values_in(const std::string& text) {
        std::set<std::string> names;
        for (std::size_t i = 0; i < text.size(); ++i) {
            const bool boundary =
                i == 0 ||
                (!std::isalnum(static_cast<unsigned char>(text[i - 1])) && text[i - 1] != '_');
            if (text[i] != 'v' || !boundary) {
                continue;
            }
            std::size_t j = i + 1;
            while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            const bool ends =
                j == text.size() ||
                (!std::isalnum(static_cast<unsigned char>(text[j])) && text[j] != '_');
            // Definitions (`vN = `) count too when they are in a nested region
            // (loop bodies rebind nothing hoisted), so mentions anywhere qualify.
            if (j > i + 1 && ends) {
                names.insert(text.substr(i, j - i));
            }
            i = j;
        }
        return names;
    }

    // An expression is constant when every identifier it mentions is a
    // constant value, a literal, or a library name (no argument, no loop
    // variable, no input-dependent value).
    bool depends_only_on(const std::string& expression,
                         const std::set<std::string>& constant_names) const {
        const std::set<std::string> arguments(arguments_.begin(), arguments_.end());
        std::size_t i = 0;
        while (i < expression.size()) {
            const unsigned char c = static_cast<unsigned char>(expression[i]);
            if (std::isalpha(c) || c == '_') {
                std::size_t j = i + 1;
                while (j < expression.size() &&
                       (std::isalnum(static_cast<unsigned char>(expression[j])) ||
                        expression[j] == '_')) {
                    ++j;
                }
                const std::string word = expression.substr(i, j - i);
                const bool value = word.size() > 1 && word[0] == 'v' &&
                                   word.find_first_not_of("0123456789", 1) == std::string::npos;
                const bool loop_variable = word.size() > 1 && word[0] == 'w' &&
                                           std::isdigit(static_cast<unsigned char>(word[1]));
                if (arguments.contains(word) || loop_variable ||
                    (value && !constant_names.contains(word))) {
                    return false;
                }
                // Skip attribute chains such as `torch.float32` whole.
                i = j;
                continue;
            }
            ++i;
        }
        return true;
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
        cse_.emplace_back(); // values computed in the body belong to one iteration
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
        cse_.pop_back();
        return names;
    }

private:
    // `std.quant::linear_int4_groups` natively. tinygemm reads the high
    // nibble first and dequantizes `(q - 8) * scale + zero`, so the weights
    // are repacked and `(q - z) * s` becomes the zero `(8 - z) * s`.
    // Linear layers that read one input with parameter weights of their own
    // -- a layer's query, key, and value projections, its gate and up --
    // become one product over their weights side by side, sliced after: at
    // a decoding step's few rows, one kernel where there were three. The
    // weights are joined by `_adjacent`, weight-only work that `prepare`
    // does once; the runtime lays each group out contiguously (`FUSED`), so
    // the joined weight is a view and costs no memory.
    std::string fuse_linears(const std::string& body,
                             std::vector<std::vector<std::string>>& fused) {
        struct Linear {
            std::size_t line;
            std::string out;
            std::string input;
            std::size_t weight;
            std::optional<std::size_t> bias;
        };
        std::vector<std::string> lines;
        std::size_t highest = 0;
        {
            std::string current;
            for (const char c : body) {
                if (c == '\n') {
                    lines.push_back(current);
                    current.clear();
                } else {
                    current += c;
                }
            }
            if (!current.empty()) {
                lines.push_back(current);
            }
        }
        const auto parameter_index = [&](const std::string& word) -> std::optional<std::size_t> {
            if (word.size() < 2 || word[0] != 'p' ||
                !std::all_of(word.begin() + 1, word.end(), [](char c) {
                    return std::isdigit(static_cast<unsigned char>(c)) != 0;
                })) {
                return std::nullopt;
            }
            const std::size_t index = std::stoul(word.substr(1));
            return index < parameter_types_.size() ? std::optional(index) : std::nullopt;
        };
        std::vector<Linear> linears;
        const std::string prefix = "    ";
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const std::string& line = lines[i];
            for (std::size_t at = 0; at + 1 < line.size(); ++at) {
                if (line[at] == 'v' &&
                    (at == 0 || !std::isalnum(static_cast<unsigned char>(line[at - 1])))) {
                    std::size_t end = at + 1;
                    while (end < line.size() &&
                           std::isdigit(static_cast<unsigned char>(line[end])) != 0) {
                        ++end;
                    }
                    if (end > at + 1) {
                        highest = std::max(highest,
                                           static_cast<std::size_t>(
                                               std::stoul(line.substr(at + 1, end - at - 1))));
                    }
                }
            }
            // `    vN = F.linear(x, pW, pB | None)`, at the top level.
            const std::string call = " = F.linear(";
            const std::size_t equals = line.find(call);
            if (!line.starts_with(prefix) || line[prefix.size()] == ' ' ||
                equals == std::string::npos || !line.ends_with(")")) {
                continue;
            }
            const std::string out = line.substr(prefix.size(), equals - prefix.size());
            const std::string arguments =
                line.substr(equals + call.size(), line.size() - equals - call.size() - 1);
            std::vector<std::string> parts;
            std::size_t start = 0;
            for (std::size_t comma = arguments.find(", "); comma != std::string::npos;
                 comma = arguments.find(", ", start)) {
                parts.push_back(arguments.substr(start, comma - start));
                start = comma + 2;
            }
            parts.push_back(arguments.substr(start));
            if (parts.size() != 3 || out.find_first_of(" ,") != std::string::npos) {
                continue;
            }
            const auto weight = parameter_index(parts[1]);
            const auto bias = parameter_index(parts[2]);
            if (!weight || (!bias && parts[2] != "None")) {
                continue;
            }
            const auto& [shape, dtype] = parameter_types_[*weight];
            if (shape.size() != 2) {
                continue;
            }
            linears.push_back({i, out, parts[0], *weight, bias});
        }
        // Groups: one input, all biased or none, the same input width and
        // dtype, distinct weights.
        std::map<std::string, std::vector<std::size_t>> groups;
        for (std::size_t i = 0; i < linears.size(); ++i) {
            const Linear& linear = linears[i];
            const auto& [shape, dtype] = parameter_types_[linear.weight];
            const std::string key = linear.input + "|" + (linear.bias ? "b" : "-") + "|" +
                                    std::to_string(shape[1]) + "|" +
                                    std::to_string(static_cast<int>(dtype));
            auto& members = groups[key];
            const bool repeated = std::any_of(members.begin(), members.end(), [&](std::size_t j) {
                return linears[j].weight == linear.weight;
            });
            if (!repeated) {
                members.push_back(i);
            }
        }
        std::map<std::size_t, std::string> replaced; // first line -> the fused lines
        std::set<std::size_t> removed;
        for (const auto& [key, members] : groups) {
            if (members.size() < 2) {
                continue;
            }
            const auto join = [&](const std::vector<std::size_t>& indices) {
                std::string list;
                std::vector<std::string> paths;
                for (std::size_t j = 0; j < indices.size(); ++j) {
                    list += (j == 0 ? "" : ", ") + std::string("p") + std::to_string(indices[j]);
                    paths.push_back(parameters_[indices[j]]);
                }
                fused.push_back(paths);
                return list;
            };
            std::vector<std::size_t> weights;
            std::vector<std::size_t> biases;
            for (const std::size_t i : members) {
                weights.push_back(linears[i].weight);
                if (const std::optional<std::size_t>& bias = linears[i].bias; bias.has_value()) {
                    biases.push_back(bias.value());
                }
            }
            const Linear& first = linears[members.front()];
            const std::string weight = "v" + std::to_string(++highest);
            std::string text = prefix + weight + " = _adjacent(" + join(weights) + ")\n";
            std::string bias = "None";
            if (!biases.empty()) {
                bias = "v" + std::to_string(++highest);
                text += prefix + bias + " = _adjacent(" + join(biases) + ")\n";
            }
            const std::string product = "v" + std::to_string(++highest);
            text += prefix;
            text += product;
            text += " = F.linear(";
            text += first.input;
            text += ", ";
            text += weight;
            text += ", ";
            text += bias;
            text += ")\n";
            std::int64_t offset = 0;
            for (const std::size_t i : members) {
                const std::int64_t rows = parameter_types_[linears[i].weight].first[0];
                text += prefix;
                text += linears[i].out;
                text += " = ";
                text += product;
                text += "[..., ";
                text += std::to_string(offset);
                text += ":";
                text += std::to_string(offset + rows);
                text += "]\n";
                offset += rows;
                removed.insert(linears[i].line);
            }
            replaced[first.line] = text;
        }
        if (replaced.empty()) {
            return body;
        }
        std::string out;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            if (const auto found = replaced.find(i); found != replaced.end()) {
                out += found->second;
            } else if (!removed.contains(i)) {
                out += lines[i] + "\n";
            }
        }
        return out;
    }

    static std::string adjacent_helper() {
        return "\n\ndef _adjacent(*parts):\n"
               "    \"\"\"`torch.cat(parts)`, as a view when the parts lie one after\n"
               "    another in one buffer, as the runtime lays out the groups in FUSED.\"\"\"\n"
               "    first = parts[0]\n"
               "    storage = first.untyped_storage().data_ptr()\n"
               "    offset = first.storage_offset()\n"
               "    for part in parts:\n"
               "        if (not part.is_contiguous() or part.dtype != first.dtype\n"
               "                or part.untyped_storage().data_ptr() != storage\n"
               "                or part.storage_offset() != offset):\n"
               "            return torch.cat(parts)\n"
               "        offset += part.numel()\n"
               "    rows = sum(part.shape[0] for part in parts)\n"
               "    return first.as_strided((rows, *first.shape[1:]), first.stride())\n\n\n";
    }

    static std::string int4_helpers() {
        return "def _int4_pack(packed, scale, zero):\n"
               "    out_features, groups, half = packed.shape\n"
               "    group = 2 * half\n"
               "    in_features = groups * group\n"
               "    packed = packed.reshape(out_features, groups * half)\n"
               "    if (\n"
               "        packed.is_cuda\n"
               "        and scale.dtype == torch.bfloat16\n"
               "        and group in (32, 64, 128, 256)\n"
               "        and in_features % 128 == 0\n"
               "        and out_features % 8 == 0\n"
               "    ):\n"
               "        swapped = ((packed & 15) << 4) | (packed >> 4)\n"
               "        weight = torch.ops.aten._convert_weight_to_int4pack(swapped.contiguous(), "
               "8)\n"
               "        zeros = (8.0 - zero.float()) * scale.float()\n"
               "        pairs = torch.stack([scale.float(), zeros], dim=-1).transpose(0, 1)\n"
               "        return (weight, pairs.contiguous().to(torch.bfloat16), group)\n"
               "    q = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(out_features, "
               "groups, group)\n"
               "    weight = (q.float() - zero.float()[..., None]) * scale.float()[..., None]\n"
               "    return (weight.reshape(out_features, in_features).to(scale.dtype),)\n"
               "\n\n"
               "try:\n"
               "    import triton\n"
               "    import triton.language as tl\n"
               "except ImportError:  # PyTorch without a GPU build\n"
               "    triton = None\n"
               "\n"
               "try:\n"
               "    from linnet.torch.kernels import int4_linear as _int4_kernel\n"
               "except ImportError:  # no Triton: tinygemm and the dequantized weight\n"
               "    _int4_kernel = None\n"
               "\n"
               "if triton is not None:\n"
               "\n"
               "    @triton.jit\n"
               "    def _int4_unpack(packed, scale, zero, out, count, half, BLOCK: tl.constexpr):\n"
               "        # Each byte is two weights, the even one low; each run of `half`\n"
               "        # bytes a group with one scale and zero point.\n"
               "        start = tl.program_id(0) * BLOCK\n"
               "        index = start + tl.arange(0, BLOCK)\n"
               "        inside = index < count\n"
               "        byte = tl.load(packed + index, mask=inside, other=0).to(tl.int32)\n"
               "        group = index // half\n"
               "        step = tl.load(scale + group, mask=inside, other=0).to(tl.float32)\n"
               "        level = tl.load(zero + group, mask=inside, other=0).to(tl.float32)\n"
               "        low = ((byte & 15).to(tl.float32) - level) * step\n"
               "        high = ((byte >> 4).to(tl.float32) - level) * step\n"
               "        pairs = 2 * start + tl.arange(0, 2 * BLOCK)\n"
               "        values = tl.interleave(low, high).to(out.dtype.element_ty)\n"
               "        tl.store(out + pairs, values, mask=pairs < 2 * count)\n"
               "\n\n"
               "def _int4_weight(raw, scale, zero):\n"
               "    \"\"\"The weight dequantized, in `scale`'s type: one kernel on a GPU.\"\"\"\n"
               "    out_features, groups, half = raw.shape\n"
               "    if triton is not None and raw.is_cuda:\n"
               "        out = torch.empty((out_features, groups * 2 * half), dtype=scale.dtype,\n"
               "                          device=raw.device)\n"
               "        count = raw.numel()\n"
               "        _int4_unpack[(triton.cdiv(count, 1024),)](\n"
               "            raw.contiguous(), scale.contiguous(), zero.contiguous(), out, count, "
               "half,\n"
               "            BLOCK=1024)\n"
               "        return out\n"
               "    q = torch.stack([raw & 15, raw >> 4], dim=-1).reshape(out_features, groups, "
               "2 * half)\n"
               "    w = (q.to(scale.dtype) - zero.to(scale.dtype)[..., None]) * scale[..., None]\n"
               "    return w.reshape(out_features, groups * 2 * half)\n"
               "\n\n"
               "def _int4_linear(x, packed, bias, raw, scale, zero):\n"
               "    flat = x.reshape(-1, x.shape[-1])\n"
               "    rows = flat.shape[0]\n"
               "    if len(packed) == 3 and rows < 8:\n"
               "        weight, pairs, group = packed\n"
               "        y = torch.ops.aten._weight_int4pack_mm(flat, weight, group, pairs)\n"
               "    elif len(packed) == 3 and rows <= 128 and _int4_kernel is not None:\n"
               "        # tinygemm's time grows with the rows; a batch of decoding\n"
               "        # requests unpacks each group inside one matrix product.\n"
               "        y = _int4_kernel(flat, raw, scale, zero)\n"
               "    elif len(packed) == 3:\n"
               "        # A prompt's many rows multiply the weight dequantized for\n"
               "        # this call instead.\n"
               "        y = F.linear(flat, _int4_weight(raw, scale, zero))\n"
               "    else:\n"
               "        y = F.linear(flat, packed[0])\n"
               "    y = y.reshape(*x.shape[:-1], y.shape[-1])\n"
               "    return y if bias is None else y + bias\n"
               "\n\n";
    }

    // `std.nn.moe::linear_experts` natively: the rows sorted by expert, one
    // grouped matrix product reading each expert's weight in place, and the
    // results put back in the rows' order. `offs` ends each expert's rows;
    // counting rather than `bincount` keeps it on the device, so the step can
    // be a CUDA graph.
    static std::string mxfp4_helper() {
        return "try:\n"
               "    from linnet.torch.kernels import mxfp4_experts as _mxfp4_kernel\n"
               "except ImportError:  # no Triton: the arithmetic in `_mxfp4_experts`\n"
               "    _mxfp4_kernel = None\n"
               "\n"
               "_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, "
               "-3.0, -4.0, -6.0)\n"
               "\n\n"
               "def _mxfp4_experts(x, blocks, scales, experts):\n"
               "    \"\"\"`std.quant::mxfp4_experts`: for a decoding step's few rows on CUDA,\n"
               "    Linnet's kernel reads the chosen experts' four-bit bytes as they are;\n"
               "    otherwise the chosen experts are unpacked and multiplied in f32.\"\"\"\n"
               "    rows, chosen, _ = x.shape\n"
               "    if _mxfp4_kernel is not None and x.is_cuda and rows * chosen <= 32:\n"
               "        return _mxfp4_kernel(x, blocks, scales, experts)\n"
               "    table = torch.tensor(_FP4, dtype=torch.float32, device=x.device)\n"
               "    taken = blocks[experts]\n"
               "    values = torch.stack([table[(taken & 15).long()], table[(taken >> "
               "4).long()]], dim=-1)\n"
               "    factor = torch.exp2(scales[experts].float() - 127)[..., None]\n"
               "    weight = (values.reshape(*taken.shape[:-1], 32) * "
               "factor).reshape(*taken.shape[:-2], -1)\n"
               "    return torch.einsum(\"rki,rkoi->rko\", x.float(), weight).to(x.dtype)\n"
               "\n\n";
    }

    static std::string sink_helper() {
        return "def _sink_attend(query, key, value, sinks, mask, scale, blocks, block_q, "
               "block_k):\n"
               "    \"\"\"`std.nn.attention::sink_attention`: attention whose softmax also\n"
               "    counts one logit per query head, that share then dropped. Compiled for\n"
               "    CUDA, FlexAttention over the blocks the mask reaches, each row scaled by\n"
               "    the share its keys keep against the sink, `sigmoid(lse - sink)`;\n"
               "    otherwise two products around the softmax with the sink in it.\"\"\"\n"
               "    batch, heads, queries, width = query.shape\n"
               "    kv_heads, keys = key.shape[1], key.shape[2]\n"
               "    group = heads // kv_heads\n"
               "    rows = mask.reshape(-1, queries, keys)\n"
               "    sink = sinks.float()\n"
               "    if blocks is not None and torch.compiler.is_compiling() and not "
               "_split(query):\n"
               "        from torch.nn.attention.flex_attention import BlockMask, "
               "flex_attention\n"
               "\n"
               "        per_row = rows.shape[0] > 1\n"
               "\n"
               "        def mask_mod(b, h, q, kv):\n"
               "            return rows[b if per_row else 0, q, kv]\n"
               "\n"
               "        count, order, whole_count, whole_order = blocks\n"
               "        block_mask = BlockMask.from_kv_blocks(\n"
               "            count,\n"
               "            order,\n"
               "            whole_count,\n"
               "            whole_order,\n"
               "            BLOCK_SIZE=(block_q, block_k),\n"
               "            mask_mod=mask_mod,\n"
               "            seq_lengths=(queries, keys),\n"
               "        )\n"
               "        out, lse = flex_attention(\n"
               "            query,\n"
               "            key,\n"
               "            value,\n"
               "            block_mask=block_mask,\n"
               "            scale=scale,\n"
               "            enable_gqa=group > 1,\n"
               "            return_lse=True,\n"
               "        )\n"
               "        keep = torch.sigmoid(lse.float() - sink[:, None])\n"
               "        return (out.float() * keep[..., None]).to(query.dtype)\n"
               "    grouped = query.reshape(batch, kv_heads, group * queries, width)\n"
               "    scores = torch.matmul(grouped, key.transpose(-1, -2)).float() * scale\n"
               "    scores = scores.reshape(batch, kv_heads, group, queries, keys)\n"
               "    scores = scores.masked_fill(~rows.reshape(rows.shape[0], 1, 1, queries, "
               "keys), -1e30)\n"
               "    sink = sink.reshape(1, kv_heads, group, 1, 1)\n"
               "    peak = torch.maximum(scores.amax(-1, keepdim=True), sink)\n"
               "    shifted = torch.exp(scores - peak)\n"
               "    total = shifted.sum(-1, keepdim=True) + torch.exp(sink - peak)\n"
               "    weights = (shifted / total).to(value.dtype)\n"
               "    weights = weights.reshape(batch, kv_heads, group * queries, keys)\n"
               "    return torch.matmul(weights, value).reshape(batch, heads, queries, width)\n"
               "\n\n";
    }

    // `std.nn.loss`'s output-head forms: `linnet.torch.loss` a block of rows
    // at a time; without it, the logits whole, as the bodies compute.
    static std::string loss_helper() {
        return "try:\n"
               "    from linnet.torch.loss import linear_cross_entropy as _linear_cross_entropy\n"
               "    from linnet.torch.loss import linear_token_log_probs as "
               "_linear_token_log_probs\n"
               "except ImportError:\n"
               "\n"
               "    def _linear_token_log_probs(hidden, weight, targets):\n"
               "        logits = (hidden @ weight.T).float()\n"
               "        picked = logits.gather(1, targets.long()[:, None]).squeeze(1)\n"
               "        return picked - torch.logsumexp(logits, dim=-1)\n"
               "\n"
               "    def _linear_cross_entropy(hidden, weight, targets, weights):\n"
               "        return -(weights.float() * _linear_token_log_probs(hidden, weight, "
               "targets)).sum()\n"
               "\n\n";
    }

    static std::string mxfp4_grouped_helper() {
        return "try:\n"
               "    from linnet.torch.moe import available as _mxfp4_grouped_available\n"
               "    from linnet.torch.moe import mxfp4_grouped as _mxfp4_grouped_op\n"
               "except ImportError:  # the experts dequantized, as the bodies do\n"
               "    _mxfp4_grouped_op = None\n"
               "\n\n"
               "def _mxfp4_grouped(x, blocks, scales, experts, weights=None):\n"
               "    \"\"\"`std.quant::mxfp4_linear_experts_shared` (no `weights`, `x` [R, "
               "In])\n"
               "    and `mxfp4_combine_experts` (`x` [R, K, In], the products weighed by\n"
               "    `weights` and summed per row). On a Hopper GPU in bf16,\n"
               "    `linnet.torch.moe` multiplies each expert by the rows that chose it\n"
               "    (`triton_kernels`' MXFP4 product where installed, bf16 experts\n"
               "    otherwise); elsewhere every expert is dequantized and multiplied in\n"
               "    the dense form the bodies take.\"\"\"\n"
               "    rows, chosen = experts.shape\n"
               "    count, out_features = blocks.shape[0], blocks.shape[1]\n"
               "    shared = weights is None\n"
               "    if _mxfp4_grouped_op is not None and _mxfp4_grouped_available(x):\n"
               "        y = _mxfp4_grouped_op(x, blocks, scales, experts, shared)\n"
               "        if shared:\n"
               "            return y\n"
               "        return (y.float() * weights.float()[..., None]).sum(1).to(x.dtype)\n"
               "    table = torch.tensor(_FP4, dtype=torch.float32, device=x.device)\n"
               "    values = torch.stack([table[(blocks & 15).long()], table[(blocks >> "
               "4).long()]], dim=-1)\n"
               "    factor = torch.exp2(scales.float() - 127)[..., None]\n"
               "    weight = (values.reshape(*blocks.shape[:-1], 32) * factor).reshape(count, "
               "out_features, -1)\n"
               "    weight = weight.to(x.dtype)\n"
               "    width = weight.shape[-1]\n"
               "    if shared:\n"
               "        every = F.linear(x, weight.reshape(count * out_features, width))\n"
               "        every = every.reshape(rows, count, out_features)\n"
               "        return every.gather(1, experts[..., None].expand(rows, chosen, "
               "out_features))\n"
               "    placed = torch.zeros(rows, count, width, dtype=x.dtype, device=x.device)\n"
               "    placed.scatter_add_(1, experts[..., None].expand(rows, chosen, width), "
               "weights[..., None] * x)\n"
               "    joined = weight.permute(1, 0, 2).reshape(out_features, count * width)\n"
               "    return F.linear(placed.reshape(rows, count * width), joined)\n"
               "\n\n";
    }

    static std::string shards_helper() {
        return "# The process group a sharded model's processes run over, which the\n"
               "# runtime sets (`load(tensor_parallel=...)`); None for one process.\n"
               "_GROUP = None\n"
               "\n\n"
               "def _all_reduce(x):\n"
               "    \"\"\"`std.nn.parallel::all_reduce`: the sum over the shards' "
               "processes;\n"
               "    a small one by Linnet's one-shot kernel, a large one by NCCL.\"\"\"\n"
               "    if _GROUP is None:\n"
               "        return x\n"
               "    from linnet.torch.collectives import all_reduce\n"
               "\n"
               "    return all_reduce(x, _GROUP)\n"
               "\n\n"
               "def _all_gather(x):\n"
               "    \"\"\"`std.nn.parallel::all_gather`: the shards' slices side by side "
               "along\n"
               "    the last axis, in the processes' order.\"\"\"\n"
               "    if _GROUP is None:\n"
               "        return x\n"
               "    from linnet.torch.collectives import all_gather\n"
               "\n"
               "    return all_gather(x, _GROUP)\n"
               "\n\n";
    }

    static std::string flex_helpers() {
        return "def _flex_blocks(mask, queries, keys, block_q, block_k, once=False):\n"
               "    \"\"\"The key blocks each query block of `mask` ([rows or 1, queries, "
               "keys],\n"
               "    or [queries, keys]) reaches in part and reaches whole, as FlexAttention's\n"
               "    block mask lists them; None where FlexAttention does not run. `once`: the\n"
               "    mask is the same every call and listed eagerly, so None also where it\n"
               "    reaches every block and there is nothing to skip.\"\"\"\n"
               "    if not mask.is_cuda:\n"
               "        return None\n"
               "    rows = mask.reshape(-1, queries, keys)\n"
               "    q_blocks, k_blocks = max(1, queries // block_q), keys // block_k\n"
               "    # Each tile's elements side by side, reduced in one step: inductor\n"
               "    # cannot fuse a reduction over an axis of one.\n"
               "    tiles = (\n"
               "        rows.reshape(rows.shape[0], q_blocks, -1, k_blocks, block_k)\n"
               "        .transpose(2, 3)\n"
               "        .reshape(rows.shape[0], q_blocks, k_blocks, -1)\n"
               "    )\n"
               "    reached = tiles.any(-1)\n"
               "    if once and bool(reached.all()):\n"
               "        return None\n"
               "    whole = tiles.all(-1)\n"
               "\n"
               "    def listed(live):\n"
               "        count = live.sum(-1, dtype=torch.int32)[:, None]\n"
               "        order = torch.argsort((~live).to(torch.int8), dim=-1, stable=True)\n"
               "        return count, order.to(torch.int32)[:, None]\n"
               "\n"
               "    return (*listed(reached & ~whole), *listed(whole))\n"
               "\n\n"
               "def _split(value):\n"
               "    \"\"\"Whether `value` is split from the outside (a DTensor), which\n"
               "    FlexAttention does not take.\"\"\"\n"
               "    from torch.distributed.tensor import DTensor\n"
               "\n"
               "    return isinstance(value, DTensor)\n"
               "\n\n"
               "def _attend(query, key, value, mask, scale, blocks, block_q, block_k):\n"
               "    \"\"\"Attention under a boolean mask (true attends): FlexAttention over "
               "the\n"
               "    blocks the mask reaches when compiled for CUDA, and otherwise the same\n"
               "    arithmetic as two products around a softmax (one query) or SDPA.\"\"\"\n"
               "    grouped = key.shape[1] != query.shape[1]\n"
               "    batch, heads, queries, width = query.shape\n"
               "    keys = key.shape[2]\n"
               "    rows = mask.reshape(-1, queries, keys)\n"
               "    if blocks is not None and torch.compiler.is_compiling() and not "
               "_split(query):\n"
               "        from torch.nn.attention.flex_attention import BlockMask, "
               "flex_attention\n"
               "\n"
               "        per_row = rows.shape[0] > 1\n"
               "\n"
               "        def mask_mod(b, h, q, kv):\n"
               "            return rows[b if per_row else 0, q, kv]\n"
               "\n"
               "        count, order, whole_count, whole_order = blocks\n"
               "        block_mask = BlockMask.from_kv_blocks(\n"
               "            count,\n"
               "            order,\n"
               "            whole_count,\n"
               "            whole_order,\n"
               "            BLOCK_SIZE=(block_q, block_k),\n"
               "            mask_mod=mask_mod,\n"
               "            seq_lengths=(queries, keys),\n"
               "        )\n"
               "        return flex_attention(\n"
               "            query, key, value, block_mask=block_mask, scale=scale, "
               "enable_gqa=grouped\n"
               "        )\n"
               "    if queries == 1:\n"
               "        # PyTorch sends a masked single query to a kernel that does not\n"
               "        # split the keys across blocks; two products are faster.\n"
               "        kv_heads = key.shape[1]\n"
               "        grouped_query = query.reshape(batch, kv_heads, heads // kv_heads, "
               "width)\n"
               "        scores = torch.matmul(grouped_query, key.transpose(-1, -2)).float() * "
               "scale\n"
               "        scores = scores.masked_fill(~rows.reshape(rows.shape[0], 1, 1, keys), "
               "-1e30)\n"
               "        weights = torch.softmax(scores, dim=-1).to(value.dtype)\n"
               "        return torch.matmul(weights, value).reshape(batch, heads, 1, width)\n"
               "    attn_mask = rows.unsqueeze(1) if rows.shape[0] > 1 else rows[0]\n"
               "    return F.scaled_dot_product_attention(\n"
               "        query, key, value, attn_mask=attn_mask, scale=scale, "
               "enable_gqa=grouped\n"
               "    )\n"
               "\n\n";
    }

    static std::string experts_helper() {
        return "def _grouped(x, weight):\n"
               "    \"\"\"Whether torch._grouped_mm runs these: CUDA, Hopper or later, "
               "bf16.\"\"\"\n"
               "    return (\n"
               "        x.is_cuda\n"
               "        and x.dtype == torch.bfloat16\n"
               "        and weight.dtype == torch.bfloat16\n"
               "        and hasattr(torch, \"_grouped_mm\")\n"
               "        and torch.cuda.get_device_capability(x.device)[0] >= 9\n"
               "        and x.shape[-1] % 8 == 0\n"
               "        and weight.shape[1] % 8 == 0\n"
               "    )\n"
               "\n\n"
               "def _linear_experts(x, weight, experts):\n"
               "    rows, chosen, width = x.shape\n"
               "    count, out_features, _ = weight.shape\n"
               "    flat = experts.reshape(-1)\n"
               "    if _grouped(x, weight):\n"
               "        order = flat.argsort(stable=True)\n"
               "        slots = torch.arange(count, device=x.device)\n"
               "        ends = (flat[None, :] <= slots[:, None]).sum(1).to(torch.int32)\n"
               "        y = torch._grouped_mm(x.reshape(-1, width)[order], weight.transpose(-2, "
               "-1),\n"
               "                              offs=ends)\n"
               "        unsorted = torch.empty_like(y).index_copy_(0, order, y)\n"
               "        return unsorted.reshape(rows, chosen, out_features)\n"
               "    taken = weight[flat].reshape(rows, chosen, out_features, width)\n"
               "    y = torch.einsum(\"rki,rkoi->rko\", x.float(), taken.float())\n"
               "    return y.to(x.dtype)\n"
               "\n\n"
               "def _linear_experts_shared(x, weight, experts):\n"
               "    rows, chosen = experts.shape\n"
               "    count, out_features, width = weight.shape\n"
               "    if _grouped(x, weight):\n"
               "        return _linear_experts(x[:, None, :].expand(rows, chosen, width), weight, "
               "experts)\n"
               "    every = F.linear(x, weight.reshape(count * out_features, width))\n"
               "    every = every.reshape(rows, count, out_features)\n"
               "    return every.gather(1, experts[..., None].expand(rows, chosen, out_features))\n"
               "\n\n"
               "def _combine_experts(x, weight, experts, weights):\n"
               "    rows, chosen, width = x.shape\n"
               "    count, out_features, _ = weight.shape\n"
               "    if _grouped(x, weight):\n"
               "        y = _linear_experts(x, weight, experts).float() * weights.float()[..., "
               "None]\n"
               "        return y.sum(1).to(x.dtype)\n"
               "    placed = torch.zeros(rows, count, width, dtype=x.dtype, device=x.device)\n"
               "    placed.scatter_add_(1, experts[..., None].expand(rows, chosen, width), "
               "weights[..., None] * x)\n"
               "    joined = weight.permute(1, 0, 2).reshape(out_features, count * width)\n"
               "    return F.linear(placed.reshape(rows, count * width), joined)\n"
               "\n\n";
    }

    std::string device() const {
        return placed_ ? "_dev[" + std::to_string(slot_) + "]" : std::string("_device");
    }

    // Every value is immutable, so an expression already computed in this
    // scope (or an enclosing one) names the same tensor: identical rotary
    // tables or masks across inlined layers are emitted once.
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

    bool placed_ = false;  // with placement, `main` and `constants` take `_dev`
    bool prepare_ = false; // split weight-only work into `prepare`
    bool fuse_ = true;     // with `prepare`, join sibling linear layers
    int slots_ = 1;
    int slot_ = 0;
    std::vector<std::string> arguments_;
    std::vector<std::string> parameters_;            // paths, in argument order
    std::vector<std::string> states_;                // paths, in argument order
    std::map<std::string, std::string> literals_;    // constant name -> Python literal
    std::set<std::string> causal_masks_;             // square masks from `causal_mask`
    bool int4_helpers_ = false;                      // `_int4_pack` and `_int4_linear` are used
    bool experts_helper_ = false;                    // `_linear_experts` is used
    bool flex_helpers_ = false;                      // `_flex_blocks` and `_attend` are used
    bool sink_helper_ = false;                       // `_sink_attend` is used
    bool shards_helper_ = false;                     // `_all_reduce` is used
    bool mxfp4_helper_ = false;                      // `_mxfp4_experts` is used
    bool mxfp4_grouped_helper_ = false;              // `_mxfp4_grouped` is used
    bool loss_helper_ = false;                       // `_linear_cross_entropy` is used
    std::map<std::string, std::string> flex_blocks_; // mask and sizes -> its `_flex_blocks`
    std::vector<std::map<std::string, std::string>> cse_{1}; // expression -> name, per scope
    std::map<std::string, double> values_;                   // constant name -> folded scalar value
    std::map<ScalarKind, std::string> zeros_;                // per-dtype zero constants
    std::string body_;
    std::string indent_ = "    ";
    std::vector<std::vector<std::string>> loop_names_;
    std::size_t loops_ = 0;
    std::size_t next_ = 0;
    std::vector<std::pair<Dims, ScalarKind>> parameter_types_; // shape and dtype, PARAMETERS order
};

} // namespace

std::expected<std::string, std::string> export_torch_source(ir::Module& module,
                                                            const TorchSourceOptions& options) {
    TorchTarget target(options.prepare, options.fuse);
    return export_graph(module, options, target);
}

} // namespace linnet::backend
