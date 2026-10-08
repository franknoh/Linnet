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

std::string tensor_type(const Dims& shape, ScalarKind dtype) {
    std::string text(dtype_names(dtype).onnx);
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
        if (dtype == ScalarKind::F16 || dtype == ScalarKind::BF16) {
            // The text format spells a 16-bit float by its bit pattern, so
            // `float16 {0.0}` does not parse. A float constant cast down says
            // the same thing and stays readable.
            const TensorInfo wide{
                constant_text(literal_text(literal, ScalarKind::F32), {}, ScalarKind::F32),
                {},
                ScalarKind::F32};
            return convert(wide, dtype);
        }
        const std::string name = constant_text(literal_text(literal, dtype), {}, dtype);
        literals_[name] = literal_text(literal, dtype);
        return name;
    }

    bool broadcasts_elementwise() const override { return true; }

    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        ScalarKind dtype) override {
        const std::string equation =
            "equation = \"" + einsum_equation(lhs_axes, rhs_axes, out_axes) + "\"";
        if (dtype == ScalarKind::BF16) {
            // ONNX's Einsum takes no bf16: contract in f32 and round back,
            // which is what a bf16 matrix product accumulates in anyway.
            const auto wide = [&](const TensorInfo& t) {
                return t.dtype == ScalarKind::F32
                           ? t
                           : TensorInfo{convert(t, ScalarKind::F32), t.shape, ScalarKind::F32};
            };
            const TensorInfo product{
                node("Einsum", {wide(lhs), wide(rhs)}, equation, shape, ScalarKind::F32),
                shape,
                ScalarKind::F32};
            return convert(product, dtype);
        }
        return node("Einsum", {lhs, rhs}, equation, shape, dtype);
    }

    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           ScalarKind dtype) override {
        // Library operations with an ONNX operator of the same meaning;
        // `(input dtype)` variants skip the f32 casts.
        const CallName parsed = call_name(implementation);
        const std::string& implementation_base = parsed.base;
        const bool fast = parsed.fast;
        const ScalarKind acc = fast ? dtype : ScalarKind::F32;
        const std::vector<const TensorInfo*> at = operand_pointers(operands);
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
        if (implementation_base == "torch.nn.functional.embedding" && operands.size() == 2 &&
            at[0] != nullptr && at[1] != nullptr) {
            return node("Gather", {*at[1], *at[0]}, "axis = 0", shape, dtype);
        }
        if (implementation_base == "torch.index_select" && operands.size() == 2 &&
            at[0] != nullptr && at[1] != nullptr) {
            return node("Gather", {*at[0], *at[1]}, "axis = -1", shape, dtype);
        }
        if (is_convolution(implementation_base) && operands.size() == 3 && at[0] != nullptr &&
            at[1] != nullptr) {
            const auto window = conv_window(implementation_base);
            if (!window) {
                return std::nullopt;
            }
            // ONNX Runtime has no bf16 convolution: convolve in f32 and
            // round back, as the contraction does.
            const ScalarKind kind = dtype == ScalarKind::BF16 ? ScalarKind::F32 : dtype;
            const auto as_kind = [&](const TensorInfo& t) {
                return t.dtype == kind ? t : TensorInfo{convert(t, kind), t.shape, kind};
            };
            std::vector<TensorInfo> inputs{as_kind(*at[0]), as_kind(*at[1])};
            if (at[2] != nullptr) {
                inputs.push_back(as_kind(*at[2]));
            }
            // Pads are every axis's beginning, then every axis's end.
            Dims pads = window->pads;
            pads.insert(pads.end(), window->pads.begin(), window->pads.end());
            const std::string out = node("Conv",
                                         inputs,
                                         "strides = [" + int_list(window->strides) + "], pads = [" +
                                             int_list(pads) + "]",
                                         shape,
                                         kind);
            return kind == dtype ? out : convert({out, shape, kind}, dtype);
        }
        if (implementation_base == "torch.nn.functional.interpolate(nearest)" &&
            operands.size() == 1 && at[0] != nullptr && at[0]->shape.size() == 4 &&
            shape.size() == 4) {
            // `Resize` to the result's shape: the canonical body stacks four
            // copies, which at a VAE's last stage is past the 2^31 elements
            // ONNX Runtime's CUDA `Concat` indexes.
            const Dims& input = at[0]->shape;
            if (input[2] <= 0 || input[3] <= 0 || shape[2] % input[2] != 0 ||
                shape[3] % input[3] != 0) {
                return std::nullopt;
            }
            const ScalarKind kind = dtype == ScalarKind::BF16 ? ScalarKind::F32 : dtype;
            const TensorInfo x =
                at[0]->dtype == kind ? *at[0] : TensorInfo{convert(*at[0], kind), input, kind};
            const TensorInfo none{"", {}, ScalarKind::F32};
            const std::string out = node("Resize",
                                         {x, none, none, int64_vector(shape)},
                                         "mode = \"nearest\", coordinate_transformation_mode = "
                                         "\"asymmetric\", nearest_mode = \"floor\"",
                                         shape,
                                         kind);
            return kind == dtype ? out : convert({out, shape, kind}, dtype);
        }
        if (implementation_base == "torch.nn.functional.batch_norm" && operands.size() == 6 &&
            at[0] != nullptr && at[1] != nullptr && at[2] != nullptr && at[3] != nullptr &&
            at[4] != nullptr && at[5] != nullptr) {
            // The operator, not its arithmetic: ONNX Runtime folds a
            // `BatchNormalization` into the convolution before it.
            const std::string epsilon = literal_of(at[5]->name);
            if (epsilon.empty()) {
                return std::nullopt;
            }
            const ScalarKind kind = dtype == ScalarKind::BF16 ? ScalarKind::F32 : dtype;
            const auto as_kind = [&](const TensorInfo& t) {
                return t.dtype == kind ? t : TensorInfo{convert(t, kind), t.shape, kind};
            };
            const std::string out = node("BatchNormalization",
                                         {as_kind(*at[0]),
                                          as_kind(*at[3]),
                                          as_kind(*at[4]),
                                          as_kind(*at[1]),
                                          as_kind(*at[2])},
                                         "epsilon = " + epsilon,
                                         shape,
                                         kind);
            return kind == dtype ? out : convert({out, shape, kind}, dtype);
        }
        if (implementation_base == "torch.ops.aten._weight_int4pack_mm" && operands.size() == 5 &&
            at[0] != nullptr && at[1] != nullptr && at[2] != nullptr && at[3] != nullptr) {
            // ONNX Runtime's `MatMulNBits` reads the packed weights as they
            // are (low nibble first) with a float zero point per group. It
            // takes f32 and f16 inputs: bf16 goes through f32.
            const TensorInfo& x = *at[0];
            const TensorInfo& packed = *at[1];
            const TensorInfo& scale = *at[2];
            const TensorInfo& zero = *at[3];
            if (x.shape.empty() || packed.shape.size() != 3 || scale.shape.size() != 2) {
                return std::nullopt;
            }
            const std::int64_t in = x.shape.back();
            const std::int64_t out = packed.shape[0];
            const std::int64_t groups = packed.shape[1];
            const std::int64_t group = 2 * packed.shape[2];
            if (group < 16 || (group & (group - 1)) != 0) {
                return std::nullopt;
            }
            const ScalarKind kind = dtype == ScalarKind::BF16 ? ScalarKind::F32 : dtype;
            const auto as_kind = [&](const TensorInfo& t) {
                return t.dtype == kind ? t : TensorInfo{convert(t, kind), t.shape, kind};
            };
            const Dims flat{out * groups};
            const TensorInfo& b = packed;
            const TensorInfo scales{reshape(as_kind(scale), flat), flat, kind};
            const TensorInfo zeros{reshape(as_kind(zero), flat), flat, kind};
            microsoft_ = true;
            std::string out_name =
                node("com.microsoft.MatMulNBits",
                     {as_kind(x), b, scales, zeros},
                     "K = " + std::to_string(in) + ", N = " + std::to_string(out) +
                         ", bits = 4, block_size = " + std::to_string(group),
                     shape,
                     kind);
            if (at[4] != nullptr) {
                out_name = node("Add", {{out_name, shape, kind}, as_kind(*at[4])}, "", shape, kind);
            }
            return kind == dtype ? out_name : convert({out_name, shape, kind}, dtype);
        }
        if (implementation_base == "torch.nn.functional.max_pool2d" && operands.size() == 1 &&
            at[0] != nullptr) {
            const auto window = call_generic("K");
            const auto stride = call_generic("Stride");
            const auto pad = call_generic("Pad");
            if (!window || !stride || !pad || 2 * *pad > *window) {
                return std::nullopt;
            }
            // No bf16 pooling in ONNX Runtime either: pool in f32.
            const ScalarKind kind = dtype == ScalarKind::BF16 ? ScalarKind::F32 : dtype;
            const TensorInfo x = at[0]->dtype == kind
                                     ? *at[0]
                                     : TensorInfo{convert(*at[0], kind), at[0]->shape, kind};
            const std::string out =
                node("MaxPool",
                     {x},
                     "kernel_shape = [" + int_list({*window, *window}) + "], strides = [" +
                         int_list({*stride, *stride}) + "], pads = [" +
                         int_list({*pad, *pad, *pad, *pad}) + "]",
                     shape,
                     kind);
            return kind == dtype ? out : convert({out, shape, kind}, dtype);
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
        if (implementation_base == "torch.rms_norm" && operands.size() == 3 && at[0] != nullptr &&
            at[1] != nullptr && at[2] != nullptr && !at[0]->shape.empty()) {
            // In f32, spelled as ONNX Runtime's `SimplifiedLayerNormFusion`
            // matches it (`Pow`, `ReduceMean`, `Add`, `Sqrt`, `Div`), so the
            // session runs one kernel for the whole norm.
            const std::string epsilon = literal_of(at[2]->name);
            if (epsilon.empty()) {
                return std::nullopt;
            }
            const Dims& full = at[0]->shape;
            Dims reduced = full;
            reduced.back() = 1;
            const TensorInfo x =
                at[0]->dtype == ScalarKind::F32
                    ? *at[0]
                    : TensorInfo{convert(*at[0], ScalarKind::F32), full, ScalarKind::F32};
            const TensorInfo exponent = scalar_constant(Literal::of_real(2.0), ScalarKind::F32);
            const TensorInfo squares{
                node("Pow", {x, exponent}, "", full, ScalarKind::F32), full, ScalarKind::F32};
            const TensorInfo mean{node("ReduceMean",
                                       {squares, int64_vector({-1})},
                                       "keepdims = 1",
                                       reduced,
                                       ScalarKind::F32),
                                  reduced,
                                  ScalarKind::F32};
            const TensorInfo eps =
                scalar_constant(Literal::of_real(std::stod(epsilon)), ScalarKind::F32);
            const TensorInfo shifted{
                node("Add", {mean, eps}, "", reduced, ScalarKind::F32), reduced, ScalarKind::F32};
            const TensorInfo root{
                node("Sqrt", {shifted}, "", reduced, ScalarKind::F32), reduced, ScalarKind::F32};
            TensorInfo normalized{
                node("Div", {x, root}, "", full, ScalarKind::F32), full, ScalarKind::F32};
            if (dtype != ScalarKind::F32) {
                normalized = {convert(normalized, dtype), full, dtype};
            }
            const TensorInfo weight = at[1]->dtype == dtype
                                          ? *at[1]
                                          : TensorInfo{convert(*at[1], dtype), at[1]->shape, dtype};
            return node("Mul", {normalized, weight}, "", shape, dtype);
        }
        if (implementation_base == "torch.nn.functional.group_norm" && operands.size() == 4 &&
            at[0] != nullptr && at[1] != nullptr && at[2] != nullptr && at[3] != nullptr &&
            at[0]->shape.size() == 4) {
            // Each group's channels and positions as one row of `[B, G, L]`,
            // which `InstanceNormalization` normalizes in one kernel (with a
            // unit scale and no shift); the channels' own scale and shift
            // follow in the model's dtype, as in the body. ONNX Runtime has
            // no bf16 instance norm: bf16 goes through f32.
            const auto groups = call_generic("Groups");
            const std::string epsilon = literal_of(at[3]->name);
            const Dims& full = at[0]->shape;
            if (!groups || *groups <= 0 || full[1] % *groups != 0 || epsilon.empty()) {
                return std::nullopt;
            }
            const ScalarKind kind = dtype == ScalarKind::BF16 ? ScalarKind::F32 : dtype;
            const Dims rows{full[0], *groups, full[1] / *groups * full[2] * full[3]};
            const TensorInfo x =
                at[0]->dtype == kind ? *at[0] : TensorInfo{convert(*at[0], kind), full, kind};
            const TensorInfo grouped{reshape(x, rows), rows, kind};
            const Dims per_group{*groups};
            const TensorInfo ones{
                node("Expand",
                     {scalar_constant(Literal::of_real(1.0), kind), int64_vector(per_group)},
                     "",
                     per_group,
                     kind),
                per_group,
                kind};
            const TensorInfo zeros{node("Expand",
                                        {scalar_constant(Literal{}, kind), int64_vector(per_group)},
                                        "",
                                        per_group,
                                        kind),
                                   per_group,
                                   kind};
            const TensorInfo normalized{node("InstanceNormalization",
                                             {grouped, ones, zeros},
                                             "epsilon = " + epsilon,
                                             rows,
                                             kind),
                                        rows,
                                        kind};
            TensorInfo restored{reshape(normalized, full), full, kind};
            if (kind != dtype) {
                restored = {convert(restored, dtype), full, dtype};
            }
            const Dims channel{full[1], 1, 1};
            const TensorInfo weight{reshape(*at[1], channel), channel, at[1]->dtype};
            const TensorInfo bias{reshape(*at[2], channel), channel, at[2]->dtype};
            const TensorInfo scaled{node("Mul", {restored, weight}, "", full, dtype), full, dtype};
            return node("Add", {scaled, bias}, "", shape, dtype);
        }
        if (implementation_base == "torch.nn.functional.gelu" && operands.size() == 1 &&
            at[0] != nullptr) {
            return node("Gelu", {*at[0]}, "approximate = \"none\"", shape, dtype);
        }
        if (implementation_base == "torch.nn.functional.gelu(tanh)" && operands.size() == 1 &&
            at[0] != nullptr) {
            return node("Gelu", {*at[0]}, "approximate = \"tanh\"", shape, dtype);
        }
        if ((implementation_base == "torch.Tensor.index_copy" ||
             implementation_base == "torch.Tensor.index_put") &&
            (operands.size() == 3 || operands.size() == 4) && at[0] != nullptr &&
            at[1] != nullptr && at[2] != nullptr && at[0]->shape.size() == 4 &&
            at[1]->shape.size() == 4 && (operands.size() == 3 || at[3] != nullptr)) {
            return cache_write(implementation_base == "torch.Tensor.index_copy", at, shape, dtype);
        }
        if (implementation_base == "torch.sigmoid" && operands.size() == 1 && at[0] != nullptr) {
            return node("Sigmoid", {*at[0]}, "", shape, dtype);
        }
        if (implementation_base == "torch.relu" && operands.size() == 1 && at[0] != nullptr) {
            return node("Relu", {*at[0]}, "", shape, dtype);
        }
        if (implementation_base == "torch.nn.functional.scaled_dot_product_attention" &&
            operands.size() == 5) {
            return attention(at, fast, shape, dtype);
        }
        return std::nullopt;
    }

    // q·kᵀ, scaled, masked, Softmax, ·v. Grouped query heads are folded
    // under their key/value head (`[B, Hk, G * Q, D]`, row `g * Q + q`), so
    // each key and value is read once, not broadcast to every query head.
    // The mask is shared (`[Q, K]`) or per sequence (`[B, Q, K]`). The
    // products run in f32, or in the input dtype for `(input dtype)` (f32
    // for bf16, which ONNX Runtime's `MatMul` lacks); scores and `Softmax`
    // are f32 either way.
    std::optional<std::string> attention(const std::vector<const TensorInfo*>& at,
                                         bool fast,
                                         const Dims& shape,
                                         ScalarKind dtype) {
        const TensorInfo* query = at[0];
        const TensorInfo* key = at[1];
        const TensorInfo* value = at[2];
        const TensorInfo* scale = at[3];
        const TensorInfo* mask = at[4];
        if (query == nullptr || key == nullptr || value == nullptr || scale == nullptr ||
            query->shape.size() != 4 || key->shape.size() != 4 || value->shape.size() != 4 ||
            shape.size() != 4) {
            return std::nullopt;
        }
        const std::int64_t batch = query->shape[0];
        const std::int64_t heads = query->shape[1];
        const std::int64_t queries = query->shape[2];
        const std::int64_t width = query->shape[3];
        const std::int64_t kv_heads = key->shape[1];
        const std::int64_t keys = key->shape[2];
        const std::int64_t value_width = value->shape[3];
        if (kv_heads <= 0 || heads % kv_heads != 0 || key->shape[0] != batch ||
            value->shape[1] != kv_heads || value->shape[2] != keys) {
            return std::nullopt;
        }
        if (mask != nullptr &&
            !(mask->shape == Dims{queries, keys} || mask->shape == Dims{batch, queries, keys})) {
            return std::nullopt;
        }
        const std::int64_t group = heads / kv_heads;
        const std::int64_t rows = group * queries;
        const ScalarKind product =
            fast ? (dtype == ScalarKind::BF16 ? ScalarKind::F32 : dtype) : ScalarKind::F32;
        const auto as = [&](const TensorInfo& t, ScalarKind kind) -> TensorInfo {
            return t.dtype == kind ? t : TensorInfo{convert(t, kind), t.shape, kind};
        };
        TensorInfo q = as(*query, product);
        if (group > 1) {
            const Dims folded{batch, kv_heads, rows, width};
            q = {reshape(q, folded), folded, product};
        }
        const TensorInfo k = as(*key, product);
        const Dims kt_shape{batch, kv_heads, width, keys};
        const TensorInfo kt{transpose(k, {0, 1, 3, 2}, kt_shape), kt_shape, product};
        const Dims scores_shape{batch, kv_heads, rows, keys};
        TensorInfo scores =
            as({node("MatMul", {q, kt}, "", scores_shape, product), scores_shape, product},
               ScalarKind::F32);
        scores = {
            node("Mul", {scores, as(*scale, ScalarKind::F32)}, "", scores_shape, ScalarKind::F32),
            scores_shape,
            ScalarKind::F32};
        if (mask != nullptr) {
            // `[B or 1, 1, Q, K]` against the heads; repeated over the group
            // when a group holds several query rows.
            const std::int64_t mask_batch = mask->shape.size() == 3 ? batch : 1;
            const Dims lifted{mask_batch, 1, queries, keys};
            TensorInfo m{reshape(*mask, lifted), lifted, mask->dtype};
            if (group > 1 && queries > 1) {
                const Dims repeated{mask_batch, group, queries, keys};
                m = {node("Expand", {m, int64_vector(repeated)}, "", repeated, mask->dtype),
                     repeated,
                     mask->dtype};
                const Dims flat{mask_batch, 1, rows, keys};
                m = {reshape(m, flat), flat, mask->dtype};
            }
            const TensorInfo fill = scalar_constant(Literal::of_real(-1e30), ScalarKind::F32);
            scores = {node("Where", {m, scores, fill}, "", scores_shape, ScalarKind::F32),
                      scores_shape,
                      ScalarKind::F32};
        }
        const TensorInfo weights =
            as({node("Softmax", {scores}, "axis = -1", scores_shape, ScalarKind::F32),
                scores_shape,
                ScalarKind::F32},
               product);
        const Dims mixed_shape{batch, kv_heads, rows, value_width};
        std::string mixed =
            node("MatMul", {weights, as(*value, product)}, "", mixed_shape, product);
        if (group > 1) {
            mixed = reshape({mixed, mixed_shape, product}, shape);
        }
        return product == dtype ? mixed : convert({mixed, shape, product}, dtype);
    }

    std::string literal_of(const std::string& name) const {
        const auto found = literals_.find(name);
        return found == literals_.end() ? "" : found->second;
    }

    // ONNX Runtime has kernels for arithmetic, comparisons, and `Where` on
    // 32- and 64-bit integers only: 8- and 16-bit ones compute in i32 and
    // come back (a dequantizer's bit fiddling over uint8 blocks).
    static bool narrow_integer(ScalarKind dtype) {
        return dtype == ScalarKind::U8 || dtype == ScalarKind::I8 || dtype == ScalarKind::U16 ||
               dtype == ScalarKind::I16;
    }

    std::vector<TensorInfo> widened(const std::vector<TensorInfo>& operands) {
        std::vector<TensorInfo> out;
        out.reserve(operands.size());
        for (const TensorInfo& operand : operands) {
            out.push_back(
                narrow_integer(operand.dtype)
                    ? TensorInfo{convert(operand, ScalarKind::I32), operand.shape, ScalarKind::I32}
                    : operand);
        }
        return out;
    }

    std::string elementwise(Elementwise kind,
                            const std::vector<TensorInfo>& operands,
                            const Dims& shape,
                            ScalarKind dtype) override {
        const bool arithmetic =
            kind == Elementwise::Add || kind == Elementwise::Sub || kind == Elementwise::Mul ||
            kind == Elementwise::Div || kind == Elementwise::Rem || kind == Elementwise::Min ||
            kind == Elementwise::Max || kind == Elementwise::Neg || kind == Elementwise::Abs;
        if (arithmetic && narrow_integer(dtype)) {
            const std::string wide = elementwise(kind, widened(operands), shape, ScalarKind::I32);
            return convert({wide, shape, ScalarKind::I32}, dtype);
        }
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
        if (narrow_integer(a.dtype) || narrow_integer(b.dtype)) {
            const std::vector<TensorInfo> wide = widened({a, b});
            return compare(kind, wide[0], wide[1], shape);
        }
        if (a.dtype == ScalarKind::BF16 || b.dtype == ScalarKind::BF16) {
            // No bf16 comparisons either: compare in f32, which is exact.
            const auto f32 = [&](const TensorInfo& t) {
                return t.dtype == ScalarKind::BF16
                           ? TensorInfo{convert(t, ScalarKind::F32), t.shape, ScalarKind::F32}
                           : t;
            };
            return compare(kind, f32(a), f32(b), shape);
        }
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
        if (narrow_integer(dtype)) {
            const std::vector<TensorInfo> wide = widened({on_true, on_false});
            const std::string out =
                node("Where", {condition, wide[0], wide[1]}, "", shape, ScalarKind::I32);
            return convert({out, shape, ScalarKind::I32}, dtype);
        }
        return node("Where", {condition, on_true, on_false}, "", shape, dtype);
    }

    std::string convert(const TensorInfo& value, ScalarKind dtype) override {
        return node("Cast",
                    {value},
                    "to = " + std::to_string(dtype_names(dtype).onnx_code),
                    value.shape,
                    dtype);
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
        const TensorInfo start = scalar_constant(Literal::of_integer(0), ScalarKind::I64);
        const TensorInfo stop = scalar_constant(Literal::of_integer(length), ScalarKind::I64);
        const TensorInfo delta = scalar_constant(Literal::of_integer(1), ScalarKind::I64);
        return node("Range", {start, stop, delta}, "", {length}, ScalarKind::I64);
    }

    // `GatherND`'s result as a `Gather` along one axis: the `K` gathered
    // leading axes of `source` flattened into it, each index row folded
    // into one position. ONNX Runtime's `GatherND` stages its strides
    // through host memory, which a CUDA graph replay reads stale.
    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override {
        const std::size_t depth = static_cast<std::size_t>(indices.shape.back());
        const Dims rows(indices.shape.begin(), indices.shape.end() - 1);
        const auto column = [&](std::size_t j) -> TensorInfo {
            Dims starts(indices.shape.size(), 0);
            Dims limits = indices.shape;
            const Dims strides(indices.shape.size(), 1);
            starts.back() = static_cast<std::int64_t>(j);
            limits.back() = static_cast<std::int64_t>(j) + 1;
            Dims sliced = rows;
            sliced.push_back(1);
            const TensorInfo part{
                slice(indices, starts, limits, strides, sliced), sliced, ScalarKind::I64};
            return {reshape(part, rows), rows, ScalarKind::I64};
        };
        TensorInfo position = column(0);
        std::int64_t flat = source.shape.front();
        for (std::size_t j = 1; j < depth; ++j) {
            const TensorInfo size =
                scalar_constant(Literal::of_integer(source.shape[j]), ScalarKind::I64);
            const TensorInfo scaled{
                node("Mul", {position, size}, "", rows, ScalarKind::I64), rows, ScalarKind::I64};
            position = {
                node("Add", {scaled, column(j)}, "", rows, ScalarKind::I64), rows, ScalarKind::I64};
            flat *= source.shape[j];
        }
        TensorInfo table = source;
        if (depth > 1) {
            Dims flattened{flat};
            flattened.insert(flattened.end(),
                             source.shape.begin() + static_cast<std::ptrdiff_t>(depth),
                             source.shape.end());
            table = {reshape(source, flattened), flattened, source.dtype};
        }
        return node("Gather", {table, position}, "axis = 0", shape, source.dtype);
    }

    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        const TensorInfo axes = int64_vector(dims);
        const bool is_logical = kind == Reduction::Any || kind == Reduction::All;
        TensorInfo source = body;
        if (is_logical) {
            // Boolean reductions go through integers: ONNX reduces numbers.
            source = {convert(body, ScalarKind::I32), body.shape, ScalarKind::I32};
        } else if (body.dtype == ScalarKind::BF16) {
            // ONNX Runtime has no bf16 reductions: reduce in f32.
            source = {convert(body, ScalarKind::F32), body.shape, ScalarKind::F32};
        }
        const char* op = kind == Reduction::Sum    ? "ReduceSum"
                         : kind == Reduction::Prod ? "ReduceProd"
                         : kind == Reduction::Max  ? "ReduceMax"
                         : kind == Reduction::Min  ? "ReduceMin"
                         : kind == Reduction::Any  ? "ReduceMax"
                                                   : "ReduceMin";
        std::string reduced = node(op, {source, axes}, "keepdims = 0", shape, source.dtype);
        if (source.dtype != body.dtype) {
            return convert({reduced, shape, source.dtype}, body.dtype);
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
        // Outputs need names of their own: `output<N>` for the results, and
        // `next_state<N>` for each assigned state member, mapped to its path
        // by `linnet.next_state.<name>`. A value a node computes takes the
        // output's name itself; an `Identity` would copy it, which for a
        // KV cache the graph also reads is the whole cache every call. An
        // input returned as it is, or a value returned twice, goes through
        // `Identity`.
        std::map<std::string, std::string> renamed;
        const auto name_output = [&](const std::string& value, const std::string& name) {
            const bool computed = value.size() > 1 && value[0] == 'v' &&
                                  std::all_of(value.begin() + 1, value.end(), [](char c) {
                                      return c >= '0' && c <= '9';
                                  });
            if (computed && renamed.emplace(value, name).second) {
                return;
            }
            body_ += "    " + name + " = Identity(" + value + ")\n";
        };
        std::vector<std::string> outputs;
        for (std::size_t i = 0; i < results.size(); ++i) {
            const std::string name = "output" + std::to_string(i);
            name_output(results[i].name, name);
            outputs.push_back(tensor_type(results[i].shape, results[i].dtype) + " " + name);
        }
        for (std::size_t i = 0; i < states.size(); ++i) {
            const std::string name = "next_state" + std::to_string(i);
            name_output(states[i].second.name, name);
            outputs.push_back(tensor_type(states[i].second.shape, states[i].second.dtype) + " " +
                              name);
            metadata_.push_back("\"linnet.next_state." + name + "\": \"" + states[i].first + "\"");
        }
        if (!renamed.empty()) {
            body_ = rename_values(body_, renamed);
        }
        const std::string opsets =
            microsoft_ ? "[\"\" : 20, \"com.microsoft\" : 1]" : "[\"\" : 20]";
        std::string out = "<ir_version: 10, opset_import: " + opsets +
                          ", producer_name: \"linnet\", "
                          "doc_string: \"" +
                          entry_label(block_name, entry_name) + " from module " + module_path +
                          "\"";
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

    // `text` with every identifier in `renamed` replaced, whole words only.
    static std::string rename_values(const std::string& text,
                                     const std::map<std::string, std::string>& renamed) {
        const auto identifier = [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                   c == '_';
        };
        std::string out;
        out.reserve(text.size());
        std::size_t i = 0;
        while (i < text.size()) {
            if (!identifier(text[i])) {
                out += text[i++];
                continue;
            }
            std::size_t end = i;
            while (end < text.size() && identifier(text[end])) {
                ++end;
            }
            const std::string word = text.substr(i, end - i);
            const auto found = renamed.find(word);
            out += found == renamed.end() ? word : found->second;
            i = end;
        }
        return out;
    }

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

    // A KV-cache write as one `ScatterND`: the value's vectors go to (row,
    // head, position) triples, and nothing else of the cache is touched --
    // the canonical `Where` rewrites all of it. The four writes differ only
    // in where the rows and positions come from:
    //   write_at / write_span (`index_copy`): every row, positions at + [0, N);
    //   write_rows (`index_put`, 3 operands): row b at at[b];
    //   write_slot (4 operands, a scalar slot): one row, positions at + [0, N);
    //   write_slots (4 operands, slots [M]): rows slots[m], positions at + [0, N).
    std::string cache_write(bool every_row,
                            const std::vector<const TensorInfo*>& at,
                            const Dims& shape,
                            ScalarKind dtype) {
        const TensorInfo& cache = *at[0];
        const TensorInfo& value = *at[1];
        const std::int64_t rows = value.shape[0];
        const std::int64_t heads = value.shape[1];
        const std::int64_t span = value.shape[2];
        const Dims grid{rows, heads, span};
        const auto as_i64 = [&](const TensorInfo& t) {
            return t.dtype == ScalarKind::I64
                       ? t
                       : TensorInfo{convert(t, ScalarKind::I64), t.shape, ScalarKind::I64};
        };
        // `values` laid along `axis` of the grid and broadcast over the rest.
        const auto spread = [&](const TensorInfo& values, std::size_t axis) {
            Dims placed{1, 1, 1};
            placed[axis] = values.shape.empty() ? 1 : values.shape[0];
            const TensorInfo shaped{reshape(values, placed), placed, ScalarKind::I64};
            const TensorInfo target = int64_vector(grid);
            const TensorInfo full{
                node("Expand", {shaped, target}, "", grid, ScalarKind::I64), grid, ScalarKind::I64};
            const Dims last{rows, heads, span, 1};
            return TensorInfo{reshape(full, last), last, ScalarKind::I64};
        };
        const auto span_from = [&](const TensorInfo& start) {
            const TensorInfo offsets{iota(span), {span}, ScalarKind::I64};
            const TensorInfo first = as_i64(start);
            return TensorInfo{
                elementwise(Elementwise::Add, {first, offsets}, {span}, ScalarKind::I64),
                {span},
                ScalarKind::I64};
        };
        if (every_row || at.size() == 3) {
            // Every row of the cache, so each element of `value` scatters
            // along the sequence axis alone (`ScatterElements`, a thread per
            // element): its position, the span's start plus its offset or
            // its row's own. `ScatterND` would take a thread per `[D]` slice
            // and copy it element by element.
            const Dims& full = value.shape;
            const TensorInfo along = every_row ? span_from(*at[2]) : as_i64(*at[2]);
            const Dims placed = every_row ? Dims{1, 1, span, 1} : Dims{rows, 1, 1, 1};
            const TensorInfo shaped{reshape(along, placed), placed, ScalarKind::I64};
            const TensorInfo spread_out{
                node("Expand", {shaped, int64_vector(full)}, "", full, ScalarKind::I64),
                full,
                ScalarKind::I64};
            return node("ScatterElements", {cache, spread_out, value}, "axis = 2", shape, dtype);
        }
        // Rows `slots` (one or several), a span each: an index per `[D]`
        // slice, `(row, head, position)`.
        const TensorInfo head_ids{iota(heads), {heads}, ScalarKind::I64};
        const TensorInfo positions = span_from(*at[3]);
        const TensorInfo slots = as_i64(*at[2]);
        const TensorInfo row_ids =
            slots.shape.empty() ? TensorInfo{reshape(slots, {1}), {1}, ScalarKind::I64} : slots;
        const Dims index_shape{rows, heads, span, 3};
        const TensorInfo indices{
            node("Concat",
                 {spread(row_ids, 0), spread(head_ids, 1), spread(positions, 2)},
                 "axis = 3",
                 index_shape,
                 ScalarKind::I64),
            index_shape,
            ScalarKind::I64};
        return node("ScatterND", {cache, indices, value}, "", shape, dtype);
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
    bool microsoft_ = false;                      // a `com.microsoft` operator is used
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
