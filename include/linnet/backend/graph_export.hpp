#pragma once

#include "linnet/ir/ir.hpp"

#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace linnet::backend {

// Exporting one entry of a root block as a static-shape tensor graph.
//
// `export_graph` evaluates the Core IR with every generic bound to a
// constant: calls are inlined, `static for` is unrolled, `option.match` and
// matches on constant enum values are resolved, and index notation becomes
// broadcasts, gathers, and reductions over the output grid. What remains is
// a sequence of primitive tensor operations, which it hands to a
// `GraphTarget` one at a time; the target names tensors and prints the
// document in its own format (StableHLO, ONNX). Anything the evaluator
// cannot express is a capability failure reported in the error string,
// never approximated.
//
// State is threaded functionally: a `state` member the entry reads becomes
// an extra input (its value before the call), and every state member the
// entry assigns becomes an extra result after the entry's own results (its
// value after the call), both named by parameter path.

using Dims = std::vector<std::int64_t>;

struct TensorInfo {
    std::string name;
    Dims shape; // empty for a scalar
    sema::ScalarKind dtype = sema::ScalarKind::F32;
};

// What an export is of, for the document's header: `Block.entry`, or the
// entry's name alone when it is a module-level one (no block).
inline std::string entry_label(const std::string& block_name, const std::string& entry_name) {
    return block_name.empty() ? entry_name : block_name + "." + entry_name;
}

enum class Elementwise : std::uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Min,
    Max,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl, // shift left by the second operand's bits
    Shr, // shift right: arithmetic for signed dtypes, logical for unsigned
    Not,
    Neg,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tanh,
    Abs,
}; // clang-format: keep one line per group

enum class Reduction : std::uint8_t { Sum, Prod, Max, Min, Any, All };

// A scalar constant. `Lowest`/`Highest` are the identities of max/min.
struct Literal {
    enum class Kind : std::uint8_t { Integer, Real, Boolean, Lowest, Highest };
    Kind kind = Kind::Integer;
    std::int64_t integer = 0;
    double real = 0.0;
};

// One graph format. Each method appends an operation and returns the name
// of its result; operands carry their names and types, results the shape
// and dtype the evaluator computed.
class GraphTarget {
public:
    virtual ~GraphTarget() = default;

    // Entry inputs and block parameters, in argument order.
    virtual std::string
    input(const std::string& name, const Dims& shape, sema::ScalarKind dtype) = 0;
    virtual std::string
    parameter(const std::string& path, const Dims& shape, sema::ScalarKind dtype) = 0;
    // The value of a `state` member before the call.
    virtual std::string
    state(const std::string& path, const Dims& shape, sema::ScalarKind dtype) = 0;

    virtual std::string constant(const Literal& literal, sema::ScalarKind dtype) = 0;
    virtual std::string elementwise(Elementwise kind,
                                    const std::vector<TensorInfo>& operands,
                                    const Dims& shape,
                                    sema::ScalarKind dtype) = 0;
    virtual std::string
    compare(ir::CompareKind kind, const TensorInfo& a, const TensorInfo& b, const Dims& shape) = 0;
    virtual std::string select(const TensorInfo& condition,
                               const TensorInfo& on_true,
                               const TensorInfo& on_false,
                               const Dims& shape,
                               sema::ScalarKind dtype) = 0;
    virtual std::string convert(const TensorInfo& value, sema::ScalarKind dtype) = 0;
    virtual std::string reshape(const TensorInfo& value, const Dims& shape) = 0;
    virtual std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) = 0;
    // Axis i of `value` becomes axis dims[i] of the result; other axes broadcast.
    virtual std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) = 0;

    // Whether the target's elementwise operations broadcast right-aligned
    // operands themselves (PyTorch, NumPy, ONNX do; StableHLO does not). When
    // they do, the exporter leaves those broadcasts out: one emitted
    // operation fewer for every operand of every elementwise operation.
    virtual bool broadcasts_elementwise() const { return false; }
    // The sum of the product of two tensors over the axes the result does
    // not keep, without building the product: each operand's axes are named
    // by grid axis, and the result has `out_axes`, in order. A target without
    // one returns nothing, and the product is built and summed instead.
    virtual std::optional<std::string> contract(const TensorInfo& lhs,
                                                const Dims& lhs_axes,
                                                const TensorInfo& rhs,
                                                const Dims& rhs_axes,
                                                const Dims& out_axes,
                                                const Dims& shape,
                                                sema::ScalarKind dtype) {
        (void)lhs, (void)lhs_axes, (void)rhs, (void)rhs_axes, (void)out_axes, (void)shape;
        (void)dtype;
        return std::nullopt;
    }

    // Placement (see `GraphExportOptions::placement`). A target that supports
    // it is told how many slots there are, which slot the following
    // operations run on, and asked to move a tensor or to drop names.
    virtual bool supports_placement() const { return false; }
    virtual void enable_placement(int slots) { (void)slots; }
    virtual void set_slot(int slot) { (void)slot; }
    virtual std::string transfer(const TensorInfo& value, int slot) {
        (void)slot;
        return value.name;
    }
    virtual void release(const std::vector<std::string>& names) { (void)names; }
    // Fully sharded parameters (see `GraphExportOptions::fully_shard`): a
    // target that supports them gathers a parameter's parts into the whole.
    virtual bool supports_fully_shard() const { return false; }
    virtual std::string gather(const TensorInfo& value) { return value.name; }
    // Recomputed blocks (see `GraphExportOptions::remat`): the code between
    // `begin_remat` and `end_remat` is one call of a listed block, which the
    // backward pass computes again instead of keeping its values.
    virtual bool supports_remat() const { return false; }
    virtual void begin_remat() {}
    virtual void end_remat() {}
    virtual std::string slice(const TensorInfo& value,
                              const Dims& starts,
                              const Dims& limits,
                              const Dims& strides,
                              const Dims& shape) = 0;
    virtual std::string
    concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) = 0;
    // 0, 1, ..., length - 1 as i64.
    virtual std::string iota(std::int64_t length) = 0;
    // `indices` is [..., rank(source)] of i64 positions; the result has the
    // leading shape of `indices`.
    virtual std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) = 0;
    // Reduces `body` over `dims` (trailing axes) to `shape`.
    virtual std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) = 0;
    // A semantic call whose selected candidate (`opt::select_candidates`) is
    // `implementation`, with its tensor operands (absent optionals as
    // nullopt). A target that has the implementation returns the result's
    // name; otherwise the call's canonical body is exported instead.
    // The dimension generics of the semantic call being lowered, by name
    // (`Stride`, `Pad`), set before each `native_call`. A kernel whose
    // arguments are not tensors reads them here instead of inferring them
    // from shapes, which is ambiguous: a 3x3 window taking 4 positions to 2
    // fits both stride 1 without padding and stride 2 with one.
    void set_call_generics(std::map<std::string, std::int64_t> generics) {
        call_generics_ = std::move(generics);
    }
    std::optional<std::int64_t> call_generic(const std::string& name) const {
        const auto found = call_generics_.find(name);
        return found == call_generics_.end() ? std::nullopt : std::optional(found->second);
    }

    // A convolution's window geometry, from the call's own generics: one
    // stride and one padding per spatial axis. `conv1d` and the square
    // `conv2d` take `Stride` and `Pad` for every axis, `conv2d_rect` takes
    // `StrideH`, `StrideW`, `PadH`, and `PadW`. Without them, nothing.
    struct ConvWindow {
        std::vector<std::int64_t> strides;
        std::vector<std::int64_t> pads;
    };
    std::optional<ConvWindow> conv_window(const std::string& implementation) const {
        if (implementation == "torch.nn.functional.conv2d(rect)") {
            const auto sh = call_generic("StrideH");
            const auto sw = call_generic("StrideW");
            const auto ph = call_generic("PadH");
            const auto pw = call_generic("PadW");
            if (!sh || !sw || !ph || !pw) {
                return std::nullopt;
            }
            return ConvWindow{{*sh, *sw}, {*ph, *pw}};
        }
        const std::size_t spatial = implementation == "torch.nn.functional.conv1d" ? 1 : 2;
        const auto stride = call_generic("Stride");
        const auto pad = call_generic("Pad");
        if (!stride || !pad) {
            return std::nullopt;
        }
        return ConvWindow{std::vector<std::int64_t>(spatial, *stride),
                          std::vector<std::int64_t>(spatial, *pad)};
    }

    static bool is_convolution(const std::string& implementation) {
        return implementation == "torch.nn.functional.conv2d" ||
               implementation == "torch.nn.functional.conv2d(rect)" ||
               implementation == "torch.nn.functional.conv1d";
    }

    virtual std::optional<std::string>
    native_call(const std::string& implementation,
                const std::vector<std::optional<TensorInfo>>& operands,
                const Dims& shape,
                sema::ScalarKind dtype) {
        (void)implementation;
        (void)operands;
        (void)shape;
        (void)dtype;
        return std::nullopt;
    }

    // Runtime loops. The evaluator carries the loop's values (the `while`
    // operands, then every state member) through three calls: `begin_while`
    // opens the loop and names the values the condition sees; after the
    // condition is emitted, `while_condition` names the values the body
    // sees; after the body, `end_while` closes the loop and names the final
    // values. A target whose loop form checks the condition after the body
    // (ONNX `Loop`) asks for the predicate before the loop and again after
    // the body through the two `needs_` hooks.
    virtual bool supports_while() const { return false; }
    virtual bool while_needs_initial_condition() const { return false; }
    virtual void while_initial_condition(const TensorInfo& predicate) { (void)predicate; }
    virtual std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) {
        (void)initial;
        return {};
    }
    virtual std::vector<std::string> while_condition(const TensorInfo& predicate) {
        (void)predicate;
        return {};
    }
    virtual bool while_needs_trailing_condition() const { return false; }
    virtual void while_trailing_condition(const TensorInfo& predicate) { (void)predicate; }
    virtual std::vector<std::string> end_while(const std::vector<TensorInfo>& next) {
        (void)next;
        return {};
    }

    // The whole document, once the entry's results are known: the entry's
    // own results, then the final values of the state members it assigned,
    // by path.
    virtual std::string finish(const std::vector<TensorInfo>& results,
                               const std::vector<std::pair<std::string, TensorInfo>>& states,
                               const std::string& module_path,
                               const std::string& block_name,
                               const std::string& entry_name) = 0;

private:
    std::map<std::string, std::int64_t> call_generics_;
};

// Whether `text` matches the glob `pattern`: `*` any run of characters,
// `?` any one. LoRA patterns name the weights they adapt this way.
bool glob_match(std::string_view pattern, std::string_view text);

struct GraphExportOptions {
    std::string root;  // root block; empty selects the only block with entries
    std::string entry; // entry to export; empty selects the block's only entry
    std::uint32_t root_module = 0;
    std::map<std::string, std::string> bindings; // generic name -> integer or dtype
    bool optionals_present = false;
    // Optional parameters that are absent even though `optionals_present`
    // says otherwise, by path (`classifier.bias`, `layers.3.k_proj.bias`).
    // Real checkpoints mix them -- a ResNet's convolutions have no bias while
    // its classifier has one -- and one switch for the whole model would
    // either drop the classifier's bias or ask for convolution biases the
    // checkpoint does not have.
    std::set<std::string> absent;
    // Placement across devices, for a target that supports it. A block whose
    // member path starts with a key runs on that device slot (the longest key
    // wins; everything else runs on slot 0), and a tensor consumed on another
    // slot is transferred once. Keys end in `.`, like `layers.3.`.
    std::map<std::string, int> placement;
    // Blocks whose parameters stay on the host: each is transferred to the
    // block's slot when first used and released when the block returns.
    std::vector<std::string> offload;
    // Blocks whose parameters are split across processes (fully sharded
    // data parallelism): each is gathered whole when first used and released
    // when the block returns. One device per process, so not with placement
    // or offload.
    std::vector<std::string> fully_shard;
    // Generated JAX only: blocks whose every call the backward pass computes
    // again (`jax.checkpoint`) instead of keeping its values. What it keeps
    // is what the call reads; a weight the call gathers is gathered again.
    // Paths as for `fully_shard`.
    std::vector<std::string> remat;
    // Generated Python only: split computation that reads nothing but
    // parameters (dequantizing MXFP4 experts, say) into a `prepare` function
    // the runtime calls once per loaded model rather than on every call.
    bool prepare = false;
    // Generated PyTorch only, with `prepare`: sibling linear layers run as
    // one product over their joined weights. Weights split over devices
    // (tensor parallelism) are joined only by gathering them, so a runtime
    // that splits them turns this off.
    bool fuse = true;
    // Generated PyTorch only: low-rank adapters (LoRA) on the weights whose
    // path matches one of these glob patterns (`layers.*.attention.*.weight`).
    // A `linear` over such a weight adds `(x @ A.T) @ B.T * alpha / rank`,
    // with `A` and `B` parameters of the weight's block: `lora_a`
    // [rank, in] and `lora_b` [out, rank]. Needs `prepare` off: a joined or
    // prepared weight is no longer the parameter the pattern names.
    std::vector<std::string> lora;
    std::int64_t lora_rank = 0;
    double lora_alpha = 0.0;
    // StableHLO and generated JAX only: f32 products (matrix products,
    // convolutions, attention) at full f32 precision. XLA otherwise runs
    // them at the device's default, which on an NVIDIA GPU since Ampere is
    // TF32: a 10-bit mantissa, errors near 1e-3 against the canonical body.
    // Every `--numerics` but `fast` sets it.
    bool full_precision = false;
};

std::expected<std::string, std::string>
export_graph(ir::Module& module, const GraphExportOptions& options, GraphTarget& target);

// Generated Python (the `torch` and `jax` targets): drops `vN = ...` lines
// whose value nothing later mentions, including `live_tail` (the return
// statement). Uses always follow definitions in the emitted text, so one
// backward pass suffices.
std::string prune_python_assignments(const std::string& body, const std::string& live_tail);

// Generated Python (the `torch` target): `del`s each value right after the
// statement that uses it last, so an entry holds what is still needed rather
// than every intermediate until it returns -- which for a deep model is the
// difference between one layer's activations and all of them. Only values a
// top-level statement assigns are released (a loop body may run zero times);
// a use inside a loop counts at the loop's end; `live_tail` is never released.
std::string release_dead_values(const std::string& body, const std::string& live_tail);

// `"ik,kj->ij"` for a contraction whose operands and result name grid axes:
// one letter per axis, in order of first appearance.
// Weight-only computation split out of a generated Python body so that it
// runs once, at load, rather than on every call (see `GraphExportOptions::
// prepare`). A top-level `vN = ...` line is weight-only when it reads nothing
// but parameters (`pN`), constants defined in `constants` (the lines the
// target already hoisted), other weight-only values, and library names. Only
// chains that compute something are split out: a value that is a parameter
// seen through views and casts stays inline, since preparing it would just
// copy the weights.
struct PreparedSplit {
    std::string prepare;              // the weight-only lines, as they were written
    std::vector<std::string> inputs;  // what they read: `pN` and constant `vN` names
    std::vector<std::string> outputs; // the values the rest reads, which `prepare` returns
    // One per output: a hash of the computation over parameter paths, equal
    // for equal computations in any entry, so a runtime shares the result.
    std::vector<std::string> keys;
    std::string body; // everything else, in order
};
PreparedSplit split_prepared(const std::string& body,
                             const std::string& live_tail,
                             const std::vector<std::string>& parameter_paths,
                             const std::string& constants);

std::string einsum_equation(const Dims& lhs_axes, const Dims& rhs_axes, const Dims& out_axes);

// Python literals the source targets (`torch`, `jax`) print: a tuple of
// sizes, `(2, 3)` or `(4,)`, and a float with every digit kept and a
// decimal point (`1.0`, not `1`).
std::string python_tuple(const Dims& dims);
std::string python_float(double value);

// How each format names a scalar dtype: MLIR (`bf16`, `ui8`), the ONNX text
// format (`bfloat16`, `double`) and its `TensorProto.DataType` number, and
// NumPy and the libraries that follow it (`bfloat16`, `float64`).
struct DTypeNames {
    std::string_view mlir;
    std::string_view onnx;
    int onnx_code = 0;
    std::string_view numpy;
};
const DTypeNames& dtype_names(sema::ScalarKind dtype);

// Python text the source targets print, read back: whether `c` can be part
// of an identifier; whether `word` is a generated name with `prefix` (`v3`,
// `p0`); every identifier-like word of `text` with its position (numbers
// are not words); and the `vN` values `text` mentions.
bool word_char(char c);
bool numbered(std::string_view word, char prefix);
std::vector<std::pair<std::size_t, std::string>> words_of(std::string_view text);
std::set<std::string> value_names(std::string_view text);

} // namespace linnet::backend
