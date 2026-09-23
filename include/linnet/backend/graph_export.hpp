#pragma once

#include "linnet/ir/ir.hpp"

#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
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

    // The whole document, once the entry's results are known: the entry's
    // own results, then the final values of the state members it assigned,
    // by path.
    virtual std::string finish(const std::vector<TensorInfo>& results,
                               const std::vector<std::pair<std::string, TensorInfo>>& states,
                               const std::string& module_path,
                               const std::string& block_name,
                               const std::string& entry_name) = 0;
};

struct GraphExportOptions {
    std::string root;  // root block; empty selects the only block with entries
    std::string entry; // entry to export; empty selects the block's only entry
    std::uint32_t root_module = 0;
    std::map<std::string, std::string> bindings; // generic name -> integer or dtype
    bool optionals_present = false;
};

std::expected<std::string, std::string>
export_graph(ir::Module& module, const GraphExportOptions& options, GraphTarget& target);

} // namespace linnet::backend
