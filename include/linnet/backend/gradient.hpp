#pragma once

#include "linnet/backend/graph_export.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace linnet::backend {

// Reverse-mode differentiation of an exported entry (`GraphExportOptions::
// gradient`).
//
// `GradientTarget` stands between the evaluator and a graph format. Every
// operation passes through to the format and is recorded. When the entry
// finishes, the backward pass of its result -- one floating scalar, a loss --
// is emitted through the same format, from the last operation to the first,
// and the export returns the loss followed by its gradient with respect to
// every floating parameter (a block's entry) or every floating input (a
// module-level entry), in the order the export takes them; the format is
// told their paths (`GraphTarget::gradients`). Library calls run as their
// canonical bodies, whose primitive operations all have rules. `while`
// loops are refused. An entry that assigns `state` returns its new values
// after the gradients, as its forward export does.
class GradientTarget final : public GraphTarget {
public:
    explicit GradientTarget(GraphTarget& inner) : inner_(inner) {}

    std::string input(const std::string& name, const Dims& shape, sema::ScalarKind dtype) override;
    std::string
    parameter(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
    std::string state(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
    std::string constant(const Literal& literal, sema::ScalarKind dtype) override;
    std::string elementwise(Elementwise kind,
                            const std::vector<TensorInfo>& operands,
                            const Dims& shape,
                            sema::ScalarKind dtype) override;
    std::string compare(ir::CompareKind kind,
                        const TensorInfo& a,
                        const TensorInfo& b,
                        const Dims& shape) override;
    std::string select(const TensorInfo& condition,
                       const TensorInfo& on_true,
                       const TensorInfo& on_false,
                       const Dims& shape,
                       sema::ScalarKind dtype) override;
    std::string convert(const TensorInfo& value, sema::ScalarKind dtype) override;
    std::string reshape(const TensorInfo& value, const Dims& shape) override;
    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override;
    std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) override;
    bool broadcasts_elementwise() const override { return inner_.broadcasts_elementwise(); }
    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        sema::ScalarKind dtype) override;
    // The backward pass reads what the forward one computed: nothing is
    // released.
    void release(const std::vector<std::string>& names) override { (void)names; }
    std::string slice(const TensorInfo& value,
                      const Dims& starts,
                      const Dims& limits,
                      const Dims& strides,
                      const Dims& shape) override;
    std::string
    concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) override;
    std::string iota(std::int64_t length) override;
    std::string cumsum(const TensorInfo& value, std::int64_t axis) override;
    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override;
    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override;
    // Library kernels have no backward pass here: the canonical bodies run.
    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           sema::ScalarKind dtype) override;
    bool supports_while() const override { return inner_.supports_while(); }
    std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) override;
    // A `for` loop, written in the format's counted form or as a `while`
    // over its index. Each iteration's starting values are kept, and the
    // backward pass runs the iterations in reverse, computing each again.
    bool supports_counted() const override { return true; }
    // An op's `grad` stands for the backward pass of its body.
    std::vector<std::string>
    begin_custom_gradient(const std::vector<TensorInfo>& arguments) override;
    std::string end_custom_gradient(const std::vector<TensorInfo>& arguments,
                                    const TensorInfo& result,
                                    const Pullback& pullback) override;
    std::vector<std::string> begin_counted(std::int64_t start,
                                           std::int64_t stop,
                                           const std::vector<TensorInfo>& initial) override;
    std::vector<std::string> end_counted(const std::vector<TensorInfo>& next) override;
    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override;

private:
    // One recorded operation that has a backward rule.
    struct Step {
        enum class Kind : std::uint8_t {
            Elementwise,
            Select,
            Convert,
            Reshape,
            Transpose,
            Broadcast,
            Contract,
            Slice,
            Concat,
            Gather,
            Reduce,
            Custom, // an op's `grad`: `pullbacks_[custom]`
            Loop,   // a `for` loop: `loops_[custom]`
            Cumsum, // along `axis`
        };
        Kind kind = Kind::Elementwise;
        Elementwise op = Elementwise::Add;
        Reduction reduction = Reduction::Sum;
        std::vector<TensorInfo> operands;
        TensorInfo result;
        Dims first;  // broadcast dims, permutation, reduced axes, lhs axes, slice starts
        Dims second; // contract rhs axes, slice limits
        Dims third;  // contract result axes, slice strides
        std::int64_t axis = 0;
        std::size_t custom = 0;
    };

    // One `for` loop as recorded: its values before, inside and after it,
    // each iteration's starting values (`stacks`, `[iterations, ...]` each),
    // and its body, as backward rules and as calls to emit again.
    struct Replay {
        std::function<std::string()> call;
        std::string result;
    };
    struct Loop {
        std::int64_t start = 0;
        std::int64_t stop = 0;
        std::vector<TensorInfo> initial;
        TensorInfo index;
        std::vector<TensorInfo> carried; // as the body sees them
        std::vector<TensorInfo> next;    // as the body leaves them
        std::vector<TensorInfo> finals;
        std::vector<TensorInfo> stacks; // after the loop; inside it, `written`
        std::vector<TensorInfo> written;
        std::size_t tape_mark = 0;
        std::vector<Step> steps;
        std::vector<Replay> log;
    };
    // A format's loop over `start <= i < stop`: its counted form, or a
    // `while` carrying the index first. The names the body sees, index
    // first; then the final values.
    struct Lowered {
        bool is_counted = false;
        std::int64_t stop = 0;
        std::string index;
    };
    std::vector<std::string>
    open_loop(std::int64_t start, std::int64_t stop, const std::vector<TensorInfo>& initial);
    std::vector<std::string> close_loop(const std::vector<TensorInfo>& next);
    void propagate_loop(const Loop& loop);
    // Inside a loop being recorded, `call` emits the operation that made
    // `result` again, from the renamed values of an iteration.
    void replayable(const std::string& result, std::function<std::string()> call);
    std::string renamed(const std::string& name) const;
    TensorInfo renamed(TensorInfo value) const;
    std::vector<TensorInfo> renamed(std::vector<TensorInfo> values) const;
    TensorInfo zeros(const Dims& shape, sema::ScalarKind dtype);
    TensorInfo scalar_integer(std::int64_t value);

    std::string record(Step step);
    // Notes where `name` is first defined, with its shape.
    void define(const std::string& name, const Dims& shape);
    void backward(const TensorInfo& loss);
    void propagate(const Step& step, const TensorInfo& grad);
    void accumulate(const TensorInfo& value, const TensorInfo& grad);

    // Emission helpers: every operand made the result's full shape.
    TensorInfo full(double value, const Dims& shape, sema::ScalarKind dtype);
    TensorInfo apply(Elementwise kind, const std::vector<TensorInfo>& operands);
    TensorInfo
    choose(const TensorInfo& condition, const TensorInfo& on_true, const TensorInfo& on_false);
    TensorInfo widen(const TensorInfo& value, const Dims& shape);
    // `grad`, of a result of `shape`, summed down to `operand`'s shape: the
    // backward pass of an implicit (right-aligned) broadcast.
    TensorInfo unbroadcast(const TensorInfo& grad, const TensorInfo& operand);
    // Sums `value` over `axes`, keeping the others in order.
    TensorInfo sum_over(const TensorInfo& value, const std::vector<std::size_t>& axes);
    TensorInfo product(const TensorInfo& lhs,
                       const Dims& lhs_axes,
                       const TensorInfo& rhs,
                       const Dims& rhs_axes,
                       const Dims& out_axes,
                       const Dims& shape,
                       sema::ScalarKind dtype);

    GraphTarget& inner_;
    std::vector<Step> tape_;
    std::vector<TensorInfo> inputs_;
    std::vector<std::string> input_names_;
    std::vector<TensorInfo> parameters_;
    std::vector<std::string> parameter_paths_;
    std::map<std::string, TensorInfo> adjoints_;
    // Every value's shape where it was defined, and the order of
    // definitions.
    std::map<std::string, Dims> shapes_;
    std::map<std::string, std::size_t> order_;
    // Open `grad` ops: where each began, on the tape and among definitions.
    std::vector<std::pair<std::size_t, std::size_t>> customs_;
    std::vector<Pullback> pullbacks_;
    std::vector<Loop> loops_;
    std::vector<std::size_t> open_;              // the loop being recorded
    std::vector<Lowered> lowered_;               // the format's loops being written
    std::map<std::string, std::string> renames_; // an iteration's names, while it is emitted again
    // The backward pass is being emitted: the evaluator's calls are part of
    // it, not recorded.
    bool replaying_ = false;
    // Which differentiated values -- floating inputs, floating parameters --
    // each value is computed from; the backward pass follows only those
    // `wanted_`.
    static constexpr std::uint8_t from_inputs = 1;
    static constexpr std::uint8_t from_parameters = 2;
    std::map<std::string, std::uint8_t> sources_;
    std::uint8_t wanted_ = 0;
};

// What `GradientTarget` cannot differentiate; `export_graph` reports it.
struct GradientError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace linnet::backend
