#pragma once

#include "linnet/backend/graph_export.hpp"

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
// canonical bodies, whose primitive operations all have rules. Runtime
// loops and entries that assign `state` are refused.
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
    bool supports_counted() const override { return inner_.supports_counted(); }
    std::vector<std::string> begin_counted(std::int64_t start,
                                           std::int64_t stop,
                                           const std::vector<TensorInfo>& initial) override;
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
    };

    std::string record(Step step);
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
};

// What `GradientTarget` cannot differentiate; `export_graph` reports it.
struct GradientError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace linnet::backend
