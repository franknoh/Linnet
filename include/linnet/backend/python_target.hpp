#pragma once

#include "linnet/backend/graph_export.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace linnet::backend {

// What the targets that print Python (`linnet torch`, `linnet jax`) share:
// `main`'s arguments, one name per distinct expression, scalar constants
// folded to literals, low-rank adapters, and the operations both libraries
// spell alike. A target names its library (`torch`, `jnp`) and spells the
// rest.
class PythonTarget : public GraphTarget {
public:
    std::string input(const std::string& name, const Dims& shape, sema::ScalarKind dtype) override;
    std::string
    parameter(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
    std::string state(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
    std::string constant(const Literal& literal, sema::ScalarKind dtype) override;
    std::string elementwise(Elementwise kind,
                            const std::vector<TensorInfo>& operands,
                            const Dims& shape,
                            sema::ScalarKind dtype) override;
    std::string select(const TensorInfo& condition,
                       const TensorInfo& on_true,
                       const TensorInfo& on_false,
                       const Dims& shape,
                       sema::ScalarKind dtype) override;
    std::string reshape(const TensorInfo& value, const Dims& shape) override;
    std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) override;
    std::string slice(const TensorInfo& value,
                      const Dims& starts,
                      const Dims& limits,
                      const Dims& strides,
                      const Dims& shape) override;
    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override;
    std::string cumsum(const TensorInfo& value, std::int64_t axis) override;
    bool broadcasts_elementwise() const override { return true; }
    // An op with a `grad`: its body computes from detached arguments, so
    // that the library's autodiff takes the op's gradient from a function
    // the target defines at module level (`end_custom_gradient`).
    std::vector<std::string>
    begin_custom_gradient(const std::vector<TensorInfo>& arguments) override;
    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        sema::ScalarKind dtype) override;

protected:
    // Adapters (`--lora`): the patterns over weight paths, their rank and alpha.
    struct Lora {
        std::vector<std::string> patterns;
        std::int64_t rank = 0;
        double alpha = 0.0;
    };

    PythonTarget(std::string library, std::string bool_name, bool prepare, Lora lora);

    // ---- what each library spells its own way

    // A constant tensor of `dtype` holding Python literal `text`.
    virtual std::string constant_expression(const std::string& text, sema::ScalarKind dtype) = 0;
    // Integer division, rounding toward zero.
    virtual std::string divide_integers(const std::string& a, const std::string& b) = 0;
    virtual std::string shift(const std::string& a, const std::string& b, bool left) = 0;
    virtual std::string reciprocal_sqrt(const std::string& a) = 0;
    // `value` broadcast to `shape`, each axis it has of size one or the full size.
    virtual std::string expand(const std::string& value, const Dims& shape) = 0;
    // Positive infinity, as Python text.
    virtual std::string infinity() const = 0;
    // `value` cut from the library's autodiff.
    virtual std::string detach(const std::string& value) const = 0;

    // ---- shared

    // `torch.float32`, `jnp.bfloat16`.
    std::string dtype_name(sema::ScalarKind dtype) const;
    // Every value is immutable, so an expression already computed in this
    // scope (or an enclosing one) names the same tensor: identical rotary
    // tables or masks across inlined layers are emitted once.
    std::string define(const std::string& expression);
    std::string spell(Elementwise kind, const std::vector<TensorInfo>& operands);
    // Scalar arithmetic on constants is folded to a Python literal as well,
    // so a computed `rsqrt(cast<f32>(D))` reaches a kernel as `scale=...`
    // rather than as a tensor the compiler has to read back.
    void fold(const std::string& name,
              Elementwise kind,
              const std::vector<TensorInfo>& operands,
              sema::ScalarKind dtype);
    // `name`, `value` converted to `dtype`, folded as well when `value` was:
    // rounded to f32's precision, or truncated toward zero to an integer
    // the dtype holds.
    void fold_convert(const std::string& name, const TensorInfo& value, sema::ScalarKind dtype);
    std::string literal_text(const Literal& literal, sema::ScalarKind dtype) const;
    // The highest value of `dtype`, or the lowest.
    std::string extreme(bool highest, sema::ScalarKind dtype) const;
    // The parameter path of `name` when it is a weight an adapter pattern matches.
    std::optional<std::string> lora_target(const std::string& name) const;
    // The adapter parameters of the weight at `path` ([out, in]): `lora_a`
    // [rank, in] and `lora_b` [out, rank] of its block, made once. They go
    // after the other parameters among `main`'s arguments.
    std::pair<std::string, std::string>
    adapters(const std::string& path, const Dims& weight, sema::ScalarKind dtype);
    static std::string string_list(const std::vector<std::string>& items);

    // ---- kernels, for a target's `launch_kernel`

    // The op's `_opN` at the launch's argument shapes and dtypes: N, and
    // whether it is new (the target then defines `_opN`).
    std::pair<std::size_t, bool> op_of(const KernelLaunch& launch);
    // The function of the launch's program, `_kernelK`, and whether it is
    // new (the target then defines it).
    std::pair<std::string, bool> kernel_of(const KernelLaunch& launch);
    // `_opN(...)` on the launch's arguments: its results' names.
    std::vector<std::string> call_op(const KernelLaunch& launch, std::size_t op);
    // The reduction result `i` is written with atomically, if it is.
    static std::optional<Reduction> atomic_result(const KernelLaunch& launch, std::size_t i);
    // Where an atomic `reduction` over `dtype` starts: its identity.
    std::string reduction_identity(Reduction reduction, sema::ScalarKind dtype) const;

    // An op's backward pass, emitted at `indent` in a scope of its own: the
    // arguments named `a0`, `a1`, ..., the result `out`, its gradient
    // `grad_out`. Nothing in `body` reads a name of `main`'s.
    struct Backward {
        std::string body;
        std::vector<std::optional<std::string>> gradients; // per argument
        std::vector<std::string> arguments;                // `a0`, `a1`, ...
    };
    Backward emit_backward(const std::vector<TensorInfo>& arguments,
                           const TensorInfo& result,
                           const Pullback& pullback,
                           const std::string& indent);
    // An op's body, emitted at `indent` in a scope of its own from the
    // arguments `a0`, `a1`, ...: the text and its results' names.
    std::pair<std::string, std::vector<std::string>>
    emit_body(const std::vector<TensorInfo>& arguments,
              const std::function<std::vector<std::string>(const std::vector<TensorInfo>&)>& body,
              const std::string& indent);

    std::string library_;   // `torch`, `jnp`
    std::string bool_name_; // the library's boolean dtype, after `library_.`
    bool prepare_ = false;  // split weight-only work into `prepare`
    Lora lora_;
    std::vector<std::string> arguments_;
    std::size_t parameters_end_ = 0;      // `arguments_` past the last parameter
    std::vector<std::string> parameters_; // paths, in argument order
    std::vector<std::pair<Dims, sema::ScalarKind>> parameter_types_; // shape and dtype, in order
    std::vector<std::string> states_;                                // paths, in argument order
    std::map<std::string, std::string> literals_; // constant name -> Python literal
    std::map<std::string, double> values_;        // constant name -> folded value
    // An `expand` of a smaller tensor -> that tensor, with the expanded value's
    // dtype: elementwise operations on such values run at the smaller shape
    // and expand their result, so a weight's or a constant's arithmetic is
    // not materialized at the activations' shape (`prepare` would keep it).
    std::map<std::string, TensorInfo> broadcasts_;
    std::map<std::string, std::pair<std::string, std::string>> adapters_; // weight -> a, b
    std::map<std::string, std::string> gathered_;            // gathered value -> its argument
    std::set<std::string> causal_masks_;                     // square masks from `causal_mask`
    std::vector<std::map<std::string, std::string>> cse_{1}; // expression -> name, per scope
    std::string body_;
    std::string indent_ = "    ";
    // Module-level definitions before `main`: ops' gradients and kernels,
    // after the `helpers_` they use.
    std::string helpers_;
    std::string definitions_;
    std::map<std::string, std::string> kernels_;  // a kernel's program -> its function
    std::map<std::string, std::size_t> launched_; // an op at its shapes -> its `_opN`
    std::size_t launches_ = 0;
    std::size_t custom_gradients_ = 0;
    std::size_t next_ = 0;
};

// Options the generated Python cannot honour together, as the error to
// report: adapters on prepared weights (a joined or prepared weight is no
// longer the parameter an adapter names), adapters without a rank, and
// sharded weights with `prepare` (it would gather each whole once and keep
// it). Empty when they fit.
std::optional<std::string> python_options_conflict(const GraphExportOptions& options);

} // namespace linnet::backend
