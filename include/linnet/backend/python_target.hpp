#pragma once

#include "linnet/backend/graph_export.hpp"

#include <cstddef>
#include <cstdint>
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
    bool broadcasts_elementwise() const override { return true; }
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
    std::string literal_text(const Literal& literal, sema::ScalarKind dtype) const;
    // The parameter path of `name` when it is a weight an adapter pattern matches.
    std::optional<std::string> lora_target(const std::string& name) const;
    // The adapter parameters of the weight at `path` ([out, in]): `lora_a`
    // [rank, in] and `lora_b` [out, rank] of its block, made once. They go
    // after the other parameters among `main`'s arguments.
    std::pair<std::string, std::string>
    adapters(const std::string& path, const Dims& weight, sema::ScalarKind dtype);
    static std::string string_list(const std::vector<std::string>& items);

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
    std::map<std::string, std::pair<std::string, std::string>> adapters_; // weight -> a, b
    std::map<std::string, std::string> gathered_;            // gathered value -> its argument
    std::set<std::string> causal_masks_;                     // square masks from `causal_mask`
    std::vector<std::map<std::string, std::string>> cse_{1}; // expression -> name, per scope
    std::string body_;
    std::string indent_ = "    ";
    std::size_t next_ = 0;
};

} // namespace linnet::backend
