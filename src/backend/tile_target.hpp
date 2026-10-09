#pragma once

// What the kernel targets (Triton, Pallas) share: tiles as Python arrays
// of a library that spells most operations alike (`tl.where`, `jnp.where`),
// one name per distinct expression, constants as Python literals. A target
// spells the rest and its memory, loops, and program.

#include "linnet/backend/kernel.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace linnet::backend {

class TileTarget : public KernelTarget {
public:
    std::string input(const std::string& name, const Dims& shape, sema::ScalarKind dtype) override;
    std::string
    parameter(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
    std::string state(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override;

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
    bool broadcasts_elementwise() const override { return true; }
    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        sema::ScalarKind dtype) override;
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
    void release(const std::vector<std::string>& names) override { (void)names; }
    bool supports_while() const override { return false; }
    bool supports_counted() const override { return true; }

    KernelProgram program() override { return {parameters_, body_}; }

protected:
    TileTarget(std::string library, bool full_precision)
        : library_(std::move(library)), full_precision_(full_precision) {}

    // ---- what each library spells its own way

    virtual std::string dtype_name(sema::ScalarKind dtype) const = 0;
    virtual std::string cast(const std::string& value, sema::ScalarKind dtype) const = 0;
    virtual std::string arange(std::int64_t length) const = 0;
    virtual std::string permute(const std::string& value, const Dims& permutation) const = 0;
    // Integer division and remainder, rounding toward zero.
    virtual std::string
    divide_integers(const std::string& a, const std::string& b, sema::ScalarKind dtype) const = 0;
    virtual std::string
    remainder(const std::string& a, const std::string& b, sema::ScalarKind dtype) const = 0;
    virtual std::string rsqrt(const std::string& a) const = 0;
    virtual std::string tanh(const std::string& a) const = 0;
    // `a @ b`, two 2-D tiles, accumulated in `f32`.
    virtual std::string
    dot(const std::string& a, const std::string& b, bool full_precision) const = 0;
    // ---- shared

    std::string define(const std::string& expression);
    // A parameter's name, unique and not a Python word.
    std::string unique(const std::string& name);
    static std::string shape_list(const Dims& shape);
    // `x[None, :]`: a tile of `own` axes placed at `position` among `rank`.
    static std::string place(std::size_t position, std::size_t own, std::size_t rank);

    std::string library_; // `tl`, `jnp`
    bool full_precision_ = false;
    std::vector<std::string> parameters_;
    std::set<std::string> used_;
    std::set<std::string> literals_;
    std::vector<std::map<std::string, std::string>> scopes_{1};
    std::string body_;
    std::string indent_ = "    ";
    std::size_t next_ = 0;
    std::size_t loops_ = 0;
};

} // namespace linnet::backend
