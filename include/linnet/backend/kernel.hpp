#pragma once

#include "linnet/backend/graph_export.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace linnet::backend {

// A format for kernels (spec §7.9): the graph operations, over tiles, and
// the memory a kernel reads and writes.
class KernelTarget : public GraphTarget {
public:
    // A tensor in memory: a parameter, or a result the kernel writes.
    virtual std::string
    memory(const std::string& name, const Dims& shape, sema::ScalarKind dtype, bool is_result) = 0;
    // A scalar parameter.
    virtual std::string scalar(const std::string& name, sema::ScalarKind dtype) = 0;
    virtual std::string program_id(std::int64_t axis) = 0;
    // `load(memory[indices], mask, other)`: each index tile adds its axes
    // to the result, of `shape`.
    virtual std::string load(const TensorInfo& memory,
                             const std::vector<TensorInfo>& indices,
                             const std::optional<TensorInfo>& mask,
                             const std::optional<TensorInfo>& other,
                             const Dims& shape,
                             sema::ScalarKind dtype) = 0;
    // `store`, or with `atomic` an atomic sum, maximum or minimum.
    virtual void store(const TensorInfo& memory,
                       const std::vector<TensorInfo>& indices,
                       const TensorInfo& value,
                       const std::optional<TensorInfo>& mask,
                       std::optional<Reduction> atomic) = 0;
    virtual KernelProgram program() = 0;
};

// What a kernel target cannot write: the export reports it.
struct KernelError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// `@triton.jit` code, for generated PyTorch. Without `full_precision`, f32
// tile products may run in TF32.
std::unique_ptr<KernelTarget> make_triton_target(bool full_precision);
// A Pallas kernel (`pl.pallas_call`, its GPU lowering through Triton), for
// generated JAX.
std::unique_ptr<KernelTarget> make_pallas_target(bool full_precision);

} // namespace linnet::backend
