// Kernels as Triton: one `@triton.jit` function a kernel, its tiles Triton
// blocks, its memory pointers to contiguous tensors whose shapes the export
// knows, so that strides are constants.

#include "tile_target.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

class TritonTarget final : public TileTarget {
public:
    explicit TritonTarget(bool full_precision) : TileTarget("tl", full_precision) {}

    // ---------------------------------------------------------------- loops

    std::vector<std::string> begin_counted(std::int64_t start,
                                           std::int64_t stop,
                                           const std::vector<TensorInfo>& initial) override {
        const std::string prefix = "f" + std::to_string(loops_++) + "_";
        std::vector<std::string> names;
        names.reserve(initial.size() + 1);
        for (std::size_t i = 0; i < initial.size(); ++i) {
            names.push_back(prefix + std::to_string(i));
            // A carried value keeps one type: a literal becomes a tensor.
            const std::string value = literals_.contains(initial[i].name)
                                          ? cast(initial[i].name, initial[i].dtype)
                                          : initial[i].name;
            body_ += indent_ + names.back() + " = " + value + "\n";
        }
        body_ += indent_ + "for " + prefix + "n in range(" + std::to_string(start) + ", " +
                 std::to_string(stop) + "):\n";
        indent_ += "    ";
        open_.push_back(names);
        scopes_.emplace_back();
        names.insert(names.begin(), prefix + "n");
        return names;
    }

    std::vector<std::string> end_counted(const std::vector<TensorInfo>& next) override {
        std::vector<std::string> names = std::move(open_.back());
        open_.pop_back();
        for (std::size_t i = 0; i < next.size(); ++i) {
            body_ += indent_ + names[i] + " = " + next[i].name + "\n";
        }
        indent_.resize(indent_.size() - 4);
        scopes_.pop_back();
        return names;
    }

    // --------------------------------------------------------------- memory

    std::string
    memory(const std::string& name, const Dims& shape, ScalarKind dtype, bool is_result) override {
        (void)dtype;
        (void)is_result;
        const std::string parameter = unique(name);
        parameters_.push_back(parameter);
        shapes_[parameter] = shape;
        return parameter;
    }

    // A scalar crosses as a one-element tensor, read once.
    std::string scalar(const std::string& name, ScalarKind dtype) override {
        (void)dtype;
        const std::string value = unique(name);
        const std::string pointer = value + "_ptr";
        parameters_.push_back(pointer);
        body_ += indent_ + value + " = tl.load(" + pointer + ")\n";
        return value;
    }

    std::string program_id(std::int64_t axis) override {
        return define("tl.program_id(" + std::to_string(axis) + ")");
    }

    std::string load(const TensorInfo& memory,
                     const std::vector<TensorInfo>& indices,
                     const std::optional<TensorInfo>& mask,
                     const std::optional<TensorInfo>& other,
                     const Dims& shape,
                     ScalarKind dtype) override {
        (void)dtype;
        std::string call = "tl.load(" + pointer(memory, indices, shape.size());
        if (mask) {
            call += ", mask=" + mask->name + ", other=" + (other ? other->name : "0");
        }
        return define(call + ")");
    }

    void store(const TensorInfo& memory,
               const std::vector<TensorInfo>& indices,
               const TensorInfo& value,
               const std::optional<TensorInfo>& mask,
               std::optional<Reduction> atomic) override {
        std::size_t rank = 0;
        for (const TensorInfo& index : indices) {
            rank += index.shape.size();
        }
        std::string write = "tl.store(";
        if (atomic) {
            atomics_[memory.name] = *atomic;
            write = *atomic == Reduction::Sum   ? "tl.atomic_add("
                    : *atomic == Reduction::Max ? "tl.atomic_max("
                                                : "tl.atomic_min(";
        }
        body_ += indent_ + write + pointer(memory, indices, rank) + ", " + value.name +
                 (mask ? ", mask=" + mask->name : "") + ")\n";
    }

private:
    std::string dtype_name(ScalarKind dtype) const override {
        switch (dtype) {
        case ScalarKind::Bool:
            return "tl.int1";
        case ScalarKind::I8:
            return "tl.int8";
        case ScalarKind::I16:
            return "tl.int16";
        case ScalarKind::I32:
            return "tl.int32";
        case ScalarKind::I64:
            return "tl.int64";
        case ScalarKind::U8:
            return "tl.uint8";
        case ScalarKind::U16:
            return "tl.uint16";
        case ScalarKind::U32:
            return "tl.uint32";
        case ScalarKind::U64:
            return "tl.uint64";
        case ScalarKind::F16:
            return "tl.float16";
        case ScalarKind::BF16:
            return "tl.bfloat16";
        case ScalarKind::F32:
            return "tl.float32";
        case ScalarKind::F64:
            return "tl.float64";
        }
        return "tl.float32";
    }
    std::string cast(const std::string& value, ScalarKind dtype) const override {
        return "tl.cast(" + value + ", " + dtype_name(dtype) + ")";
    }
    std::string arange(std::int64_t length) const override {
        return "tl.arange(0, " + std::to_string(length) + ")";
    }
    std::string permute(const std::string& value, const Dims& permutation) const override {
        return "tl.permute(" + value + ", " + shape_list(permutation) + ")";
    }
    // Triton's integer `//` and `%` round toward zero, as Linnet's do.
    std::string
    divide_integers(const std::string& a, const std::string& b, ScalarKind dtype) const override {
        (void)dtype;
        return a + " // " + b;
    }
    std::string
    remainder(const std::string& a, const std::string& b, ScalarKind dtype) const override {
        (void)dtype;
        return a + " % " + b;
    }
    std::string rsqrt(const std::string& a) const override { return "tl.rsqrt(" + a + ")"; }
    std::string tanh(const std::string& a) const override {
        return "2.0 * tl.sigmoid(2.0 * " + a + ") - 1.0";
    }
    std::string
    dot(const std::string& a, const std::string& b, bool full_precision) const override {
        return "tl.dot(" + a + ", " + b + (full_precision ? ", input_precision=\"ieee\"" : "") +
               ")";
    }

    // Triton has no product reduction of its own: `tl.reduce` with the
    // generated module's `_multiply` (`TorchSourceTarget` writes it).
    std::string product(const std::string& value, std::int64_t axis) const override {
        return "tl.reduce(" + value + ", " + std::to_string(axis) + ", _multiply)";
    }

    // `memory + offset`: contiguous strides; index `k`'s tile takes its
    // axes' place among the result's `rank` (`rows[:, None] * K`). Offsets
    // are 64-bit where 32 bits cannot count the elements.
    std::string
    pointer(const TensorInfo& memory, const std::vector<TensorInfo>& indices, std::size_t rank) {
        const Dims& shape = shapes_.at(memory.name);
        std::int64_t elements = 1;
        for (const std::int64_t size : shape) {
            elements *= size;
        }
        const bool is_wide = elements > std::numeric_limits<std::int32_t>::max();
        std::string offset;
        std::size_t position = 0;
        for (std::size_t k = 0; k < indices.size(); ++k) {
            std::int64_t stride = 1;
            for (std::size_t later = k + 1; later < shape.size(); ++later) {
                stride *= shape[later];
            }
            std::string term = is_wide ? cast(indices[k].name, ScalarKind::I64) : indices[k].name;
            const std::size_t own = indices[k].shape.size();
            if (own > 0 && own < rank) {
                term += place(position, own, rank);
            }
            position += own;
            if (stride != 1) {
                term += " * " + std::to_string(stride);
            }
            offset += (offset.empty() ? "" : " + ") + term;
        }
        return offset.empty() ? memory.name : memory.name + " + " + offset;
    }

    std::map<std::string, Dims> shapes_;
    std::vector<std::vector<std::string>> open_;
};

} // namespace

std::unique_ptr<KernelTarget> make_triton_target(bool full_precision) {
    return std::make_unique<TritonTarget>(full_precision);
}

} // namespace linnet::backend
