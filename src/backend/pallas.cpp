// Kernels as Pallas: one kernel function of refs a kernel, its tiles
// `jax.numpy` arrays, memory read and written through `plgpu.load` and
// `plgpu.store` at index arrays, which Pallas lowers to Triton on a GPU.

#include "linnet/support/text.hpp"

#include "tile_target.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

class PallasTarget final : public TileTarget {
public:
    explicit PallasTarget(bool full_precision) : TileTarget("jnp", full_precision) {}

    // ---------------------------------------------------------------- loops

    // `jax.lax.fori_loop` over a body function of the carried values: a
    // Python loop would unroll as the kernel is traced.
    std::vector<std::string> begin_counted(std::int64_t start,
                                           std::int64_t stop,
                                           const std::vector<TensorInfo>& initial) override {
        const std::string prefix = "f" + std::to_string(loops_++) + "_";
        Loop loop{prefix, start, stop, {}, {}};
        for (std::size_t i = 0; i < initial.size(); ++i) {
            loop.names.push_back(prefix + std::to_string(i));
            loop.initial.push_back(literals_.contains(initial[i].name)
                                       ? cast(initial[i].name, initial[i].dtype)
                                       : initial[i].name);
        }
        body_ += indent_ + "def " + prefix + "body(" + prefix + "n, " + prefix + "carry):\n";
        indent_ += "    ";
        if (!loop.names.empty()) {
            body_ += indent_ + "(" + join(loop.names, ", ") + ",) = " + prefix + "carry\n";
        }
        scopes_.emplace_back();
        std::vector<std::string> names{prefix + "n"};
        names.insert(names.end(), loop.names.begin(), loop.names.end());
        open_.push_back(std::move(loop));
        return names;
    }

    std::vector<std::string> end_counted(const std::vector<TensorInfo>& next) override {
        const Loop loop = std::move(open_.back());
        open_.pop_back();
        std::vector<std::string> values;
        values.reserve(next.size());
        for (const TensorInfo& value : next) {
            values.push_back(value.name);
        }
        body_ += indent_ + "return (" + join(values, ", ") + (values.empty() ? ")\n" : ",)\n");
        indent_.resize(indent_.size() - 4);
        scopes_.pop_back();
        const std::string carried =
            loop.initial.empty() ? "()" : "(" + join(loop.initial, ", ") + ",)";
        body_ += indent_ + (loop.names.empty() ? "" : "(" + join(loop.names, ", ") + ",) = ") +
                 "jax.lax.fori_loop(" + std::to_string(loop.start) + ", " +
                 std::to_string(loop.stop) + ", " + loop.prefix + "body, " + carried + ")\n";
        return loop.names;
    }

    // --------------------------------------------------------------- memory

    std::string
    memory(const std::string& name, const Dims& shape, ScalarKind dtype, bool is_result) override {
        (void)shape;
        (void)dtype;
        (void)is_result;
        const std::string ref = unique(name);
        parameters_.push_back(ref);
        return ref;
    }

    // A scalar crosses as a one-element array, read once.
    std::string scalar(const std::string& name, ScalarKind dtype) override {
        (void)dtype;
        const std::string value = unique(name);
        const std::string ref = value + "_ref";
        parameters_.push_back(ref);
        body_ += indent_ + value + " = " + ref + "[0]\n";
        return value;
    }

    std::string program_id(std::int64_t axis) override {
        return define("pl.program_id(" + std::to_string(axis) + ")");
    }

    std::string load(const TensorInfo& memory,
                     const std::vector<TensorInfo>& indices,
                     const std::optional<TensorInfo>& mask,
                     const std::optional<TensorInfo>& other,
                     const Dims& shape,
                     ScalarKind dtype) override {
        (void)dtype;
        std::string call = "plgpu.load(" + view(memory, indices, shape.size());
        if (mask) {
            call += ", mask=" + mask->name + ", other=" + (other ? other->name : "0");
        }
        return define(call + ")");
    }

    void store(const TensorInfo& memory,
               const std::vector<TensorInfo>& indices,
               const TensorInfo& value,
               const std::optional<TensorInfo>& mask) override {
        Dims shape;
        for (const TensorInfo& index : indices) {
            shape.insert(shape.end(), index.shape.begin(), index.shape.end());
        }
        // Pallas stores a value of the indexed shape.
        const std::string stored =
            value.shape == shape
                ? value.name
                : define("jnp.broadcast_to(" + value.name + ", " + shape_list(shape) + ")");
        body_ += indent_ + "plgpu.store(" + view(memory, indices, shape.size()) + ", " + stored +
                 (mask ? ", mask=" + mask->name : "") + ")\n";
    }

private:
    struct Loop {
        std::string prefix;
        std::int64_t start = 0;
        std::int64_t stop = 0;
        std::vector<std::string> names;   // the carried values, in the body and after
        std::vector<std::string> initial; // their values before the loop
    };

    std::string dtype_name(ScalarKind dtype) const override {
        switch (dtype) {
        case ScalarKind::Bool:
            return "jnp.bool_";
        case ScalarKind::I8:
            return "jnp.int8";
        case ScalarKind::I16:
            return "jnp.int16";
        case ScalarKind::I32:
            return "jnp.int32";
        case ScalarKind::I64:
            return "jnp.int64";
        case ScalarKind::U8:
            return "jnp.uint8";
        case ScalarKind::U16:
            return "jnp.uint16";
        case ScalarKind::U32:
            return "jnp.uint32";
        case ScalarKind::U64:
            return "jnp.uint64";
        case ScalarKind::F16:
            return "jnp.float16";
        case ScalarKind::BF16:
            return "jnp.bfloat16";
        case ScalarKind::F32:
            return "jnp.float32";
        case ScalarKind::F64:
            return "jnp.float64";
        }
        return "jnp.float32";
    }
    std::string cast(const std::string& value, ScalarKind dtype) const override {
        return "jax.lax.convert_element_type(" + value + ", " + dtype_name(dtype) + ")";
    }
    std::string arange(std::int64_t length) const override {
        return "jnp.arange(" + std::to_string(length) + ", dtype=jnp.int32)";
    }
    std::string permute(const std::string& value, const Dims& permutation) const override {
        return "jnp.transpose(" + value + ", " + shape_list(permutation) + ")";
    }
    // `jax.lax.div` and `rem` round toward zero, as Linnet does; they take
    // operands of one dtype.
    std::string
    divide_integers(const std::string& a, const std::string& b, ScalarKind dtype) const override {
        return "jax.lax.div(" + cast(a, dtype) + ", " + cast(b, dtype) + ")";
    }
    std::string
    remainder(const std::string& a, const std::string& b, ScalarKind dtype) const override {
        return "jax.lax.rem(" + cast(a, dtype) + ", " + cast(b, dtype) + ")";
    }
    std::string rsqrt(const std::string& a) const override { return "jax.lax.rsqrt(" + a + ")"; }
    std::string tanh(const std::string& a) const override { return "jnp.tanh(" + a + ")"; }
    std::string
    dot(const std::string& a, const std::string& b, bool full_precision) const override {
        return "jnp.dot(" + a + ", " + b + ", preferred_element_type=jnp.float32" +
               (full_precision ? ", precision=jax.lax.Precision.HIGHEST" : "") + ")";
    }

    // `x.at[rows[:, None], cols[None, :]]`: index arrays broadcast against
    // one another, each tile in its axes' place.
    std::string
    view(const TensorInfo& memory, const std::vector<TensorInfo>& indices, std::size_t rank) const {
        std::string text = memory.name + ".at[";
        std::size_t position = 0;
        for (std::size_t k = 0; k < indices.size(); ++k) {
            const std::size_t own = indices[k].shape.size();
            text += (k == 0 ? "" : ", ") + indices[k].name;
            if (own > 0 && own < rank) {
                text += place(position, own, rank);
            }
            position += own;
        }
        return text + "]";
    }

    std::vector<Loop> open_;
};

} // namespace

std::unique_ptr<KernelTarget> make_pallas_target(bool full_precision) {
    return std::make_unique<PallasTarget>(full_precision);
}

} // namespace linnet::backend
