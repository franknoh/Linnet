// Kernels as Triton: one `@triton.jit` function a kernel, its tiles Triton
// blocks, its memory pointers to contiguous tensors whose shapes the export
// knows, so that strides are constants.

#include "linnet/backend/kernel.hpp"
#include "linnet/support/text.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

std::string tl_dtype(ScalarKind dtype) {
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

bool is_power_of_two(std::int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

std::string shape_list(const Dims& shape) {
    std::string text = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return text + (shape.size() == 1 ? ",)" : ")");
}

// The extremes of an integer dtype, the identities of `min` and `max`.
std::string integer_limit(ScalarKind dtype, bool highest) {
    const auto text = [](auto low, auto high, bool top) {
        return top ? std::to_string(high) : std::to_string(low);
    };
    switch (dtype) {
    case ScalarKind::I8:
        return text(std::numeric_limits<std::int8_t>::min(),
                    std::numeric_limits<std::int8_t>::max(),
                    highest);
    case ScalarKind::I16:
        return text(std::numeric_limits<std::int16_t>::min(),
                    std::numeric_limits<std::int16_t>::max(),
                    highest);
    case ScalarKind::I32:
        return text(std::numeric_limits<std::int32_t>::min(),
                    std::numeric_limits<std::int32_t>::max(),
                    highest);
    case ScalarKind::U8:
        return highest ? "255" : "0";
    case ScalarKind::U16:
        return highest ? "65535" : "0";
    case ScalarKind::U32:
        return highest ? "4294967295" : "0";
    case ScalarKind::U64:
        return highest ? "18446744073709551615" : "0";
    default:
        return text(std::numeric_limits<std::int64_t>::min(),
                    std::numeric_limits<std::int64_t>::max(),
                    highest);
    }
}

class TritonTarget final : public KernelTarget {
public:
    explicit TritonTarget(bool full_precision) : full_precision_(full_precision) {}

    // ------------------------------------------------------------ not here

    std::string input(const std::string& name, const Dims& shape, ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        throw KernelError("internal: kernel input `" + name + "`");
    }
    std::string parameter(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        throw KernelError("internal: kernel parameter `" + path + "`");
    }
    std::string state(const std::string& path, const Dims& shape, ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        throw KernelError("internal: kernel state `" + path + "`");
    }
    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override {
        (void)results;
        (void)states;
        (void)module_path;
        (void)block_name;
        (void)entry_name;
        throw KernelError("internal: a kernel is not an entry");
    }

    // --------------------------------------------------------------- tiles

    // A constant is a Python literal; Triton types it where it is used.
    std::string constant(const Literal& literal, ScalarKind dtype) override {
        std::string text;
        switch (literal.kind) {
        case Literal::Kind::Integer:
            text = sema::is_float(dtype) ? python_float(static_cast<double>(literal.integer))
                                         : std::to_string(literal.integer);
            break;
        case Literal::Kind::Real:
            text = python_float(literal.real);
            break;
        case Literal::Kind::Boolean:
            text = literal.integer != 0 ? "True" : "False";
            break;
        case Literal::Kind::Lowest:
        case Literal::Kind::Highest: {
            const bool highest = literal.kind == Literal::Kind::Highest;
            text = sema::is_float(dtype)       ? (highest ? "float(\"inf\")" : "-float(\"inf\")")
                   : dtype == ScalarKind::Bool ? (highest ? "True" : "False")
                                               : integer_limit(dtype, highest);
            break;
        }
        }
        if (text.starts_with('-')) {
            text = "(" + text + ")";
        }
        literals_.insert(text);
        return text;
    }

    std::string elementwise(Elementwise kind,
                            const std::vector<TensorInfo>& operands,
                            const Dims& shape,
                            ScalarKind dtype) override {
        (void)shape;
        const std::string& a = operands[0].name;
        const std::string b = operands.size() > 1 ? operands[1].name : "";
        // Triton's transcendental functions take f32: half-precision tiles
        // go through it.
        const bool is_half = dtype == ScalarKind::F16 || dtype == ScalarKind::BF16;
        const auto math = [&](const std::string& expression) {
            return is_half ? "(" + expression + ").to(" + tl_dtype(dtype) + ")" : expression;
        };
        const std::string wide = is_half ? a + ".to(tl.float32)" : a;
        switch (kind) {
        case Elementwise::Add:
            return define(a + " + " + b);
        case Elementwise::Sub:
            return define(a + " - " + b);
        case Elementwise::Mul:
            return define(a + " * " + b);
        case Elementwise::Div:
            // Integer division rounds toward zero in Triton, as in Linnet.
            return define(a + (sema::is_float(dtype) ? " / " : " // ") + b);
        case Elementwise::Rem:
            return define(a + " % " + b);
        case Elementwise::Min:
            return define("tl.minimum(" + a + ", " + b + ")");
        case Elementwise::Max:
            return define("tl.maximum(" + a + ", " + b + ")");
        case Elementwise::And:
        case Elementwise::BitAnd:
            return define(a + " & " + b);
        case Elementwise::Or:
        case Elementwise::BitOr:
            return define(a + " | " + b);
        case Elementwise::BitXor:
            return define(a + " ^ " + b);
        case Elementwise::Shl:
            return define(a + " << " + b);
        case Elementwise::Shr:
            return define(a + " >> " + b);
        case Elementwise::Not:
            return define("tl.where(" + a + ", False, True)");
        case Elementwise::Neg:
            return define("-" + a);
        case Elementwise::Exp:
            return define(math("tl.exp(" + wide + ")"));
        case Elementwise::Log:
            return define(math("tl.log(" + wide + ")"));
        case Elementwise::Sqrt:
            return define(math("tl.sqrt(" + wide + ")"));
        case Elementwise::Rsqrt:
            return define(math("tl.rsqrt(" + wide + ")"));
        case Elementwise::Sin:
            return define(math("tl.sin(" + wide + ")"));
        case Elementwise::Cos:
            return define(math("tl.cos(" + wide + ")"));
        case Elementwise::Tanh:
            return define(math("2.0 * tl.sigmoid(2.0 * " + wide + ") - 1.0"));
        case Elementwise::Abs:
            return define("tl.abs(" + a + ")");
        }
        throw KernelError("internal: elementwise operation");
    }

    std::string compare(ir::CompareKind kind,
                        const TensorInfo& a,
                        const TensorInfo& b,
                        const Dims& shape) override {
        (void)shape;
        static const std::map<ir::CompareKind, const char*> spelled{{ir::CompareKind::Eq, " == "},
                                                                    {ir::CompareKind::Ne, " != "},
                                                                    {ir::CompareKind::Lt, " < "},
                                                                    {ir::CompareKind::Le, " <= "},
                                                                    {ir::CompareKind::Gt, " > "},
                                                                    {ir::CompareKind::Ge, " >= "}};
        return define(a.name + spelled.at(kind) + b.name);
    }

    std::string select(const TensorInfo& condition,
                       const TensorInfo& on_true,
                       const TensorInfo& on_false,
                       const Dims& shape,
                       ScalarKind dtype) override {
        (void)shape;
        (void)dtype;
        return define("tl.where(" + condition.name + ", " + on_true.name + ", " + on_false.name +
                      ")");
    }

    std::string convert(const TensorInfo& value, ScalarKind dtype) override {
        return define("tl.cast(" + value.name + ", " + tl_dtype(dtype) + ")");
    }

    std::string reshape(const TensorInfo& value, const Dims& shape) override {
        if (shape.empty()) {
            throw KernelError("a tile cannot become a scalar by reshaping; reduce it instead");
        }
        return define("tl.reshape(" + value.name + ", " + shape_list(shape) + ")");
    }

    std::string
    transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) override {
        (void)shape;
        return define("tl.permute(" + value.name + ", " + shape_list(permutation) + ")");
    }

    // A scalar fills the tile; a tile gains axes (`x[:, None]`) and spreads
    // over those of one.
    std::string broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) override {
        if (value.shape.empty()) {
            if (literals_.contains(value.name)) {
                return define("tl.full(" + shape_list(shape) + ", " + value.name + ", " +
                              tl_dtype(value.dtype) + ")");
            }
            return define("tl.zeros(" + shape_list(shape) + ", " + tl_dtype(value.dtype) + ") + " +
                          value.name);
        }
        std::string source = value.name;
        Dims order = dims;
        Dims placed_shape = value.shape;
        if (!std::is_sorted(order.begin(), order.end())) {
            // Into the result's axis order first.
            Dims permutation(order.size());
            std::iota(permutation.begin(), permutation.end(), std::int64_t{0});
            std::sort(permutation.begin(), permutation.end(), [&](std::int64_t x, std::int64_t y) {
                return order[static_cast<std::size_t>(x)] < order[static_cast<std::size_t>(y)];
            });
            Dims sorted;
            Dims sorted_shape;
            for (const std::int64_t axis : permutation) {
                sorted.push_back(order[static_cast<std::size_t>(axis)]);
                sorted_shape.push_back(value.shape[static_cast<std::size_t>(axis)]);
            }
            source = define("tl.permute(" + source + ", " + shape_list(permutation) + ")");
            order = sorted;
            placed_shape = sorted_shape;
        }
        Dims expanded(shape.size(), 1);
        std::string index;
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            const auto at = std::find(order.begin(), order.end(), static_cast<std::int64_t>(axis));
            index += axis == 0 ? "" : ", ";
            if (at == order.end()) {
                index += "None";
            } else {
                index += ":";
                expanded[axis] = placed_shape[static_cast<std::size_t>(at - order.begin())];
            }
        }
        if (order.size() != shape.size()) {
            source = define(source + "[" + index + "]");
        }
        if (expanded != shape) {
            source = define("tl.broadcast_to(" + source + ", " + shape_list(shape) + ")");
        }
        return source;
    }

    bool broadcasts_elementwise() const override { return true; }

    // A product of two tiles over one shared axis: `tl.dot` when every
    // side is at least 16, its minimum; the evaluator multiplies and sums
    // the rest.
    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        ScalarKind dtype) override {
        if (lhs_axes.size() != 2 || rhs_axes.size() != 2 || out_axes.size() != 2) {
            return std::nullopt;
        }
        const auto contains = [](const Dims& axes, std::int64_t axis) {
            return std::find(axes.begin(), axes.end(), axis) != axes.end();
        };
        std::int64_t shared = -1;
        for (const std::int64_t axis : lhs_axes) {
            if (contains(rhs_axes, axis) && !contains(out_axes, axis)) {
                shared = axis;
            }
        }
        if (shared < 0) {
            return std::nullopt;
        }
        // `a[i, k] @ b[k, j]`, transposing what is the other way around.
        std::string a = lhs.name;
        std::string b = rhs.name;
        const std::int64_t i = lhs_axes[0] == shared ? lhs_axes[1] : lhs_axes[0];
        const std::int64_t j = rhs_axes[0] == shared ? rhs_axes[1] : rhs_axes[0];
        if (contains(rhs_axes, i) || contains(lhs_axes, j)) {
            return std::nullopt;
        }
        if (lhs_axes[1] != shared) {
            a = define("tl.trans(" + a + ")");
        }
        if (rhs_axes[0] != shared) {
            b = define("tl.trans(" + b + ")");
        }
        const std::int64_t m = lhs.shape[lhs_axes[0] == i ? 0 : 1];
        const std::int64_t n = rhs.shape[rhs_axes[0] == j ? 0 : 1];
        const std::int64_t k = lhs.shape[lhs_axes[0] == shared ? 0 : 1];
        if (m < 16 || n < 16 || k < 16) {
            return std::nullopt;
        }
        const std::string precision =
            lhs.dtype == ScalarKind::F32 && full_precision_ ? ", input_precision=\"ieee\"" : "";
        std::string product = define("tl.dot(" + a + ", " + b + precision + ")");
        if (dtype != ScalarKind::F32) {
            product = define("tl.cast(" + product + ", " + tl_dtype(dtype) + ")");
        }
        if (out_axes[0] != i) {
            product = define("tl.trans(" + product + ")");
        }
        (void)shape;
        return product;
    }

    std::string slice(const TensorInfo& value,
                      const Dims& starts,
                      const Dims& limits,
                      const Dims& strides,
                      const Dims& shape) override {
        (void)value;
        (void)starts;
        (void)limits;
        (void)strides;
        (void)shape;
        throw KernelError("a Triton tile cannot be sliced; `load` the part of memory instead");
    }

    std::string
    concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) override {
        (void)parts;
        (void)axis;
        (void)shape;
        throw KernelError("Triton tiles cannot be joined with `concat`");
    }

    std::string iota(std::int64_t length) override {
        if (!is_power_of_two(length)) {
            throw KernelError("a Triton tile's sizes are powers of two; `iota(" +
                              std::to_string(length) + ")` is not");
        }
        return define("tl.arange(0, " + std::to_string(length) + ")");
    }

    std::string
    gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) override {
        (void)source;
        (void)indices;
        (void)shape;
        throw KernelError("a Triton tile cannot be indexed by a tile; `load` from memory instead");
    }

    // Over the trailing axes, the last first.
    std::string
    reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) override {
        (void)shape;
        if (kind == Reduction::Prod) {
            throw KernelError("a Triton kernel cannot take a tile's `prod` yet");
        }
        const bool is_logical = kind == Reduction::Any || kind == Reduction::All;
        const std::string function = kind == Reduction::Sum                             ? "tl.sum("
                                     : kind == Reduction::Min || kind == Reduction::All ? "tl.min("
                                                                                        : "tl.max(";
        std::string value = body.name;
        for (auto axis = dims.rbegin(); axis != dims.rend(); ++axis) {
            std::string expression = function;
            expression += is_logical ? "tl.cast(" + value + ", tl.int32)" : value;
            expression += ", axis=";
            expression += std::to_string(*axis);
            expression += is_logical ? ") > 0" : ")";
            value = define(expression);
        }
        return value;
    }

    void release(const std::vector<std::string>& names) override { (void)names; }

    // ---------------------------------------------------------------- loops

    bool supports_while() const override { return false; }
    bool supports_counted() const override { return true; }

    std::vector<std::string> begin_counted(std::int64_t start,
                                           std::int64_t stop,
                                           const std::vector<TensorInfo>& initial) override {
        const std::string prefix = "f" + std::to_string(loops_++) + "_";
        std::vector<std::string> names;
        for (std::size_t i = 0; i < initial.size(); ++i) {
            names.push_back(prefix + std::to_string(i));
            // A carried value keeps one type: a literal becomes a tensor.
            const std::string value =
                literals_.contains(initial[i].name)
                    ? "tl.cast(" + initial[i].name + ", " + tl_dtype(initial[i].dtype) + ")"
                    : initial[i].name;
            body_ += indent_ + names.back() + " = " + value + "\n";
        }
        body_ += indent_ + "for " + prefix + "n in range(" + std::to_string(start) + ", " +
                 std::to_string(stop) + "):\n";
        indent_ += "    ";
        loops_open_.push_back(names);
        scopes_.emplace_back();
        names.insert(names.begin(), prefix + "n");
        return names;
    }

    std::vector<std::string> end_counted(const std::vector<TensorInfo>& next) override {
        std::vector<std::string> names = std::move(loops_open_.back());
        loops_open_.pop_back();
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
               const std::optional<TensorInfo>& mask) override {
        std::size_t rank = 0;
        for (const TensorInfo& index : indices) {
            rank += index.shape.size();
        }
        body_ += indent_ + "tl.store(" + pointer(memory, indices, rank) + ", " + value.name +
                 (mask ? ", mask=" + mask->name : "") + ")\n";
    }

    KernelProgram program() override { return {parameters_, body_}; }

private:
    std::string define(const std::string& expression) {
        for (auto scope = scopes_.rbegin(); scope != scopes_.rend(); ++scope) {
            const auto found = scope->find(expression);
            if (found != scope->end()) {
                return found->second;
            }
        }
        const std::string name = "v" + std::to_string(next_++);
        body_ += indent_ + name + " = " + expression + "\n";
        scopes_.back()[expression] = name;
        return name;
    }

    std::string unique(const std::string& name) {
        std::string candidate = name.empty() ? "arg" : name;
        for (int suffix = 1; used_.contains(candidate); ++suffix) {
            candidate = name + "_" + std::to_string(suffix);
        }
        used_.insert(candidate);
        return candidate;
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
            std::string term =
                is_wide ? "tl.cast(" + indices[k].name + ", tl.int64)" : indices[k].name;
            const std::size_t own = indices[k].shape.size();
            if (own > 0 && own < rank) {
                std::string index;
                for (std::size_t axis = 0; axis < rank; ++axis) {
                    index += axis == 0 ? "" : ", ";
                    index += axis >= position && axis < position + own ? ":" : "None";
                }
                term += "[" + index + "]";
            }
            position += own;
            if (stride != 1) {
                term += " * " + std::to_string(stride);
            }
            offset += (offset.empty() ? "" : " + ") + term;
        }
        return offset.empty() ? memory.name : memory.name + " + " + offset;
    }

    bool full_precision_ = false;
    std::vector<std::string> parameters_;
    std::map<std::string, Dims> shapes_;
    std::set<std::string> used_;
    std::set<std::string> literals_;
    std::vector<std::map<std::string, std::string>> scopes_{1};
    std::vector<std::vector<std::string>> loops_open_;
    std::string body_;
    std::string indent_ = "    ";
    std::size_t next_ = 0;
    std::size_t loops_ = 0;
};

} // namespace

std::unique_ptr<KernelTarget> make_triton_target(bool full_precision) {
    return std::make_unique<TritonTarget>(full_precision);
}

} // namespace linnet::backend
