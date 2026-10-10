#include "tile_target.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <numeric>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

bool is_power_of_two(std::int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
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

} // namespace

// ------------------------------------------------------------- not here

std::string TileTarget::input(const std::string& name, const Dims& shape, ScalarKind dtype) {
    (void)shape;
    (void)dtype;
    throw KernelError("internal: kernel input `" + name + "`");
}

std::string TileTarget::parameter(const std::string& path, const Dims& shape, ScalarKind dtype) {
    (void)shape;
    (void)dtype;
    throw KernelError("internal: kernel parameter `" + path + "`");
}

std::string TileTarget::state(const std::string& path, const Dims& shape, ScalarKind dtype) {
    (void)shape;
    (void)dtype;
    throw KernelError("internal: kernel state `" + path + "`");
}

std::string TileTarget::finish(const std::vector<TensorInfo>& results,
                               const std::vector<std::pair<std::string, TensorInfo>>& states,
                               const std::string& module_path,
                               const std::string& block_name,
                               const std::string& entry_name) {
    (void)results;
    (void)states;
    (void)module_path;
    (void)block_name;
    (void)entry_name;
    throw KernelError("internal: a kernel is not an entry");
}

// ---------------------------------------------------------------- tiles

// A constant is a Python literal; the library types it where it is used.
std::string TileTarget::constant(const Literal& literal, ScalarKind dtype) {
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

std::string TileTarget::elementwise(Elementwise kind,
                                    const std::vector<TensorInfo>& operands,
                                    const Dims& shape,
                                    ScalarKind dtype) {
    (void)shape;
    const std::string& a = operands[0].name;
    const std::string b = operands.size() > 1 ? operands[1].name : "";
    // Transcendental functions run in f32: half-precision tiles go through it.
    const bool is_half = dtype == ScalarKind::F16 || dtype == ScalarKind::BF16;
    const std::string wide = is_half ? cast(a, ScalarKind::F32) : a;
    const auto math = [&](const std::string& expression) {
        return is_half ? cast("(" + expression + ")", dtype) : expression;
    };
    const auto call = [&](const char* function) {
        return math(library_ + "." + function + "(" + wide + ")");
    };
    switch (kind) {
    case Elementwise::Add:
        return define(a + " + " + b);
    case Elementwise::Sub:
        return define(a + " - " + b);
    case Elementwise::Mul:
        return define(a + " * " + b);
    case Elementwise::Div:
        return define(sema::is_float(dtype) ? a + " / " + b : divide_integers(a, b, dtype));
    case Elementwise::Rem:
        return define(remainder(a, b, dtype));
    case Elementwise::Min:
        return define(library_ + ".minimum(" + a + ", " + b + ")");
    case Elementwise::Max:
        return define(library_ + ".maximum(" + a + ", " + b + ")");
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
        return define(library_ + ".where(" + a + ", False, True)");
    case Elementwise::Neg:
        return define("-" + a);
    case Elementwise::Exp:
        return define(call("exp"));
    case Elementwise::Log:
        return define(call("log"));
    case Elementwise::Sqrt:
        return define(call("sqrt"));
    case Elementwise::Rsqrt:
        return define(math(rsqrt(wide)));
    case Elementwise::Sin:
        return define(call("sin"));
    case Elementwise::Cos:
        return define(call("cos"));
    case Elementwise::Tanh:
        return define(math(tanh(wide)));
    case Elementwise::Abs:
        return define(library_ + ".abs(" + a + ")");
    }
    throw KernelError("internal: elementwise operation");
}

std::string TileTarget::compare(ir::CompareKind kind,
                                const TensorInfo& a,
                                const TensorInfo& b,
                                const Dims& shape) {
    (void)shape;
    static const std::map<ir::CompareKind, const char*> spelled{{ir::CompareKind::Eq, " == "},
                                                                {ir::CompareKind::Ne, " != "},
                                                                {ir::CompareKind::Lt, " < "},
                                                                {ir::CompareKind::Le, " <= "},
                                                                {ir::CompareKind::Gt, " > "},
                                                                {ir::CompareKind::Ge, " >= "}};
    return define(a.name + spelled.at(kind) + b.name);
}

std::string TileTarget::select(const TensorInfo& condition,
                               const TensorInfo& on_true,
                               const TensorInfo& on_false,
                               const Dims& shape,
                               ScalarKind dtype) {
    (void)shape;
    (void)dtype;
    return define(library_ + ".where(" + condition.name + ", " + on_true.name + ", " +
                  on_false.name + ")");
}

std::string TileTarget::convert(const TensorInfo& value, ScalarKind dtype) {
    return define(cast(value.name, dtype));
}

std::string TileTarget::reshape(const TensorInfo& value, const Dims& shape) {
    if (shape.empty()) {
        throw KernelError("a tile cannot become a scalar by reshaping; reduce it instead");
    }
    return define(library_ + ".reshape(" + value.name + ", " + shape_list(shape) + ")");
}

std::string
TileTarget::transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) {
    (void)shape;
    return define(permute(value.name, permutation));
}

// A scalar fills the tile; a tile gains axes (`x[:, None]`) and spreads
// over those of size one.
std::string TileTarget::broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) {
    if (value.shape.empty()) {
        if (literals_.contains(value.name)) {
            return define(library_ + ".full(" + shape_list(shape) + ", " + value.name + ", " +
                          dtype_name(value.dtype) + ")");
        }
        return define(library_ + ".zeros(" + shape_list(shape) + ", " + dtype_name(value.dtype) +
                      ") + " + value.name);
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
        sorted.reserve(permutation.size());
        sorted_shape.reserve(permutation.size());
        for (const std::int64_t axis : permutation) {
            sorted.push_back(order[static_cast<std::size_t>(axis)]);
            sorted_shape.push_back(value.shape[static_cast<std::size_t>(axis)]);
        }
        source = define(permute(source, permutation));
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
        source = define(library_ + ".broadcast_to(" + source + ", " + shape_list(shape) + ")");
    }
    return source;
}

// A product of two tiles over one shared axis, when every side is at least
// 16; the evaluator multiplies and sums the rest.
std::optional<std::string> TileTarget::contract(const TensorInfo& lhs,
                                                const Dims& lhs_axes,
                                                const TensorInfo& rhs,
                                                const Dims& rhs_axes,
                                                const Dims& out_axes,
                                                const Dims& shape,
                                                ScalarKind dtype) {
    (void)shape;
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
    const std::int64_t i = lhs_axes[0] == shared ? lhs_axes[1] : lhs_axes[0];
    const std::int64_t j = rhs_axes[0] == shared ? rhs_axes[1] : rhs_axes[0];
    if (contains(rhs_axes, i) || contains(lhs_axes, j)) {
        return std::nullopt;
    }
    const std::int64_t m = lhs.shape[lhs_axes[0] == i ? 0 : 1];
    const std::int64_t n = rhs.shape[rhs_axes[0] == j ? 0 : 1];
    const std::int64_t k = lhs.shape[lhs_axes[0] == shared ? 0 : 1];
    if (m < 16 || n < 16 || k < 16) {
        return std::nullopt;
    }
    std::string a = lhs.name;
    std::string b = rhs.name;
    if (lhs_axes[1] != shared) {
        a = define(permute(a, {1, 0}));
    }
    if (rhs_axes[0] != shared) {
        b = define(permute(b, {1, 0}));
    }
    std::string product = define(dot(a, b, full_precision_ && lhs.dtype == ScalarKind::F32));
    if (dtype != ScalarKind::F32) {
        product = define(cast(product, dtype));
    }
    if (out_axes[0] != i) {
        product = define(permute(product, {1, 0}));
    }
    return product;
}

std::string TileTarget::slice(const TensorInfo& value,
                              const Dims& starts,
                              const Dims& limits,
                              const Dims& strides,
                              const Dims& shape) {
    (void)value;
    (void)starts;
    (void)limits;
    (void)strides;
    (void)shape;
    throw KernelError("a tile cannot be sliced; `load` the part of memory instead");
}

std::string
TileTarget::concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) {
    (void)parts;
    (void)axis;
    (void)shape;
    throw KernelError("tiles cannot be joined with `concat`");
}

// A scan within a tile; integer sums cast back, as the libraries widen them.
std::string TileTarget::cumsum(const TensorInfo& value, std::int64_t axis) {
    const std::string sum =
        library_ + ".cumsum(" + value.name + ", axis=" + std::to_string(axis) + ")";
    return define(sema::is_float(value.dtype) ? sum : cast(sum, value.dtype));
}

std::string TileTarget::iota(std::int64_t length) {
    if (!is_power_of_two(length)) {
        throw KernelError("a tile's sizes are powers of two; `iota(" + std::to_string(length) +
                          ")` is not");
    }
    return define(arange(length));
}

std::string
TileTarget::gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) {
    (void)source;
    (void)indices;
    (void)shape;
    throw KernelError("a tile cannot be indexed by a tile; `load` from memory instead");
}

// Over the trailing axes, the last first.
std::string
TileTarget::reduce(Reduction kind, const TensorInfo& body, const Dims& dims, const Dims& shape) {
    (void)shape;
    if (kind == Reduction::Prod) {
        throw KernelError("a kernel cannot take a tile's `prod` yet");
    }
    const bool is_logical = kind == Reduction::Any || kind == Reduction::All;
    const std::string function =
        library_ + (kind == Reduction::Sum                             ? ".sum("
                    : kind == Reduction::Min || kind == Reduction::All ? ".min("
                                                                       : ".max(");
    std::string value = body.name;
    for (auto axis = dims.rbegin(); axis != dims.rend(); ++axis) {
        std::string expression = function;
        expression += is_logical ? cast(value, ScalarKind::I32) : value;
        expression += ", axis=";
        expression += std::to_string(*axis);
        expression += is_logical ? ") > 0" : ")";
        value = define(expression);
    }
    return value;
}

// ---------------------------------------------------------------- shared

std::string TileTarget::define(const std::string& expression) {
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

std::string TileTarget::unique(const std::string& name) {
    // Not one of the names the program itself uses.
    static const std::set<std::string> taken{
        "tl",  "jnp",    "jax",   "pl",     "plgpu",  "triton", "range",   "float",
        "int", "None",   "True",  "False",  "and",    "or",     "not",     "in",
        "is",  "def",    "class", "lambda", "pass",   "return", "for",     "while",
        "if",  "else",   "elif",  "import", "from",   "as",     "with",    "del",
        "try", "except", "raise", "yield",  "assert", "break",  "continue"};
    const bool is_generated = name.size() > 1 && (name[0] == 'v' || name[0] == 'f') &&
                              std::isdigit(static_cast<unsigned char>(name[1])) != 0;
    std::string base = name.empty() ? "arg" : name;
    if (taken.contains(base) || is_generated) {
        base += "_";
    }
    std::string candidate = base;
    for (int suffix = 1; used_.contains(candidate); ++suffix) {
        candidate = base + "_" + std::to_string(suffix);
    }
    used_.insert(candidate);
    return candidate;
}

std::string TileTarget::shape_list(const Dims& shape) {
    std::string text = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return text + (shape.size() == 1 ? ",)" : ")");
}

std::string TileTarget::place(std::size_t position, std::size_t own, std::size_t rank) {
    std::string index;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        index += axis == 0 ? "" : ", ";
        index += axis >= position && axis < position + own ? ":" : "None";
    }
    return "[" + index + "]";
}

} // namespace linnet::backend
