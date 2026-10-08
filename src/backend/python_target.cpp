#include "linnet/backend/python_target.hpp"

#include "linnet/sema/types.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string_view>

namespace linnet::backend {

using sema::ScalarKind;

PythonTarget::PythonTarget(std::string library, std::string bool_name, bool prepare, Lora lora)
    : library_(std::move(library)), bool_name_(std::move(bool_name)), prepare_(prepare),
      lora_(std::move(lora)) {}

std::string PythonTarget::input(const std::string& name, const Dims& shape, ScalarKind dtype) {
    (void)shape;
    (void)dtype;
    std::string clean;
    for (const char c : name) {
        clean += std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' ? c : '_';
    }
    const std::string argument = "in_" + clean;
    arguments_.push_back(argument);
    return argument;
}

std::string PythonTarget::parameter(const std::string& path, const Dims& shape, ScalarKind dtype) {
    const std::string argument = "p" + std::to_string(parameters_.size());
    parameters_.push_back(path);
    parameter_types_.emplace_back(shape, dtype);
    arguments_.push_back(argument);
    parameters_end_ = arguments_.size();
    return argument;
}

std::string PythonTarget::state(const std::string& path, const Dims& shape, ScalarKind dtype) {
    (void)shape;
    (void)dtype;
    const std::string argument = "s" + std::to_string(states_.size());
    states_.push_back(path);
    arguments_.push_back(argument);
    return argument;
}

std::string PythonTarget::constant(const Literal& literal, ScalarKind dtype) {
    const std::string text = literal_text(literal, dtype);
    const std::string name = define(constant_expression(text, dtype));
    literals_[name] = text;
    if (literal.kind == Literal::Kind::Integer || literal.kind == Literal::Kind::Real) {
        values_[name] = literal.kind == Literal::Kind::Real ? literal.real
                                                            : static_cast<double>(literal.integer);
    }
    return name;
}

std::string PythonTarget::elementwise(Elementwise kind,
                                      const std::vector<TensorInfo>& operands,
                                      const Dims& shape,
                                      ScalarKind dtype) {
    (void)shape;
    const std::string name = spell(kind, operands);
    fold(name, kind, operands, dtype);
    return name;
}

std::string PythonTarget::select(const TensorInfo& condition,
                                 const TensorInfo& on_true,
                                 const TensorInfo& on_false,
                                 const Dims& shape,
                                 ScalarKind dtype) {
    (void)shape;
    (void)dtype;
    return define(library_ + ".where(" + condition.name + ", " + on_true.name + ", " +
                  on_false.name + ")");
}

std::string PythonTarget::reshape(const TensorInfo& value, const Dims& shape) {
    return define(value.name + ".reshape(" + python_tuple(shape) + ")");
}

std::string PythonTarget::broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) {
    // Axes first in the order they take in the result, then size-one axes
    // inserted, then the expansion.
    Dims order(dims.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<std::int64_t>(i);
    }
    std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        return dims[static_cast<std::size_t>(a)] < dims[static_cast<std::size_t>(b)];
    });
    TensorInfo source = value;
    Dims sorted_dims = dims;
    bool is_identity = true;
    for (std::size_t i = 0; i < order.size(); ++i) {
        is_identity = is_identity && order[i] == static_cast<std::int64_t>(i);
    }
    if (!is_identity) {
        Dims permuted;
        for (const std::int64_t axis : order) {
            permuted.push_back(value.shape[static_cast<std::size_t>(axis)]);
            sorted_dims[permuted.size() - 1] = dims[static_cast<std::size_t>(axis)];
        }
        source = {transpose(value, order, permuted), permuted, value.dtype};
    }
    Dims placed(shape.size(), 1);
    for (std::size_t i = 0; i < sorted_dims.size(); ++i) {
        placed[static_cast<std::size_t>(sorted_dims[i])] = source.shape[i];
    }
    if (placed != source.shape) {
        source = {reshape(source, placed), placed, source.dtype};
    }
    if (placed == shape) {
        return source.name;
    }
    return define(expand(source.name, shape));
}

std::string PythonTarget::slice(const TensorInfo& value,
                                const Dims& starts,
                                const Dims& limits,
                                const Dims& strides,
                                const Dims& shape) {
    (void)shape;
    std::string index;
    for (std::size_t i = 0; i < starts.size(); ++i) {
        index += (i == 0 ? "" : ", ") + std::to_string(starts[i]) + ":" +
                 std::to_string(limits[i]) + ":" + std::to_string(strides[i]);
    }
    return define(value.name + "[" + index + "]");
}

std::string
PythonTarget::gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) {
    (void)shape;
    // `indices[..., k]` selects along axis k of the source: advanced indexing
    // with one index tensor per leading source axis, the trailing ones whole.
    std::string index;
    for (std::size_t k = 0; k < static_cast<std::size_t>(indices.shape.back()); ++k) {
        index += (k == 0 ? "" : ", ") + indices.name + "[..., " + std::to_string(k) + "]";
    }
    return define(source.name + "[" + index + "]");
}

std::optional<std::string> PythonTarget::contract(const TensorInfo& lhs,
                                                  const Dims& lhs_axes,
                                                  const TensorInfo& rhs,
                                                  const Dims& rhs_axes,
                                                  const Dims& out_axes,
                                                  const Dims& shape,
                                                  ScalarKind dtype) {
    (void)shape, (void)dtype;
    return define(library_ + ".einsum(\"" + einsum_equation(lhs_axes, rhs_axes, out_axes) + "\", " +
                  lhs.name + ", " + rhs.name + ")");
}

std::string PythonTarget::dtype_name(ScalarKind dtype) const {
    switch (dtype) {
    case ScalarKind::Bool:
        return library_ + "." + bool_name_;
    case ScalarKind::I8:
        return library_ + ".int8";
    case ScalarKind::I16:
        return library_ + ".int16";
    case ScalarKind::I32:
        return library_ + ".int32";
    case ScalarKind::I64:
        return library_ + ".int64";
    case ScalarKind::U8:
        return library_ + ".uint8";
    case ScalarKind::U16:
        return library_ + ".uint16";
    case ScalarKind::U32:
        return library_ + ".uint32";
    case ScalarKind::U64:
        return library_ + ".uint64";
    case ScalarKind::F16:
        return library_ + ".float16";
    case ScalarKind::BF16:
        return library_ + ".bfloat16";
    case ScalarKind::F32:
        return library_ + ".float32";
    case ScalarKind::F64:
        return library_ + ".float64";
    }
    return library_ + ".float32";
}

std::string PythonTarget::define(const std::string& expression) {
    for (auto scope = cse_.rbegin(); scope != cse_.rend(); ++scope) {
        const auto found = scope->find(expression);
        if (found != scope->end()) {
            return found->second;
        }
    }
    const std::string name = "v" + std::to_string(next_++);
    body_ += indent_ + name + " = " + expression + "\n";
    cse_.back()[expression] = name;
    return name;
}

std::string PythonTarget::spell(Elementwise kind, const std::vector<TensorInfo>& operands) {
    const std::string& a = operands[0].name;
    const std::string b = operands.size() > 1 ? operands[1].name : "";
    const auto call = [&](const char* function, bool binary) {
        return define(library_ + "." + function + "(" + a + (binary ? ", " + b : "") + ")");
    };
    switch (kind) {
    case Elementwise::Add:
        return define(a + " + " + b);
    case Elementwise::Sub:
        return define(a + " - " + b);
    case Elementwise::Mul:
        return define(a + " * " + b);
    case Elementwise::Div:
        return define(sema::is_float(operands[0].dtype) ? a + " / " + b : divide_integers(a, b));
    case Elementwise::Rem:
        return call("fmod", true);
    case Elementwise::Min:
        return call("minimum", true);
    case Elementwise::Max:
        return call("maximum", true);
    case Elementwise::And:
    case Elementwise::BitAnd:
        return define(a + " & " + b);
    case Elementwise::Or:
    case Elementwise::BitOr:
        return define(a + " | " + b);
    case Elementwise::BitXor:
        return define(a + " ^ " + b);
    case Elementwise::Shl:
        return define(shift(a, b, true));
    case Elementwise::Shr:
        return define(shift(a, b, false));
    case Elementwise::Not:
        return define("~" + a);
    case Elementwise::Neg:
        return define("-" + a);
    case Elementwise::Exp:
        return call("exp", false);
    case Elementwise::Log:
        return call("log", false);
    case Elementwise::Sqrt:
        return call("sqrt", false);
    case Elementwise::Rsqrt:
        return define(reciprocal_sqrt(a));
    case Elementwise::Sin:
        return call("sin", false);
    case Elementwise::Cos:
        return call("cos", false);
    case Elementwise::Tanh:
        return call("tanh", false);
    case Elementwise::Abs:
        return call("abs", false);
    }
    return define(a);
}

void PythonTarget::fold(const std::string& name,
                        Elementwise kind,
                        const std::vector<TensorInfo>& operands,
                        ScalarKind dtype) {
    if (dtype == ScalarKind::Bool || operands.empty() || !operands[0].shape.empty()) {
        return;
    }
    std::vector<double> values;
    for (const TensorInfo& operand : operands) {
        const auto found = values_.find(operand.name);
        if (found == values_.end()) {
            return;
        }
        values.push_back(found->second);
    }
    double result = 0.0;
    const double a = values[0];
    const double b = values.size() > 1 ? values[1] : 0.0;
    switch (kind) {
    case Elementwise::Add:
        result = a + b;
        break;
    case Elementwise::Sub:
        result = a - b;
        break;
    case Elementwise::Mul:
        result = a * b;
        break;
    case Elementwise::Div:
        if (b == 0.0) {
            return;
        }
        result = sema::is_float(dtype) ? a / b : std::trunc(a / b);
        break;
    case Elementwise::Neg:
        result = -a;
        break;
    case Elementwise::Sqrt:
        result = std::sqrt(a);
        break;
    case Elementwise::Rsqrt:
        result = 1.0 / std::sqrt(a);
        break;
    case Elementwise::Exp:
        result = std::exp(a);
        break;
    case Elementwise::Log:
        result = std::log(a);
        break;
    default:
        return;
    }
    if (!sema::is_float(dtype)) {
        if (kind != Elementwise::Add && kind != Elementwise::Sub && kind != Elementwise::Mul &&
            kind != Elementwise::Div && kind != Elementwise::Neg) {
            return;
        }
        values_[name] = result;
        literals_[name] = std::to_string(static_cast<long long>(result));
        return;
    }
    if (dtype == ScalarKind::F32) {
        result = static_cast<double>(static_cast<float>(result));
    }
    values_[name] = result;
    literals_[name] = python_float(result);
}

std::string PythonTarget::literal_text(const Literal& literal, ScalarKind dtype) const {
    switch (literal.kind) {
    case Literal::Kind::Integer:
        return sema::is_float(dtype) ? python_float(static_cast<double>(literal.integer))
                                     : std::to_string(literal.integer);
    case Literal::Kind::Real:
        return python_float(literal.real);
    case Literal::Kind::Boolean:
        return literal.integer != 0 ? "True" : "False";
    case Literal::Kind::Lowest:
        if (sema::is_float(dtype)) {
            return "-" + infinity();
        }
        return dtype == ScalarKind::Bool ? "False"
                                         : library_ + ".iinfo(" + dtype_name(dtype) + ").min";
    case Literal::Kind::Highest:
        if (sema::is_float(dtype)) {
            return infinity();
        }
        return dtype == ScalarKind::Bool ? "True"
                                         : library_ + ".iinfo(" + dtype_name(dtype) + ").max";
    }
    return "0";
}

std::optional<std::string> PythonTarget::lora_target(const std::string& name) const {
    // A sharded weight is used as the value gathered from its parameter.
    const auto gathered = gathered_.find(name);
    const std::string& argument = gathered != gathered_.end() ? gathered->second : name;
    if (lora_.patterns.empty() || !numbered(argument, 'p')) {
        return std::nullopt;
    }
    const std::size_t index = std::stoul(argument.substr(1));
    if (index >= parameters_.size() || !parameters_[index].ends_with(".weight")) {
        return std::nullopt;
    }
    const std::string& path = parameters_[index];
    if (std::ranges::any_of(lora_.patterns, [&](const std::string& pattern) {
            return glob_match(pattern, path);
        })) {
        return path;
    }
    return std::nullopt;
}

std::pair<std::string, std::string>
PythonTarget::adapters(const std::string& path, const Dims& weight, ScalarKind dtype) {
    if (const auto found = adapters_.find(path); found != adapters_.end()) {
        return found->second;
    }
    const std::string block = path.substr(0, path.size() - std::string_view(".weight").size());
    const auto add = [&](const std::string& adapter, const Dims& shape) {
        const std::string argument = "p" + std::to_string(parameters_.size());
        parameters_.push_back(block + "." + adapter);
        parameter_types_.emplace_back(shape, dtype);
        arguments_.insert(arguments_.begin() + static_cast<std::ptrdiff_t>(parameters_end_),
                          argument);
        ++parameters_end_;
        return argument;
    };
    const std::string a = add("lora_a", {lora_.rank, weight[1]});
    const std::string b = add("lora_b", {weight[0], lora_.rank});
    return adapters_.emplace(path, std::make_pair(a, b)).first->second;
}

std::string PythonTarget::string_list(const std::vector<std::string>& items) {
    std::string out = "[";
    for (std::size_t i = 0; i < items.size(); ++i) {
        out += (i == 0 ? "\"" : ", \"") + items[i] + "\"";
    }
    return out + "]";
}

} // namespace linnet::backend
