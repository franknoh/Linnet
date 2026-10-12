#include "linnet/backend/recorded_graph.hpp"

#include "linnet/ast/ast.hpp"
#include "linnet/backend/kernel.hpp"
#include "linnet/diagnostic/json.hpp"
#include "linnet/ir/ir.hpp"
#include "linnet/support/text.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

constexpr std::array<std::pair<Elementwise, std::string_view>, 24> elementwise_spellings = {{
    {Elementwise::Add, "add"},       {Elementwise::Sub, "sub"},     {Elementwise::Mul, "mul"},
    {Elementwise::Div, "div"},       {Elementwise::Rem, "rem"},     {Elementwise::Min, "min"},
    {Elementwise::Max, "max"},       {Elementwise::And, "and"},     {Elementwise::Or, "or"},
    {Elementwise::BitAnd, "bitand"}, {Elementwise::BitOr, "bitor"}, {Elementwise::BitXor, "bitxor"},
    {Elementwise::Shl, "shl"},       {Elementwise::Shr, "shr"},     {Elementwise::Not, "not"},
    {Elementwise::Neg, "neg"},       {Elementwise::Exp, "exp"},     {Elementwise::Log, "log"},
    {Elementwise::Sqrt, "sqrt"},     {Elementwise::Rsqrt, "rsqrt"}, {Elementwise::Sin, "sin"},
    {Elementwise::Cos, "cos"},       {Elementwise::Tanh, "tanh"},   {Elementwise::Abs, "abs"},
}};

std::string spelling(Elementwise kind) {
    for (const auto& [each, name] : elementwise_spellings) {
        if (each == kind) {
            return std::string(name);
        }
    }
    throw std::logic_error("an elementwise operation without a spelling");
}

constexpr std::array<std::pair<Literal::Kind, std::string_view>, 5> literal_spellings = {{
    {Literal::Kind::Integer, "integer"},
    {Literal::Kind::Real, "real"},
    {Literal::Kind::Boolean, "boolean"},
    {Literal::Kind::Lowest, "lowest"},
    {Literal::Kind::Highest, "highest"},
}};

std::string_view spelling(Literal::Kind kind) {
    for (const auto& [each, name] : literal_spellings) {
        if (each == kind) {
            return name;
        }
    }
    return "integer";
}

std::string dtype_attr(ScalarKind dtype) {
    return std::string(sema::scalar_name(dtype));
}

// A target answered differently on replay than when it was recorded, or a
// recorded node is malformed.
struct Diverged : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace

const RecordedAttr* RecordedNode::attr(std::string_view key) const {
    for (const auto& [name, value] : attrs) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

// ------------------------------------------------------------- recording

RecordingTarget::RecordingTarget(GraphTarget& inner, std::string target) : inner_(inner) {
    graph_.target = std::move(target);
    open_.push_back(&graph_.nodes);
}

std::unique_ptr<KernelTarget> RecordingTarget::kernel_target() const {
    return inner_.kernel_target();
}

RecordedNode& RecordingTarget::add(std::string op) {
    RecordedNode& node = open_.back()->emplace_back();
    node.op = std::move(op);
    return node;
}

std::uint32_t
RecordingTarget::define(const std::string& name, const Dims& shape, ScalarKind dtype) {
    const auto id = static_cast<std::uint32_t>(graph_.values.size());
    graph_.values.push_back({name, shape, dtype});
    ids_[name] = id;
    return id;
}

RecordedOperand RecordingTarget::operand(const TensorInfo& value) {
    const auto found = ids_.find(value.name);
    // A name no call returned (a literal the exporter spelled itself) is a
    // value of its own, which a replay passes on as it is.
    const std::uint32_t id =
        found != ids_.end() ? found->second : define(value.name, value.shape, value.dtype);
    return {id, value.shape, value.dtype};
}

RecordedOperand RecordingTarget::operand(const std::optional<TensorInfo>& value) {
    return value ? operand(*value) : RecordedOperand{};
}

std::uint32_t RecordingTarget::result(RecordedNode& node,
                                      const std::string& name,
                                      const Dims& shape,
                                      ScalarKind dtype) {
    const std::uint32_t id = define(name, shape, dtype);
    node.results.push_back(id);
    return id;
}

void RecordingTarget::results(RecordedNode& node,
                              const std::vector<std::string>& names,
                              const std::vector<TensorInfo>& like) {
    for (std::size_t i = 0; i < names.size(); ++i) {
        const TensorInfo shape = i < like.size() ? like[i] : TensorInfo{};
        result(node, names[i], shape.shape, shape.dtype);
    }
}

RecordedRegion& RecordingTarget::open_region(RecordedNode& node,
                                             std::string role,
                                             const std::vector<TensorInfo>& arguments) {
    RecordedRegion& region = node.regions.emplace_back();
    region.role = std::move(role);
    for (const TensorInfo& argument : arguments) {
        region.arguments.push_back(define(argument.name, argument.shape, argument.dtype));
    }
    open_.push_back(&region.nodes);
    return region;
}

void RecordingTarget::close_region() {
    open_.pop_back();
}

std::string RecordingTarget::input(const std::string& name, const Dims& shape, ScalarKind dtype) {
    RecordedNode& node = add("input");
    node.attrs = {{"name", name}, {"shape", shape}, {"dtype", dtype_attr(dtype)}};
    const std::string out = inner_.input(name, shape, dtype);
    result(node, out, shape, dtype);
    return out;
}

std::string
RecordingTarget::parameter(const std::string& path, const Dims& shape, ScalarKind dtype) {
    RecordedNode& node = add("parameter");
    node.attrs = {{"path", path}, {"shape", shape}, {"dtype", dtype_attr(dtype)}};
    const std::string out = inner_.parameter(path, shape, dtype);
    result(node, out, shape, dtype);
    return out;
}

std::string RecordingTarget::state(const std::string& path, const Dims& shape, ScalarKind dtype) {
    RecordedNode& node = add("state");
    node.attrs = {{"path", path}, {"shape", shape}, {"dtype", dtype_attr(dtype)}};
    const std::string out = inner_.state(path, shape, dtype);
    result(node, out, shape, dtype);
    return out;
}

std::string RecordingTarget::constant(const Literal& literal, ScalarKind dtype) {
    RecordedNode& node = add("constant");
    node.attrs = {{"literal", literal}, {"dtype", dtype_attr(dtype)}};
    const std::string out = inner_.constant(literal, dtype);
    result(node, out, {}, dtype);
    return out;
}

std::string RecordingTarget::elementwise(Elementwise kind,
                                         const std::vector<TensorInfo>& operands,
                                         const Dims& shape,
                                         ScalarKind dtype) {
    RecordedNode& node = add("elementwise");
    node.attrs = {{"kind", spelling(kind)}, {"shape", shape}, {"dtype", dtype_attr(dtype)}};
    for (const TensorInfo& each : operands) {
        node.operands.push_back(operand(each));
    }
    const std::string out = inner_.elementwise(kind, operands, shape, dtype);
    result(node, out, shape, dtype);
    return out;
}

std::string RecordingTarget::compare(ir::CompareKind kind,
                                     const TensorInfo& a,
                                     const TensorInfo& b,
                                     const Dims& shape) {
    RecordedNode& node = add("compare");
    node.attrs = {{"kind", std::string(ir::compare_spelling(kind))}, {"shape", shape}};
    node.operands = {operand(a), operand(b)};
    const std::string out = inner_.compare(kind, a, b, shape);
    result(node, out, shape, ScalarKind::Bool);
    return out;
}

std::string RecordingTarget::select(const TensorInfo& condition,
                                    const TensorInfo& on_true,
                                    const TensorInfo& on_false,
                                    const Dims& shape,
                                    ScalarKind dtype) {
    RecordedNode& node = add("select");
    node.attrs = {{"shape", shape}, {"dtype", dtype_attr(dtype)}};
    node.operands = {operand(condition), operand(on_true), operand(on_false)};
    const std::string out = inner_.select(condition, on_true, on_false, shape, dtype);
    result(node, out, shape, dtype);
    return out;
}

std::string RecordingTarget::convert(const TensorInfo& value, ScalarKind dtype) {
    RecordedNode& node = add("convert");
    node.attrs = {{"dtype", dtype_attr(dtype)}};
    node.operands = {operand(value)};
    const std::string out = inner_.convert(value, dtype);
    result(node, out, value.shape, dtype);
    return out;
}

std::string RecordingTarget::reshape(const TensorInfo& value, const Dims& shape) {
    RecordedNode& node = add("reshape");
    node.attrs = {{"shape", shape}};
    node.operands = {operand(value)};
    const std::string out = inner_.reshape(value, shape);
    result(node, out, shape, value.dtype);
    return out;
}

std::string
RecordingTarget::transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) {
    RecordedNode& node = add("transpose");
    node.attrs = {{"permutation", permutation}, {"shape", shape}};
    node.operands = {operand(value)};
    const std::string out = inner_.transpose(value, permutation, shape);
    result(node, out, shape, value.dtype);
    return out;
}

std::string
RecordingTarget::broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) {
    RecordedNode& node = add("broadcast");
    node.attrs = {{"dims", dims}, {"shape", shape}};
    node.operands = {operand(value)};
    const std::string out = inner_.broadcast(value, dims, shape);
    result(node, out, shape, value.dtype);
    return out;
}

std::optional<std::string> RecordingTarget::contract(const TensorInfo& lhs,
                                                     const Dims& lhs_axes,
                                                     const TensorInfo& rhs,
                                                     const Dims& rhs_axes,
                                                     const Dims& out_axes,
                                                     const Dims& shape,
                                                     ScalarKind dtype) {
    RecordedNode& node = add("contract");
    node.attrs = {{"lhs_axes", lhs_axes},
                  {"rhs_axes", rhs_axes},
                  {"out_axes", out_axes},
                  {"shape", shape},
                  {"dtype", dtype_attr(dtype)}};
    node.operands = {operand(lhs), operand(rhs)};
    const auto out = inner_.contract(lhs, lhs_axes, rhs, rhs_axes, out_axes, shape, dtype);
    if (out) {
        result(node, *out, shape, dtype);
    } else {
        node.declined = true;
    }
    return out;
}

void RecordingTarget::enable_placement(int slots) {
    RecordedNode& node = add("enable_placement");
    node.attrs = {{"slots", std::int64_t{slots}}};
    inner_.enable_placement(slots);
}

void RecordingTarget::set_slot(int slot) {
    RecordedNode& node = add("set_slot");
    node.attrs = {{"slot", std::int64_t{slot}}};
    inner_.set_slot(slot);
}

std::string RecordingTarget::transfer(const TensorInfo& value, int slot) {
    RecordedNode& node = add("transfer");
    node.attrs = {{"slot", std::int64_t{slot}}};
    node.operands = {operand(value)};
    const std::string out = inner_.transfer(value, slot);
    result(node, out, value.shape, value.dtype);
    return out;
}

void RecordingTarget::release(const std::vector<std::string>& names) {
    RecordedNode& node = add("release");
    for (const std::string& name : names) {
        const auto found = ids_.find(name);
        const RecordedValue* known = found == ids_.end() ? nullptr : &graph_.values[found->second];
        node.operands.push_back(operand(TensorInfo{
            name, known ? known->shape : Dims{}, known ? known->dtype : ScalarKind::F32}));
    }
    inner_.release(names);
}

std::string RecordingTarget::gather(const TensorInfo& value) {
    RecordedNode& node = add("gather_parts");
    node.operands = {operand(value)};
    const std::string out = inner_.gather(value);
    result(node, out, value.shape, value.dtype);
    return out;
}

void RecordingTarget::begin_remat() {
    add("begin_remat");
    inner_.begin_remat();
}

void RecordingTarget::end_remat() {
    add("end_remat");
    inner_.end_remat();
}

std::string RecordingTarget::slice(const TensorInfo& value,
                                   const Dims& starts,
                                   const Dims& limits,
                                   const Dims& strides,
                                   const Dims& shape) {
    RecordedNode& node = add("slice");
    node.attrs = {{"starts", starts}, {"limits", limits}, {"strides", strides}, {"shape", shape}};
    node.operands = {operand(value)};
    const std::string out = inner_.slice(value, starts, limits, strides, shape);
    result(node, out, shape, value.dtype);
    return out;
}

std::string RecordingTarget::concat(const std::vector<TensorInfo>& parts,
                                    std::int64_t axis,
                                    const Dims& shape) {
    RecordedNode& node = add("concat");
    node.attrs = {{"axis", axis}, {"shape", shape}};
    for (const TensorInfo& part : parts) {
        node.operands.push_back(operand(part));
    }
    const std::string out = inner_.concat(parts, axis, shape);
    result(node, out, shape, parts.empty() ? ScalarKind::F32 : parts.front().dtype);
    return out;
}

std::string RecordingTarget::iota(std::int64_t length) {
    RecordedNode& node = add("iota");
    node.attrs = {{"length", length}};
    const std::string out = inner_.iota(length);
    result(node, out, {length}, ScalarKind::I64);
    return out;
}

std::string
RecordingTarget::gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) {
    RecordedNode& node = add("gather");
    node.attrs = {{"shape", shape}};
    node.operands = {operand(source), operand(indices)};
    const std::string out = inner_.gather(source, indices, shape);
    result(node, out, shape, source.dtype);
    return out;
}

std::string RecordingTarget::reduce(Reduction kind,
                                    const TensorInfo& body,
                                    const Dims& dims,
                                    const Dims& shape) {
    RecordedNode& node = add("reduce");
    node.attrs = {
        {"kind", std::string(ir::reduce_spelling(kind))}, {"dims", dims}, {"shape", shape}};
    node.operands = {operand(body)};
    const std::string out = inner_.reduce(kind, body, dims, shape);
    result(node, out, shape, body.dtype);
    return out;
}

void RecordingTarget::begin_call(const CallInfo& call) {
    RecordedNode& node = add("begin_call");
    node.attrs = {{"callee", call.callee},
                  {"semantic", std::int64_t{call.semantic ? 1 : 0}},
                  {"path", call.path},
                  {"generics", call.generics}};
    for (const auto& argument : call.arguments) {
        node.operands.push_back(operand(argument));
    }
    if (inner_.wants_calls()) {
        inner_.begin_call(call);
    }
}

void RecordingTarget::end_call(const std::vector<TensorInfo>& results) {
    RecordedNode& node = add("end_call");
    for (const TensorInfo& each : results) {
        node.operands.push_back(operand(each));
    }
    if (inner_.wants_calls()) {
        inner_.end_call(results);
    }
}

std::optional<std::string>
RecordingTarget::native_call(const std::string& implementation,
                             const std::vector<std::optional<TensorInfo>>& operands,
                             const Dims& shape,
                             ScalarKind dtype) {
    RecordedNode& node = add("native_call");
    node.attrs = {{"implementation", implementation},
                  {"generics", call_generics()},
                  {"shape", shape},
                  {"dtype", dtype_attr(dtype)}};
    for (const auto& each : operands) {
        node.operands.push_back(operand(each));
    }
    inner_.set_call_generics(call_generics());
    const auto out = inner_.native_call(implementation, operands, shape, dtype);
    if (out) {
        result(node, *out, shape, dtype);
    } else {
        node.declined = true;
    }
    return out;
}

std::optional<std::string> RecordingTarget::update_row(const TensorInfo& stack,
                                                       const TensorInfo& index,
                                                       const TensorInfo& value) {
    RecordedNode& node = add("update_row");
    node.operands = {operand(stack), operand(index), operand(value)};
    const auto out = inner_.update_row(stack, index, value);
    if (out) {
        result(node, *out, stack.shape, stack.dtype);
    } else {
        node.declined = true;
    }
    return out;
}

void RecordingTarget::while_initial_condition(const TensorInfo& predicate) {
    RecordedNode& node = add("while_initial_condition");
    node.operands = {operand(predicate)};
    inner_.while_initial_condition(predicate);
}

std::vector<std::string> RecordingTarget::begin_while(const std::vector<TensorInfo>& initial) {
    RecordedNode& node = add("begin_while");
    for (const TensorInfo& each : initial) {
        node.operands.push_back(operand(each));
    }
    const std::vector<std::string> out = inner_.begin_while(initial);
    results(node, out, initial);
    loops_.push_back(initial);
    return out;
}

std::vector<std::string> RecordingTarget::while_condition(const TensorInfo& predicate) {
    RecordedNode& node = add("while_condition");
    node.operands = {operand(predicate)};
    const std::vector<std::string> out = inner_.while_condition(predicate);
    results(node, out, loops_.empty() ? std::vector<TensorInfo>{} : loops_.back());
    return out;
}

void RecordingTarget::while_trailing_condition(const TensorInfo& predicate) {
    RecordedNode& node = add("while_trailing_condition");
    node.operands = {operand(predicate)};
    inner_.while_trailing_condition(predicate);
}

std::vector<std::string> RecordingTarget::end_while(const std::vector<TensorInfo>& next) {
    RecordedNode& node = add("end_while");
    for (const TensorInfo& each : next) {
        node.operands.push_back(operand(each));
    }
    const std::vector<std::string> out = inner_.end_while(next);
    results(node, out, next);
    if (!loops_.empty()) {
        loops_.pop_back();
    }
    return out;
}

std::optional<std::vector<std::string>> RecordingTarget::launch_kernel(const KernelLaunch& launch) {
    RecordedNode& node = add("launch_kernel");
    std::vector<Dims> shapes;
    std::vector<std::string> dtypes;
    std::vector<std::string> names;
    for (const TensorInfo& each : launch.results) {
        shapes.push_back(each.shape);
        dtypes.push_back(dtype_attr(each.dtype));
        names.push_back(each.name);
    }
    node.attrs = {{"op", launch.op},
                  {"name", launch.name},
                  {"program", launch.program},
                  {"grid", launch.grid},
                  {"result_shapes", shapes},
                  {"result_dtypes", dtypes},
                  {"result_names", names},
                  {"warps", launch.warps},
                  {"stages", launch.stages},
                  {"has_body", std::int64_t{launch.body ? 1 : 0}},
                  {"has_pullback", std::int64_t{launch.pullback ? 1 : 0}}};
    for (const TensorInfo& each : launch.arguments) {
        node.operands.push_back(operand(each));
    }
    KernelLaunch recorded = launch;
    if (launch.body) {
        recorded.body = [this, &node, body = launch.body, like = launch.results](
                            const std::vector<TensorInfo>& arguments) {
            RecordedRegion& region = open_region(node, "body", arguments);
            const std::vector<std::string> out = body(arguments);
            close_region();
            for (std::size_t i = 0; i < out.size(); ++i) {
                const TensorInfo shape = i < like.size() ? like[i] : TensorInfo{};
                region.results.push_back(
                    operand(TensorInfo{out[i], shape.shape, shape.dtype}).value);
            }
            return out;
        };
    }
    if (launch.pullback) {
        recorded.pullback = [this, &node, pullback = launch.pullback](
                                const std::vector<TensorInfo>& arguments,
                                const TensorInfo& result,
                                const TensorInfo& grad) {
            std::vector<TensorInfo> given = arguments;
            given.push_back(result);
            given.push_back(grad);
            RecordedRegion& region = open_region(node, "pullback", given);
            const auto out = pullback(arguments, result, grad);
            close_region();
            for (std::size_t i = 0; i < out.size(); ++i) {
                region.results.push_back(
                    out[i]
                        ? operand(TensorInfo{*out[i], arguments[i].shape, arguments[i].dtype}).value
                        : no_value);
            }
            return out;
        };
    }
    const auto out = inner_.launch_kernel(recorded);
    if (out) {
        results(node, *out, launch.results);
    } else {
        node.declined = true;
    }
    return out;
}

std::vector<std::string>
RecordingTarget::begin_custom_gradient(const std::vector<TensorInfo>& arguments) {
    RecordedNode& node = add("begin_custom_gradient");
    for (const TensorInfo& each : arguments) {
        node.operands.push_back(operand(each));
    }
    const std::vector<std::string> out = inner_.begin_custom_gradient(arguments);
    results(node, out, arguments);
    return out;
}

std::string RecordingTarget::end_custom_gradient(const std::vector<TensorInfo>& arguments,
                                                 const TensorInfo& result_info,
                                                 const Pullback& pullback) {
    RecordedNode& node = add("end_custom_gradient");
    node.attrs = {{"arguments", static_cast<std::int64_t>(arguments.size())}};
    for (const TensorInfo& each : arguments) {
        node.operands.push_back(operand(each));
    }
    node.operands.push_back(operand(result_info));
    const Pullback recorded = [this, &node, pullback](const std::vector<TensorInfo>& given,
                                                      const TensorInfo& out,
                                                      const TensorInfo& grad) {
        std::vector<TensorInfo> all = given;
        all.push_back(out);
        all.push_back(grad);
        RecordedRegion& region = open_region(node, "pullback", all);
        const auto gradients = pullback(given, out, grad);
        close_region();
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            region.results.push_back(
                gradients[i]
                    ? operand(TensorInfo{*gradients[i], given[i].shape, given[i].dtype}).value
                    : no_value);
        }
        return gradients;
    };
    const std::string out = inner_.end_custom_gradient(arguments, result_info, recorded);
    result(node, out, result_info.shape, result_info.dtype);
    return out;
}

std::string RecordingTarget::cumsum(const TensorInfo& value, std::int64_t axis) {
    RecordedNode& node = add("cumsum");
    node.attrs = {{"axis", axis}};
    node.operands = {operand(value)};
    const std::string out = inner_.cumsum(value, axis);
    result(node, out, value.shape, value.dtype);
    return out;
}

std::optional<std::string> RecordingTarget::scatter_add(const Dims& shape,
                                                        ScalarKind dtype,
                                                        const TensorInfo& indices,
                                                        const TensorInfo& values) {
    RecordedNode& node = add("scatter_add");
    node.attrs = {{"shape", shape}, {"dtype", dtype_attr(dtype)}};
    node.operands = {operand(indices), operand(values)};
    const auto out = inner_.scatter_add(shape, dtype, indices, values);
    if (out) {
        result(node, *out, shape, dtype);
    } else {
        node.declined = true;
    }
    return out;
}

std::vector<std::string> RecordingTarget::begin_counted(std::int64_t start,
                                                        std::int64_t stop,
                                                        const std::vector<TensorInfo>& initial) {
    RecordedNode& node = add("begin_counted");
    node.attrs = {{"start", start}, {"stop", stop}};
    for (const TensorInfo& each : initial) {
        node.operands.push_back(operand(each));
    }
    const std::vector<std::string> out = inner_.begin_counted(start, stop, initial);
    std::vector<TensorInfo> like{{"", {}, ScalarKind::I64}};
    like.insert(like.end(), initial.begin(), initial.end());
    results(node, out, like);
    return out;
}

std::vector<std::string> RecordingTarget::end_counted(const std::vector<TensorInfo>& next) {
    RecordedNode& node = add("end_counted");
    for (const TensorInfo& each : next) {
        node.operands.push_back(operand(each));
    }
    const std::vector<std::string> out = inner_.end_counted(next);
    results(node, out, next);
    return out;
}

std::string RecordingTarget::finish(const std::vector<TensorInfo>& results_info,
                                    const std::vector<std::pair<std::string, TensorInfo>>& states,
                                    const std::string& module_path,
                                    const std::string& block_name,
                                    const std::string& entry_name) {
    RecordedNode& node = add("finish");
    std::vector<std::string> paths;
    for (const TensorInfo& each : results_info) {
        node.operands.push_back(operand(each));
    }
    for (const auto& [path, each] : states) {
        paths.push_back(path);
        node.operands.push_back(operand(each));
    }
    node.attrs = {{"module", module_path},
                  {"block", block_name},
                  {"entry", entry_name},
                  {"results", static_cast<std::int64_t>(results_info.size())},
                  {"states", paths}};
    return inner_.finish(results_info, states, module_path, block_name, entry_name);
}

// --------------------------------------------------------------- replay

namespace {

class Replayer {
public:
    Replayer(const RecordedGraph& graph, GraphTarget& target) : target_(target) {
        names_.reserve(graph.values.size());
        for (const RecordedValue& value : graph.values) {
            names_.push_back(value.name);
        }
    }

    std::string run(const std::deque<RecordedNode>& nodes) {
        std::string out;
        for (const RecordedNode& node : nodes) {
            if (auto text = replay(node)) {
                out = std::move(*text);
            }
        }
        return out;
    }

private:
    template <typename T>
    const T& attr(const RecordedNode& node, std::string_view key) const {
        const RecordedAttr* found = node.attr(key);
        if (found == nullptr || !std::holds_alternative<T>(*found)) {
            throw Diverged("`" + node.op + "` has no attribute `" + std::string(key) + "`");
        }
        return std::get<T>(*found);
    }

    const Dims& dims(const RecordedNode& node, std::string_view key) const {
        return attr<Dims>(node, key);
    }

    std::int64_t integer(const RecordedNode& node, std::string_view key) const {
        return attr<std::int64_t>(node, key);
    }

    const std::string& text(const RecordedNode& node, std::string_view key) const {
        return attr<std::string>(node, key);
    }

    ScalarKind dtype(const RecordedNode& node, std::string_view key = "dtype") const {
        return dtype_of(text(node, key));
    }

    static ScalarKind dtype_of(const std::string& name) {
        const auto found = sema::scalar_from_name(name);
        if (!found) {
            throw Diverged("unknown dtype `" + name + "`");
        }
        return *found;
    }

    TensorInfo info(const RecordedOperand& operand) const {
        return {names_.at(operand.value), operand.shape, operand.dtype};
    }

    TensorInfo info(const RecordedNode& node, std::size_t i) const {
        if (i >= node.operands.size()) {
            throw Diverged("`" + node.op + "` has too few operands");
        }
        return info(node.operands[i]);
    }

    std::vector<TensorInfo>
    infos(const RecordedNode& node, std::size_t from = 0, std::size_t count = SIZE_MAX) const {
        std::vector<TensorInfo> out;
        for (std::size_t i = from; i < node.operands.size() && out.size() < count; ++i) {
            out.push_back(info(node.operands[i]));
        }
        return out;
    }

    std::vector<std::optional<TensorInfo>> optional_infos(const RecordedNode& node) const {
        std::vector<std::optional<TensorInfo>> out;
        out.reserve(node.operands.size());
        for (const RecordedOperand& operand : node.operands) {
            out.push_back(operand.value == no_value ? std::nullopt : std::optional(info(operand)));
        }
        return out;
    }

    void bind(const RecordedNode& node, const std::vector<std::string>& names) {
        if (names.size() != node.results.size()) {
            throw Diverged("`" + node.op + "` returned " + std::to_string(names.size()) +
                           " values, recorded " + std::to_string(node.results.size()));
        }
        for (std::size_t i = 0; i < names.size(); ++i) {
            names_[node.results[i]] = names[i];
        }
    }

    void bind(const RecordedNode& node, const std::string& name) {
        bind(node, std::vector<std::string>{name});
    }

    void bind(const RecordedNode& node, const std::optional<std::string>& name) {
        if (node.declined != !name) {
            throw Diverged("`" + node.op + "` " + (name ? "took" : "declined") +
                           " what the recorded target " + (name ? "declined" : "took"));
        }
        if (name) {
            bind(node, *name);
        }
    }

    // The next run of `node`'s callback of `role`: its arguments bound to
    // `given`, its nodes replayed, and its results' names.
    const RecordedRegion& enter(const RecordedNode& node,
                                std::string_view role,
                                std::size_t& next,
                                const std::vector<TensorInfo>& given) {
        std::size_t seen = 0;
        for (const RecordedRegion& region : node.regions) {
            if (region.role != role) {
                continue;
            }
            if (seen++ < next) {
                continue;
            }
            ++next;
            if (region.arguments.size() != given.size()) {
                throw Diverged("a `" + std::string(role) + "` callback took " +
                               std::to_string(given.size()) + " values, recorded " +
                               std::to_string(region.arguments.size()));
            }
            for (std::size_t i = 0; i < given.size(); ++i) {
                names_[region.arguments[i]] = given[i].name;
            }
            run(region.nodes);
            return region;
        }
        throw Diverged("`" + node.op + "` ran a `" + std::string(role) +
                       "` callback the recorded target did not");
    }

    std::optional<std::string> replay(const RecordedNode& node) {
        const std::string& op = node.op;
        if (op == "input") {
            bind(node, target_.input(text(node, "name"), dims(node, "shape"), dtype(node)));
        } else if (op == "parameter") {
            bind(node, target_.parameter(text(node, "path"), dims(node, "shape"), dtype(node)));
        } else if (op == "state") {
            bind(node, target_.state(text(node, "path"), dims(node, "shape"), dtype(node)));
        } else if (op == "constant") {
            bind(node, target_.constant(attr<Literal>(node, "literal"), dtype(node)));
        } else if (op == "elementwise") {
            const std::string& kind = text(node, "kind");
            const auto found = std::ranges::find(elementwise_spellings,
                                                 std::string_view(kind),
                                                 &std::pair<Elementwise, std::string_view>::second);
            if (found == elementwise_spellings.end()) {
                throw Diverged("unknown elementwise operation `" + kind + "`");
            }
            bind(node,
                 target_.elementwise(found->first, infos(node), dims(node, "shape"), dtype(node)));
        } else if (op == "compare") {
            const auto kind = ir::parse_compare(text(node, "kind"));
            if (!kind) {
                throw Diverged("unknown comparison `" + text(node, "kind") + "`");
            }
            bind(node, target_.compare(*kind, info(node, 0), info(node, 1), dims(node, "shape")));
        } else if (op == "select") {
            bind(
                node,
                target_.select(
                    info(node, 0), info(node, 1), info(node, 2), dims(node, "shape"), dtype(node)));
        } else if (op == "convert") {
            bind(node, target_.convert(info(node, 0), dtype(node)));
        } else if (op == "reshape") {
            bind(node, target_.reshape(info(node, 0), dims(node, "shape")));
        } else if (op == "transpose") {
            bind(node,
                 target_.transpose(info(node, 0), dims(node, "permutation"), dims(node, "shape")));
        } else if (op == "broadcast") {
            bind(node, target_.broadcast(info(node, 0), dims(node, "dims"), dims(node, "shape")));
        } else if (op == "contract") {
            bind(node,
                 target_.contract(info(node, 0),
                                  dims(node, "lhs_axes"),
                                  info(node, 1),
                                  dims(node, "rhs_axes"),
                                  dims(node, "out_axes"),
                                  dims(node, "shape"),
                                  dtype(node)));
        } else if (op == "enable_placement") {
            target_.enable_placement(static_cast<int>(integer(node, "slots")));
        } else if (op == "set_slot") {
            target_.set_slot(static_cast<int>(integer(node, "slot")));
        } else if (op == "transfer") {
            bind(node, target_.transfer(info(node, 0), static_cast<int>(integer(node, "slot"))));
        } else if (op == "release") {
            std::vector<std::string> names;
            for (const TensorInfo& each : infos(node)) {
                names.push_back(each.name);
            }
            target_.release(names);
        } else if (op == "gather_parts") {
            bind(node, target_.gather(info(node, 0)));
        } else if (op == "begin_remat") {
            target_.begin_remat();
        } else if (op == "end_remat") {
            target_.end_remat();
        } else if (op == "slice") {
            bind(node,
                 target_.slice(info(node, 0),
                               dims(node, "starts"),
                               dims(node, "limits"),
                               dims(node, "strides"),
                               dims(node, "shape")));
        } else if (op == "concat") {
            bind(node, target_.concat(infos(node), integer(node, "axis"), dims(node, "shape")));
        } else if (op == "iota") {
            bind(node, target_.iota(integer(node, "length")));
        } else if (op == "gather") {
            bind(node, target_.gather(info(node, 0), info(node, 1), dims(node, "shape")));
        } else if (op == "reduce") {
            const auto kind = ast::parse_reduction_kind(text(node, "kind"));
            if (!kind) {
                throw Diverged("unknown reduction `" + text(node, "kind") + "`");
            }
            bind(node,
                 target_.reduce(*kind, info(node, 0), dims(node, "dims"), dims(node, "shape")));
        } else if (op == "begin_call") {
            if (target_.wants_calls()) {
                GraphTarget::CallInfo call;
                call.callee = text(node, "callee");
                call.semantic = integer(node, "semantic") != 0;
                call.path = text(node, "path");
                call.generics = attr<std::map<std::string, std::int64_t>>(node, "generics");
                call.arguments = optional_infos(node);
                target_.begin_call(call);
            }
        } else if (op == "end_call") {
            if (target_.wants_calls()) {
                target_.end_call(infos(node));
            }
        } else if (op == "native_call") {
            target_.set_call_generics(attr<std::map<std::string, std::int64_t>>(node, "generics"));
            bind(node,
                 target_.native_call(text(node, "implementation"),
                                     optional_infos(node),
                                     dims(node, "shape"),
                                     dtype(node)));
        } else if (op == "update_row") {
            bind(node, target_.update_row(info(node, 0), info(node, 1), info(node, 2)));
        } else if (op == "while_initial_condition") {
            target_.while_initial_condition(info(node, 0));
        } else if (op == "begin_while") {
            bind(node, target_.begin_while(infos(node)));
        } else if (op == "while_condition") {
            bind(node, target_.while_condition(info(node, 0)));
        } else if (op == "while_trailing_condition") {
            target_.while_trailing_condition(info(node, 0));
        } else if (op == "end_while") {
            bind(node, target_.end_while(infos(node)));
        } else if (op == "launch_kernel") {
            replay_launch(node);
        } else if (op == "begin_custom_gradient") {
            bind(node, target_.begin_custom_gradient(infos(node)));
        } else if (op == "end_custom_gradient") {
            const auto count = static_cast<std::size_t>(integer(node, "arguments"));
            auto next = std::make_shared<std::size_t>(0);
            const GraphTarget::Pullback pullback =
                [this, &node, next](const std::vector<TensorInfo>& given,
                                    const TensorInfo& out,
                                    const TensorInfo& grad) {
                    std::vector<TensorInfo> all = given;
                    all.push_back(out);
                    all.push_back(grad);
                    return gradients(enter(node, "pullback", *next, all));
                };
            bind(node,
                 target_.end_custom_gradient(infos(node, 0, count), info(node, count), pullback));
        } else if (op == "cumsum") {
            bind(node, target_.cumsum(info(node, 0), integer(node, "axis")));
        } else if (op == "scatter_add") {
            bind(node,
                 target_.scatter_add(
                     dims(node, "shape"), dtype(node), info(node, 0), info(node, 1)));
        } else if (op == "begin_counted") {
            bind(node,
                 target_.begin_counted(integer(node, "start"), integer(node, "stop"), infos(node)));
        } else if (op == "end_counted") {
            bind(node, target_.end_counted(infos(node)));
        } else if (op == "finish") {
            const auto count = static_cast<std::size_t>(integer(node, "results"));
            const auto& paths = attr<std::vector<std::string>>(node, "states");
            std::vector<std::pair<std::string, TensorInfo>> states;
            states.reserve(paths.size());
            for (std::size_t i = 0; i < paths.size(); ++i) {
                states.emplace_back(paths[i], info(node, count + i));
            }
            return target_.finish(infos(node, 0, count),
                                  states,
                                  text(node, "module"),
                                  text(node, "block"),
                                  text(node, "entry"));
        } else {
            throw Diverged("unknown node `" + op + "`");
        }
        return std::nullopt;
    }

    std::vector<std::optional<std::string>> gradients(const RecordedRegion& region) const {
        std::vector<std::optional<std::string>> out;
        out.reserve(region.results.size());
        for (const std::uint32_t id : region.results) {
            out.push_back(id == no_value ? std::nullopt : std::optional(names_.at(id)));
        }
        return out;
    }

    void replay_launch(const RecordedNode& node) {
        KernelLaunch launch;
        launch.op = text(node, "op");
        launch.name = text(node, "name");
        launch.program = attr<KernelProgram>(node, "program");
        launch.grid = dims(node, "grid");
        launch.arguments = infos(node);
        const auto& shapes = attr<std::vector<Dims>>(node, "result_shapes");
        const auto& dtypes = attr<std::vector<std::string>>(node, "result_dtypes");
        const auto& names = attr<std::vector<std::string>>(node, "result_names");
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            launch.results.push_back({names.at(i), shapes[i], dtype_of(dtypes.at(i))});
        }
        launch.warps = integer(node, "warps");
        launch.stages = integer(node, "stages");
        if (integer(node, "has_body") != 0) {
            auto next = std::make_shared<std::size_t>(0);
            launch.body = [this, &node, next](const std::vector<TensorInfo>& given) {
                const RecordedRegion& region = enter(node, "body", *next, given);
                std::vector<std::string> out;
                out.reserve(region.results.size());
                for (const std::uint32_t id : region.results) {
                    out.push_back(names_.at(id));
                }
                return out;
            };
        }
        if (integer(node, "has_pullback") != 0) {
            auto next = std::make_shared<std::size_t>(0);
            launch.pullback = [this, &node, next](const std::vector<TensorInfo>& given,
                                                  const TensorInfo& out,
                                                  const TensorInfo& grad) {
                std::vector<TensorInfo> all = given;
                all.push_back(out);
                all.push_back(grad);
                return gradients(enter(node, "pullback", *next, all));
            };
        }
        const auto out = target_.launch_kernel(launch);
        if (node.declined != !out) {
            throw Diverged("a kernel launch went otherwise than recorded");
        }
        if (out) {
            bind(node, *out);
        }
    }

    GraphTarget& target_;
    std::vector<std::string> names_;
};

} // namespace

std::expected<std::string, std::string> replay_graph(const RecordedGraph& graph,
                                                     GraphTarget& target) {
    try {
        return Replayer(graph, target).run(graph.nodes);
    } catch (const Diverged& error) {
        return std::unexpected(std::string("replay: ") + error.what());
    } catch (const KernelError& error) {
        return std::unexpected(std::string("kernel: ") + error.what());
    }
}

// ----------------------------------------------------------------- JSON

namespace {

std::string number(double value) {
    if (!std::isfinite(value)) {
        return json_string(std::isnan(value) ? "nan" : value > 0 ? "inf" : "-inf");
    }
    std::array<char, 32> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    (void)error;
    return {buffer.data(), end};
}

std::string dims_json(const Dims& dims) {
    return "[" + join(dims, ",", [](std::int64_t d) { return std::to_string(d); }) + "]";
}

std::string attr_json(const RecordedAttr& value) {
    return std::visit(
        [](const auto& item) -> std::string {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(item);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return json_string(item);
            } else if constexpr (std::is_same_v<T, Dims>) {
                return dims_json(item);
            } else if constexpr (std::is_same_v<T, std::vector<Dims>>) {
                return "[" + join(item, ",", [](const Dims& d) { return dims_json(d); }) + "]";
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                return "[" + join(item, ",", [](const std::string& s) { return json_string(s); }) +
                       "]";
            } else if constexpr (std::is_same_v<T, std::map<std::string, std::int64_t>>) {
                return "{" +
                       join(item,
                            ",",
                            [](const auto& entry) {
                                return json_string(entry.first) + ":" +
                                       std::to_string(entry.second);
                            }) +
                       "}";
            } else if constexpr (std::is_same_v<T, Literal>) {
                std::string out = "{\"kind\":" + json_string(spelling(item.kind));
                if (item.kind == Literal::Kind::Integer || item.kind == Literal::Kind::Boolean) {
                    out += ",\"value\":" + std::to_string(item.integer);
                } else if (item.kind == Literal::Kind::Real) {
                    out += ",\"value\":" + number(item.real);
                }
                return out + "}";
            } else {
                static_assert(std::is_same_v<T, KernelProgram>);
                return "{\"parameters\":" + attr_json(RecordedAttr(item.parameters)) +
                       ",\"body\":" + json_string(item.body) + ",\"atomics\":{" +
                       join(item.atomics,
                            ",",
                            [](const auto& entry) {
                                return json_string(entry.first) + ":" +
                                       json_string(ir::reduce_spelling(entry.second));
                            }) +
                       "}}";
            }
        },
        value);
}

class JsonWriter {
public:
    explicit JsonWriter(const RecordedGraph& graph) : graph_(graph) {}

    std::string write() {
        std::string out =
            "{\"version\":1,\"target\":" + json_string(graph_.target) + ",\"values\":[";
        for (std::size_t i = 0; i < graph_.values.size(); ++i) {
            const RecordedValue& value = graph_.values[i];
            out += (i == 0 ? "" : ",");
            out += "{\"id\":" + std::to_string(i) + ",\"name\":" + json_string(value.name) +
                   ",\"shape\":" + dims_json(value.shape) +
                   ",\"dtype\":" + json_string(sema::scalar_name(value.dtype)) + "}";
        }
        out += "],\"nodes\":";
        std::size_t at = 0;
        out += nodes(graph_.nodes, at);
        return out + "}\n";
    }

private:
    std::string operand(const RecordedOperand& operand) const {
        if (operand.value == no_value) {
            return "null";
        }
        const RecordedValue& value = graph_.values.at(operand.value);
        if (value.shape == operand.shape && value.dtype == operand.dtype) {
            return std::to_string(operand.value);
        }
        // A value the call viewed with another shape or dtype.
        return "{\"value\":" + std::to_string(operand.value) +
               ",\"shape\":" + dims_json(operand.shape) +
               ",\"dtype\":" + json_string(sema::scalar_name(operand.dtype)) + "}";
    }

    std::string operands(const std::vector<RecordedOperand>& list) const {
        return "[" + join(list, ",", [this](const RecordedOperand& o) { return operand(o); }) + "]";
    }

    static std::string ids(const std::vector<std::uint32_t>& list) {
        return "[" +
               join(list,
                    ",",
                    [](std::uint32_t id) {
                        return id == no_value ? std::string("null") : std::to_string(id);
                    }) +
               "]";
    }

    // The nodes from `at` to the end of `list`, or to the `end_call` that
    // closes the call open around them, which `at` is left on.
    std::string nodes(const std::deque<RecordedNode>& list, std::size_t& at) const {
        std::string out = "[";
        bool first = true;
        for (; at < list.size(); ++at) {
            const RecordedNode& node = list[at];
            if (node.op == "end_call") {
                break;
            }
            out += first ? "" : ",";
            first = false;
            out += this->node(list, at);
        }
        return out + "]";
    }

    std::string node(const std::deque<RecordedNode>& list, std::size_t& at) const {
        const RecordedNode& node = list[at];
        if (node.op == "begin_call") {
            std::string out = "{\"op\":\"call\"";
            for (const auto& [key, value] : node.attrs) {
                out += "," + json_string(key) + ":" + attr_json(value);
            }
            out += ",\"arguments\":" + operands(node.operands);
            ++at;
            out += ",\"body\":" + nodes(list, at);
            const std::vector<RecordedOperand> none;
            out += ",\"results\":" + operands(at < list.size() ? list[at].operands : none);
            return out + "}";
        }
        std::string out = "{\"op\":" + json_string(node.op);
        for (const auto& [key, value] : node.attrs) {
            out += "," + json_string(key) + ":" + attr_json(value);
        }
        out += ",\"operands\":" + operands(node.operands) + ",\"results\":" + ids(node.results);
        if (node.declined) {
            out += ",\"declined\":true";
        }
        if (!node.regions.empty()) {
            out += ",\"regions\":[";
            bool first = true;
            for (const RecordedRegion& region : node.regions) {
                out += first ? "" : ",";
                first = false;
                std::size_t inner = 0;
                out += "{\"role\":" + json_string(region.role) +
                       ",\"arguments\":" + ids(region.arguments) +
                       ",\"nodes\":" + nodes(region.nodes, inner) +
                       ",\"results\":" + ids(region.results) + "}";
            }
            out += "]";
        }
        return out + "}";
    }

    const RecordedGraph& graph_;
};

} // namespace

std::string recorded_graph_json(const RecordedGraph& graph) {
    return JsonWriter(graph).write();
}

} // namespace linnet::backend
