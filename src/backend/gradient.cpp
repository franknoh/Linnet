#include "linnet/backend/gradient.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <set>
#include <utility>

namespace linnet::backend {

using sema::ScalarKind;

namespace {

bool differentiable(const TensorInfo& value) {
    return sema::is_float(value.dtype);
}

} // namespace

// ------------------------------------------------------------------ forward

std::string GradientTarget::input(const std::string& name, const Dims& shape, ScalarKind dtype) {
    std::string id = inner_.input(name, shape, dtype);
    define(id, shape);
    if (sema::is_float(dtype)) {
        sources_[id] |= from_inputs;
    }
    inputs_.push_back({id, shape, dtype});
    input_names_.push_back(name);
    return id;
}

std::string
GradientTarget::parameter(const std::string& path, const Dims& shape, ScalarKind dtype) {
    std::string id = inner_.parameter(path, shape, dtype);
    define(id, shape);
    if (sema::is_float(dtype)) {
        sources_[id] |= from_parameters;
    }
    parameters_.push_back({id, shape, dtype});
    parameter_paths_.push_back(path);
    return id;
}

std::string GradientTarget::state(const std::string& path, const Dims& shape, ScalarKind dtype) {
    std::string id = inner_.state(path, shape, dtype);
    define(id, shape);
    return id;
}

std::string GradientTarget::constant(const Literal& literal, ScalarKind dtype) {
    std::string id = inner_.constant(literal, dtype);
    define(id, Dims{});
    replayable(id, [=, this] { return inner_.constant(literal, dtype); });
    return id;
}

std::string GradientTarget::elementwise(Elementwise kind,
                                        const std::vector<TensorInfo>& operands,
                                        const Dims& shape,
                                        ScalarKind dtype) {
    Step step;
    step.kind = Step::Kind::Elementwise;
    step.op = kind;
    step.operands = operands;
    step.result = {inner_.elementwise(kind, operands, shape, dtype), shape, dtype};
    std::string name = record(std::move(step));
    replayable(name,
               [=, this] { return inner_.elementwise(kind, renamed(operands), shape, dtype); });
    return name;
}

std::string GradientTarget::compare(ir::CompareKind kind,
                                    const TensorInfo& a,
                                    const TensorInfo& b,
                                    const Dims& shape) {
    std::string name = inner_.compare(kind, a, b, shape);
    replayable(name, [=, this] { return inner_.compare(kind, renamed(a), renamed(b), shape); });
    return name;
}

std::string GradientTarget::select(const TensorInfo& condition,
                                   const TensorInfo& on_true,
                                   const TensorInfo& on_false,
                                   const Dims& shape,
                                   ScalarKind dtype) {
    Step step;
    step.kind = Step::Kind::Select;
    step.operands = {condition, on_true, on_false};
    step.result = {inner_.select(condition, on_true, on_false, shape, dtype), shape, dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] {
        return inner_.select(renamed(condition), renamed(on_true), renamed(on_false), shape, dtype);
    });
    return name;
}

std::string GradientTarget::convert(const TensorInfo& value, ScalarKind dtype) {
    Step step;
    step.kind = Step::Kind::Convert;
    step.operands = {value};
    step.result = {inner_.convert(value, dtype), value.shape, dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.convert(renamed(value), dtype); });
    return name;
}

std::string GradientTarget::reshape(const TensorInfo& value, const Dims& shape) {
    Step step;
    step.kind = Step::Kind::Reshape;
    step.operands = {value};
    step.result = {inner_.reshape(value, shape), shape, value.dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.reshape(renamed(value), shape); });
    return name;
}

std::string
GradientTarget::transpose(const TensorInfo& value, const Dims& permutation, const Dims& shape) {
    Step step;
    step.kind = Step::Kind::Transpose;
    step.operands = {value};
    step.first = permutation;
    step.result = {inner_.transpose(value, permutation, shape), shape, value.dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.transpose(renamed(value), permutation, shape); });
    return name;
}

std::string
GradientTarget::broadcast(const TensorInfo& value, const Dims& dims, const Dims& shape) {
    Step step;
    step.kind = Step::Kind::Broadcast;
    step.operands = {value};
    step.first = dims;
    step.result = {inner_.broadcast(value, dims, shape), shape, value.dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.broadcast(renamed(value), dims, shape); });
    return name;
}

std::optional<std::string> GradientTarget::contract(const TensorInfo& lhs,
                                                    const Dims& lhs_axes,
                                                    const TensorInfo& rhs,
                                                    const Dims& rhs_axes,
                                                    const Dims& out_axes,
                                                    const Dims& shape,
                                                    ScalarKind dtype) {
    auto name = inner_.contract(lhs, lhs_axes, rhs, rhs_axes, out_axes, shape, dtype);
    if (!name) {
        return name;
    }
    Step step;
    step.kind = Step::Kind::Contract;
    step.operands = {lhs, rhs};
    step.first = lhs_axes;
    step.second = rhs_axes;
    step.third = out_axes;
    step.result = {*name, shape, dtype};
    std::string recorded = record(std::move(step));
    replayable(recorded, [=, this] {
        const auto made =
            inner_.contract(renamed(lhs), lhs_axes, renamed(rhs), rhs_axes, out_axes, shape, dtype);
        if (!made) {
            throw GradientError("internal: a product the format wrote once it declines again");
        }
        return *made;
    });
    return recorded;
}

std::string GradientTarget::slice(const TensorInfo& value,
                                  const Dims& starts,
                                  const Dims& limits,
                                  const Dims& strides,
                                  const Dims& shape) {
    Step step;
    step.kind = Step::Kind::Slice;
    step.operands = {value};
    step.first = starts;
    step.second = limits;
    step.third = strides;
    step.result = {inner_.slice(value, starts, limits, strides, shape), shape, value.dtype};
    std::string name = record(std::move(step));
    replayable(name,
               [=, this] { return inner_.slice(renamed(value), starts, limits, strides, shape); });
    return name;
}

std::string
GradientTarget::concat(const std::vector<TensorInfo>& parts, std::int64_t axis, const Dims& shape) {
    Step step;
    step.kind = Step::Kind::Concat;
    step.operands = parts;
    step.axis = axis;
    step.result = {inner_.concat(parts, axis, shape),
                   shape,
                   parts.empty() ? ScalarKind::F32 : parts.front().dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.concat(renamed(parts), axis, shape); });
    return name;
}

std::string GradientTarget::cumsum(const TensorInfo& value, std::int64_t axis) {
    Step step;
    step.kind = Step::Kind::Cumsum;
    step.operands = {value};
    step.axis = axis;
    step.result = {inner_.cumsum(value, axis), value.shape, value.dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.cumsum(renamed(value), axis); });
    return name;
}

std::string GradientTarget::iota(std::int64_t length) {
    std::string name = inner_.iota(length);
    replayable(name, [=, this] { return inner_.iota(length); });
    return name;
}

std::string
GradientTarget::gather(const TensorInfo& source, const TensorInfo& indices, const Dims& shape) {
    Step step;
    step.kind = Step::Kind::Gather;
    step.operands = {source, indices};
    step.result = {inner_.gather(source, indices, shape), shape, source.dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.gather(renamed(source), renamed(indices), shape); });
    return name;
}

std::string GradientTarget::reduce(Reduction kind,
                                   const TensorInfo& body,
                                   const Dims& dims,
                                   const Dims& shape) {
    Step step;
    step.kind = Step::Kind::Reduce;
    step.reduction = kind;
    step.operands = {body};
    step.first = dims;
    step.result = {inner_.reduce(kind, body, dims, shape), shape, body.dtype};
    std::string name = record(std::move(step));
    replayable(name, [=, this] { return inner_.reduce(kind, renamed(body), dims, shape); });
    return name;
}

std::optional<std::string>
GradientTarget::native_call(const std::string& implementation,
                            const std::vector<std::optional<TensorInfo>>& operands,
                            const Dims& shape,
                            ScalarKind dtype) {
    (void)implementation;
    (void)operands;
    (void)shape;
    (void)dtype;
    return std::nullopt;
}

std::vector<std::string> GradientTarget::begin_while(const std::vector<TensorInfo>& initial) {
    (void)initial;
    throw GradientError("the gradient of a `while` loop is not supported yet; a `for` loop, "
                        "whose iterations are counted, has one");
}

// ------------------------------------------------------------------ loops

std::vector<std::string> GradientTarget::open_loop(std::int64_t start,
                                                   std::int64_t stop,
                                                   const std::vector<TensorInfo>& initial) {
    if (inner_.supports_counted()) {
        lowered_.push_back({true, stop, ""});
        return inner_.begin_counted(start, stop, initial);
    }
    if (!inner_.supports_while()) {
        throw GradientError("this format writes no runtime loops");
    }
    // As the evaluator writes a `for`: a `while` over the index, carried first.
    std::vector<TensorInfo> carried{scalar_integer(start)};
    carried.insert(carried.end(), initial.begin(), initial.end());
    if (inner_.while_needs_initial_condition()) {
        inner_.while_initial_condition(
            {inner_.compare(ir::CompareKind::Lt, carried.front(), scalar_integer(stop), {}),
             {},
             ScalarKind::Bool});
    }
    const std::vector<std::string> condition = inner_.begin_while(carried);
    const TensorInfo index{condition.front(), {}, ScalarKind::I64};
    std::vector<std::string> body = inner_.while_condition(
        {inner_.compare(ir::CompareKind::Lt, index, scalar_integer(stop), {}),
         {},
         ScalarKind::Bool});
    lowered_.push_back({false, stop, body.front()});
    return body;
}

std::vector<std::string> GradientTarget::close_loop(const std::vector<TensorInfo>& next) {
    const Lowered loop = lowered_.back();
    lowered_.pop_back();
    if (loop.is_counted) {
        return inner_.end_counted(next);
    }
    const TensorInfo advanced{
        inner_.elementwise(Elementwise::Add,
                           {{loop.index, {}, ScalarKind::I64}, scalar_integer(1)},
                           {},
                           ScalarKind::I64),
        {},
        ScalarKind::I64};
    std::vector<TensorInfo> carried{advanced};
    carried.insert(carried.end(), next.begin(), next.end());
    if (inner_.while_needs_trailing_condition()) {
        inner_.while_trailing_condition(
            {inner_.compare(ir::CompareKind::Lt, advanced, scalar_integer(loop.stop), {}),
             {},
             ScalarKind::Bool});
    }
    std::vector<std::string> finals = inner_.end_while(carried);
    finals.erase(finals.begin());
    return finals;
}

// Each carried value is written, as the iteration starts, into a stack of
// every iteration's: the backward pass reads them back in reverse.
std::vector<std::string> GradientTarget::begin_counted(std::int64_t start,
                                                       std::int64_t stop,
                                                       const std::vector<TensorInfo>& initial) {
    if (replaying_) {
        return open_loop(start, stop, initial);
    }
    if (!open_.empty()) {
        throw GradientError("the gradient of a runtime loop inside another is not supported yet");
    }
    const std::int64_t count = std::max<std::int64_t>(stop - start, 0);
    Loop loop;
    loop.start = start;
    loop.stop = stop;
    loop.initial = initial;
    std::vector<TensorInfo> carried = initial;
    for (const TensorInfo& value : initial) {
        Dims shape{count};
        shape.insert(shape.end(), value.shape.begin(), value.shape.end());
        carried.push_back(zeros(shape, value.dtype));
    }
    const std::vector<std::string> names = open_loop(start, stop, carried);
    const std::size_t n = initial.size();
    loop.index = {names.front(), {}, ScalarKind::I64};
    const TensorInfo row = start == 0
                               ? loop.index
                               : TensorInfo{inner_.elementwise(Elementwise::Sub,
                                                               {loop.index, scalar_integer(start)},
                                                               {},
                                                               ScalarKind::I64),
                                            {},
                                            ScalarKind::I64};
    for (std::size_t k = 0; k < n; ++k) {
        const TensorInfo value{names[1 + k], initial[k].shape, initial[k].dtype};
        define(value.name, value.shape);
        // A carried value may come to depend on anything the body reads.
        if (differentiable(value)) {
            sources_[value.name] |= from_inputs | from_parameters;
        }
        loop.carried.push_back(value);
        const TensorInfo stack{names[1 + n + k], carried[n + k].shape, value.dtype};
        const auto written = inner_.update_row(stack, row, value);
        if (!written) {
            throw GradientError("this format cannot keep a loop's iterations for its gradient");
        }
        loop.written.push_back({*written, stack.shape, stack.dtype});
    }
    loop.tape_mark = tape_.size();
    open_.push_back(loops_.size());
    loops_.push_back(std::move(loop));
    return {names.begin(), names.begin() + static_cast<std::ptrdiff_t>(1 + n)};
}

// One step stands for the loop; its body's leave the tape.
std::vector<std::string> GradientTarget::end_counted(const std::vector<TensorInfo>& next) {
    if (replaying_) {
        return close_loop(next);
    }
    const std::size_t at = open_.back();
    open_.pop_back();
    Loop& loop = loops_[at];
    loop.next = next;
    std::vector<TensorInfo> carried = next;
    carried.insert(carried.end(), loop.written.begin(), loop.written.end());
    const std::vector<std::string> finals = close_loop(carried);
    const std::size_t n = loop.initial.size();
    for (std::size_t k = 0; k < n; ++k) {
        const TensorInfo value{finals[k], loop.initial[k].shape, loop.initial[k].dtype};
        define(value.name, value.shape);
        if (differentiable(value)) {
            sources_[value.name] |= from_inputs | from_parameters;
        }
        loop.finals.push_back(value);
        loop.stacks.push_back({finals[n + k], loop.written[k].shape, loop.written[k].dtype});
    }
    loop.steps.assign(tape_.begin() + static_cast<std::ptrdiff_t>(loop.tape_mark), tape_.end());
    tape_.resize(loop.tape_mark);
    Step step;
    step.kind = Step::Kind::Loop;
    step.custom = at;
    step.operands = loop.initial;
    step.result = loop.finals.empty() ? TensorInfo{} : loop.finals.front();
    tape_.push_back(std::move(step));
    return {finals.begin(), finals.begin() + static_cast<std::ptrdiff_t>(n)};
}

void GradientTarget::replayable(const std::string& result, std::function<std::string()> call) {
    if (!replaying_ && !open_.empty()) {
        loops_[open_.back()].log.push_back({std::move(call), result});
    }
}

std::string GradientTarget::renamed(const std::string& name) const {
    const auto found = renames_.find(name);
    return found == renames_.end() ? name : found->second;
}

TensorInfo GradientTarget::renamed(TensorInfo value) const {
    value.name = renamed(value.name);
    return value;
}

std::vector<TensorInfo> GradientTarget::renamed(std::vector<TensorInfo> values) const {
    for (TensorInfo& value : values) {
        value.name = renamed(value.name);
    }
    return values;
}

TensorInfo GradientTarget::zeros(const Dims& shape, ScalarKind dtype) {
    const Literal zero = dtype == ScalarKind::Bool ? Literal{Literal::Kind::Boolean, 0, 0.0}
                         : sema::is_float(dtype)   ? Literal::of_real(0.0)
                                                   : Literal::of_integer(0);
    return widen({inner_.constant(zero, dtype), {}, dtype}, shape);
}

TensorInfo GradientTarget::scalar_integer(std::int64_t value) {
    return {inner_.constant(Literal::of_integer(value), ScalarKind::I64), {}, ScalarKind::I64};
}

// The loop's gradient: a loop over its iterations in reverse, carrying the
// gradients of the carried values and summing those of what the body reads
// from outside. Each iteration's values are read back from the stacks, its
// body emitted again from them, and that body's backward pass run.
void GradientTarget::propagate_loop(const Loop& loop) {
    const std::size_t n = loop.initial.size();
    const std::int64_t count = std::max<std::int64_t>(loop.stop - loop.start, 0);
    std::vector<std::size_t> floats;
    bool is_reached = false;
    for (std::size_t k = 0; k < n; ++k) {
        if (differentiable(loop.initial[k])) {
            floats.push_back(k);
            is_reached = is_reached || adjoints_.contains(loop.finals[k].name);
        }
    }
    if (!is_reached) {
        return;
    }
    const auto is_active = [&](const TensorInfo& value) {
        const auto found = sources_.find(value.name);
        return differentiable(value) && found != sources_.end() && (found->second & wanted_) != 0;
    };
    // What the body reads from outside: defined neither in it nor by the loop.
    std::set<std::string> inside{loop.index.name};
    for (const TensorInfo& value : loop.carried) {
        inside.insert(value.name);
    }
    for (const Replay& entry : loop.log) {
        inside.insert(entry.result);
    }
    std::vector<TensorInfo> outside;
    std::set<std::string> seen;
    const auto note = [&](const TensorInfo& value) {
        if (!inside.contains(value.name) && is_active(value) && seen.insert(value.name).second) {
            const auto shape = shapes_.find(value.name);
            outside.push_back(
                {value.name, shape == shapes_.end() ? value.shape : shape->second, value.dtype});
        }
    };
    for (const Step& step : loop.steps) {
        for (const TensorInfo& operand : step.operands) {
            note(operand);
        }
    }
    for (const TensorInfo& value : loop.next) {
        note(value);
    }

    std::vector<TensorInfo> carried;
    for (const std::size_t k : floats) {
        const auto found = adjoints_.find(loop.finals[k].name);
        carried.push_back(found != adjoints_.end()
                              ? found->second
                              : full(0.0, loop.initial[k].shape, loop.initial[k].dtype));
    }
    for (const TensorInfo& value : outside) {
        carried.push_back(full(0.0, value.shape, value.dtype));
    }
    const std::vector<std::string> names = open_loop(0, count, carried);

    // Iteration `count - 1 - j`, its index and its values.
    const TensorInfo step_index{names.front(), {}, ScalarKind::I64};
    const TensorInfo row{
        inner_.elementwise(
            Elementwise::Sub, {scalar_integer(count - 1), step_index}, {}, ScalarKind::I64),
        {},
        ScalarKind::I64};
    const TensorInfo index = loop.start == 0
                                 ? row
                                 : TensorInfo{inner_.elementwise(Elementwise::Add,
                                                                 {row, scalar_integer(loop.start)},
                                                                 {},
                                                                 ScalarKind::I64),
                                              {},
                                              ScalarKind::I64};
    const TensorInfo position{inner_.reshape(row, {1}), {1}, ScalarKind::I64};
    const auto copy = [&](const std::string& original, const std::string& made) {
        if (const auto shape = shapes_.find(original); shape != shapes_.end()) {
            shapes_.emplace(made, shape->second);
        }
        if (const auto source = sources_.find(original); source != sources_.end()) {
            sources_[made] |= source->second;
        }
    };
    renames_.clear();
    renames_[loop.index.name] = index.name;
    for (std::size_t k = 0; k < n; ++k) {
        const std::string value = inner_.gather(loop.stacks[k], position, loop.initial[k].shape);
        renames_[loop.carried[k].name] = value;
        copy(loop.carried[k].name, value);
    }
    for (const Replay& entry : loop.log) {
        const std::string made = entry.call();
        if (made != entry.result) {
            renames_[entry.result] = made;
            copy(entry.result, made);
        }
    }

    // The body's backward pass, from the gradients the next iteration left.
    std::map<std::string, TensorInfo> outer = std::exchange(adjoints_, {});
    for (std::size_t j = 0; j < floats.size(); ++j) {
        const TensorInfo& initial = loop.initial[floats[j]];
        accumulate(renamed(loop.next[floats[j]]), {names[1 + j], initial.shape, initial.dtype});
    }
    std::vector<Step> steps;
    steps.reserve(loop.steps.size());
    for (const Step& step : loop.steps) {
        Step copied = step;
        copied.operands = renamed(step.operands);
        copied.result = renamed(step.result);
        steps.push_back(std::move(copied));
    }
    std::map<std::string, std::size_t> first;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        first.emplace(steps[i].result.name, i);
    }
    for (std::size_t i = steps.size(); i-- > 0;) {
        const Step& step = steps[i];
        if (first.at(step.result.name) != i || !differentiable(step.result)) {
            continue;
        }
        const auto found = adjoints_.find(step.result.name);
        if (found != adjoints_.end()) {
            const TensorInfo grad = found->second;
            propagate(step, grad);
        }
    }
    std::vector<TensorInfo> next;
    for (const std::size_t k : floats) {
        const auto found = adjoints_.find(renames_.at(loop.carried[k].name));
        next.push_back(found != adjoints_.end()
                           ? found->second
                           : full(0.0, loop.initial[k].shape, loop.initial[k].dtype));
    }
    for (std::size_t j = 0; j < outside.size(); ++j) {
        const TensorInfo sum{names[1 + floats.size() + j], outside[j].shape, outside[j].dtype};
        const auto found = adjoints_.find(outside[j].name);
        next.push_back(found != adjoints_.end() ? apply(Elementwise::Add, {sum, found->second})
                                                : sum);
    }
    adjoints_ = std::move(outer);
    renames_.clear();
    const std::vector<std::string> finals = close_loop(next);
    for (std::size_t j = 0; j < floats.size(); ++j) {
        const TensorInfo& initial = loop.initial[floats[j]];
        accumulate(initial, {finals[j], initial.shape, initial.dtype});
    }
    for (std::size_t j = 0; j < outside.size(); ++j) {
        accumulate(outside[j], {finals[floats.size() + j], outside[j].shape, outside[j].dtype});
    }
}

// An operation whose result is one of its operands (a no-op the format
// skipped) is the operand itself: nothing to record.
std::string GradientTarget::record(Step step) {
    std::string name = step.result.name;
    if (replaying_) {
        return name;
    }
    define(name, step.result.shape);
    const bool is_identity =
        std::any_of(step.operands.begin(), step.operands.end(), [&](const TensorInfo& operand) {
            return operand.name == name;
        });
    if (is_identity) {
        return name;
    }
    if (differentiable(step.result)) {
        std::uint8_t reached = 0;
        for (const TensorInfo& operand : step.operands) {
            if (const auto found = sources_.find(operand.name); found != sources_.end()) {
                reached |= found->second;
            }
        }
        if (reached != 0) {
            sources_[name] |= reached;
        }
    }
    tape_.push_back(std::move(step));
    return name;
}

void GradientTarget::define(const std::string& name, const Dims& shape) {
    if (shapes_.emplace(name, shape).second) {
        order_.emplace(name, order_.size());
    }
}

std::vector<std::string>
GradientTarget::begin_custom_gradient(const std::vector<TensorInfo>& arguments) {
    if (!replaying_ && !open_.empty()) {
        throw GradientError("the gradient of an op with a `grad` inside a runtime loop is not "
                            "supported yet");
    }
    if (!replaying_) {
        customs_.emplace_back(tape_.size(), order_.size());
    }
    return GraphTarget::begin_custom_gradient(arguments);
}

// The body's operations stay emitted -- they compute the result -- but
// leave the tape: one step stands for them, whose backward pass is `grad`.
std::string GradientTarget::end_custom_gradient(const std::vector<TensorInfo>& arguments,
                                                const TensorInfo& result,
                                                const Pullback& pullback) {
    if (replaying_) {
        return result.name;
    }
    const auto [tape, definitions] = customs_.back();
    customs_.pop_back();
    tape_.resize(tape);
    Step step;
    step.kind = Step::Kind::Custom;
    step.operands = arguments;
    step.result = result;
    // A result named as something defined before the op (an argument it
    // returns, an expression computed before) would share that value's
    // gradient: it gets a name of its own.
    const auto defined = order_.find(result.name);
    if (defined != order_.end() && defined->second < definitions) {
        step.result.name = inner_.elementwise(Elementwise::Add,
                                              {result, full(0.0, result.shape, result.dtype)},
                                              result.shape,
                                              result.dtype);
    }
    step.custom = pullbacks_.size();
    pullbacks_.push_back(pullback);
    std::uint8_t reached = 0;
    for (const TensorInfo& argument : arguments) {
        if (const auto found = sources_.find(argument.name); found != sources_.end()) {
            reached |= found->second;
        }
    }
    define(step.result.name, step.result.shape);
    if (reached != 0 && differentiable(step.result)) {
        sources_[step.result.name] |= reached;
    }
    std::string name = step.result.name;
    tape_.push_back(std::move(step));
    return name;
}

// ------------------------------------------------------------------ finish

std::string GradientTarget::finish(const std::vector<TensorInfo>& results,
                                   const std::vector<std::pair<std::string, TensorInfo>>& states,
                                   const std::string& module_path,
                                   const std::string& block_name,
                                   const std::string& entry_name) {
    if (!states.empty()) {
        throw GradientError("an entry that assigns `state` has no gradient");
    }
    if (results.size() != 1 || !results.front().shape.empty() || !differentiable(results.front())) {
        throw GradientError("a gradient is of an entry whose one result is a floating scalar, its "
                            "loss");
    }
    const TensorInfo& loss = results.front();
    // A module-level entry's gradient is with respect to its inputs; a
    // block's, with respect to its parameters.
    const bool by_inputs = block_name.empty();
    wanted_ = by_inputs ? from_inputs : from_parameters;
    replaying_ = true;
    backward(loss);
    const std::vector<TensorInfo>& wrt = by_inputs ? inputs_ : parameters_;
    const std::vector<std::string>& paths = by_inputs ? input_names_ : parameter_paths_;
    std::vector<TensorInfo> outputs{loss};
    std::vector<std::string> labels;
    for (std::size_t i = 0; i < wrt.size(); ++i) {
        if (!differentiable(wrt[i])) {
            continue;
        }
        const auto found = adjoints_.find(wrt[i].name);
        outputs.push_back(found != adjoints_.end() ? found->second
                                                   : full(0.0, wrt[i].shape, wrt[i].dtype));
        labels.push_back(paths[i]);
    }
    inner_.set_gradients(std::move(labels));
    return inner_.finish(outputs, states, module_path, block_name, entry_name);
}

void GradientTarget::backward(const TensorInfo& loss) {
    // A format that merges equal expressions names several recorded
    // operations alike: the first carries the backward pass, once every use
    // after it has added its part.
    std::map<std::string, std::size_t> first;
    for (std::size_t i = 0; i < tape_.size(); ++i) {
        first.emplace(tape_[i].result.name, i);
    }
    adjoints_[loss.name] = full(1.0, {}, loss.dtype);
    for (std::size_t i = tape_.size(); i-- > 0;) {
        const Step& step = tape_[i];
        if (step.kind == Step::Kind::Loop) {
            propagate_loop(loops_[step.custom]);
            continue;
        }
        if (first.at(step.result.name) != i || !differentiable(step.result)) {
            continue;
        }
        const auto found = adjoints_.find(step.result.name);
        if (found == adjoints_.end()) {
            continue;
        }
        const TensorInfo grad = found->second;
        propagate(step, grad);
    }
}

void GradientTarget::accumulate(const TensorInfo& value, const TensorInfo& grad) {
    // Only what the differentiated values reach carries a gradient.
    const auto reached = sources_.find(value.name);
    if (!differentiable(value) || reached == sources_.end() || (reached->second & wanted_) == 0) {
        return;
    }
    // A format that broadcasts elementwise operands itself is handed them
    // unbroadcast, with the result's shape: the value's own shape is the one
    // it was defined with.
    TensorInfo defined = value;
    if (const auto known = shapes_.find(value.name); known != shapes_.end()) {
        defined.shape = known->second;
    }
    TensorInfo matched = grad;
    if (matched.dtype != defined.dtype) {
        matched = {inner_.convert(matched, defined.dtype), matched.shape, defined.dtype};
    }
    if (matched.shape != defined.shape) {
        matched = unbroadcast(matched, defined);
    }
    const auto found = adjoints_.find(value.name);
    if (found == adjoints_.end()) {
        adjoints_.emplace(value.name, matched);
        return;
    }
    found->second = apply(Elementwise::Add, {found->second, matched});
}

// ------------------------------------------------------------------- rules

void GradientTarget::propagate(const Step& step, const TensorInfo& g) {
    const auto& ops = step.operands;
    const TensorInfo& result = step.result;
    switch (step.kind) {
    case Step::Kind::Elementwise: {
        const TensorInfo& a = ops.front();
        const auto zero = [&] { return full(0.0, result.shape, result.dtype); };
        switch (step.op) {
        case Elementwise::Add:
            accumulate(a, g);
            accumulate(ops[1], g);
            return;
        case Elementwise::Sub:
            accumulate(a, g);
            accumulate(ops[1], apply(Elementwise::Neg, {g}));
            return;
        case Elementwise::Mul:
            accumulate(a, apply(Elementwise::Mul, {g, ops[1]}));
            accumulate(ops[1], apply(Elementwise::Mul, {g, a}));
            return;
        case Elementwise::Div: {
            accumulate(a, apply(Elementwise::Div, {g, ops[1]}));
            // d(a / b)/db = -(a / b) / b
            const TensorInfo ratio = apply(Elementwise::Div, {result, ops[1]});
            accumulate(ops[1], apply(Elementwise::Neg, {apply(Elementwise::Mul, {g, ratio})}));
            return;
        }
        case Elementwise::Min:
        case Elementwise::Max: {
            // The operand the result came from takes the gradient; a tie goes
            // to the first.
            const ir::CompareKind kind =
                step.op == Elementwise::Min ? ir::CompareKind::Le : ir::CompareKind::Ge;
            const Dims& shape = result.shape;
            const TensorInfo first{
                inner_.compare(kind, widen(a, shape), widen(ops[1], shape), shape),
                shape,
                ScalarKind::Bool};
            accumulate(a, choose(first, g, zero()));
            accumulate(ops[1], choose(first, zero(), g));
            return;
        }
        case Elementwise::Neg:
            accumulate(a, apply(Elementwise::Neg, {g}));
            return;
        case Elementwise::Exp:
            accumulate(a, apply(Elementwise::Mul, {g, result}));
            return;
        case Elementwise::Log:
            accumulate(a, apply(Elementwise::Div, {g, a}));
            return;
        case Elementwise::Sqrt:
            // d sqrt(a) = 1 / (2 sqrt(a))
            accumulate(a,
                       apply(Elementwise::Div,
                             {apply(Elementwise::Mul, {g, full(0.5, result.shape, result.dtype)}),
                              result}));
            return;
        case Elementwise::Rsqrt:
            // d a^-1/2 = -a^-1/2 / (2a)
            accumulate(a,
                       apply(Elementwise::Mul,
                             {g,
                              apply(Elementwise::Div,
                                    {apply(Elementwise::Mul,
                                           {result, full(-0.5, result.shape, result.dtype)}),
                                     a})}));
            return;
        case Elementwise::Sin:
            accumulate(a, apply(Elementwise::Mul, {g, apply(Elementwise::Cos, {a})}));
            return;
        case Elementwise::Cos:
            accumulate(a,
                       apply(Elementwise::Neg,
                             {apply(Elementwise::Mul, {g, apply(Elementwise::Sin, {a})})}));
            return;
        case Elementwise::Tanh: {
            // d tanh(a) = 1 - tanh(a)^2
            const TensorInfo square = apply(Elementwise::Mul, {result, result});
            accumulate(
                a,
                apply(
                    Elementwise::Mul,
                    {g, apply(Elementwise::Sub, {full(1.0, result.shape, result.dtype), square})}));
            return;
        }
        case Elementwise::Abs: {
            const Dims& shape = result.shape;
            const TensorInfo positive{
                inner_.compare(
                    ir::CompareKind::Ge, widen(a, shape), full(0.0, shape, a.dtype), shape),
                shape,
                ScalarKind::Bool};
            accumulate(a, choose(positive, g, apply(Elementwise::Neg, {g})));
            return;
        }
        default:
            return; // integer and logical operations
        }
    }
    case Step::Kind::Select: {
        const TensorInfo zero = full(0.0, result.shape, result.dtype);
        const TensorInfo condition = widen(ops[0], result.shape);
        accumulate(ops[1], choose(condition, g, zero));
        accumulate(ops[2], choose(condition, zero, g));
        return;
    }
    case Step::Kind::Convert:
        if (differentiable(ops.front())) {
            accumulate(ops.front(),
                       {inner_.convert(g, ops.front().dtype), g.shape, ops.front().dtype});
        }
        return;
    case Step::Kind::Reshape:
        accumulate(ops.front(), {inner_.reshape(g, ops.front().shape), ops.front().shape, g.dtype});
        return;
    case Step::Kind::Transpose: {
        // Axis i of the result is axis permutation[i] of the operand.
        Dims inverse(step.first.size());
        for (std::size_t i = 0; i < step.first.size(); ++i) {
            inverse[static_cast<std::size_t>(step.first[i])] = static_cast<std::int64_t>(i);
        }
        accumulate(ops.front(),
                   {inner_.transpose(g, inverse, ops.front().shape), ops.front().shape, g.dtype});
        return;
    }
    case Step::Kind::Broadcast: {
        // Axis i of the operand is axis dims[i] of the result: the result's
        // other axes, and the operand's axes of one spread wider, sum away.
        const TensorInfo& value = ops.front();
        std::vector<std::size_t> summed;
        std::vector<std::size_t> order; // the operand axes left, in the result's order
        for (std::size_t r = 0; r < result.shape.size(); ++r) {
            const auto at =
                std::find(step.first.begin(), step.first.end(), static_cast<std::int64_t>(r));
            if (at == step.first.end()) {
                summed.push_back(r);
                continue;
            }
            const auto axis = static_cast<std::size_t>(at - step.first.begin());
            if (value.shape[axis] == 1 && result.shape[r] != 1) {
                summed.push_back(r);
            } else {
                order.push_back(axis);
            }
        }
        TensorInfo reduced = sum_over(g, summed);
        if (!std::is_sorted(order.begin(), order.end())) {
            // Back into the operand's axis order.
            std::vector<std::size_t> by_axis(order.size());
            std::iota(by_axis.begin(), by_axis.end(), std::size_t{0});
            std::sort(by_axis.begin(), by_axis.end(), [&](std::size_t x, std::size_t y) {
                return order[x] < order[y];
            });
            Dims permutation;
            Dims shape;
            permutation.reserve(by_axis.size());
            shape.reserve(by_axis.size());
            for (const std::size_t position : by_axis) {
                permutation.push_back(static_cast<std::int64_t>(position));
                shape.push_back(reduced.shape[position]);
            }
            reduced = {inner_.transpose(reduced, permutation, shape), shape, reduced.dtype};
        }
        if (reduced.shape != value.shape) {
            reduced = {inner_.reshape(reduced, value.shape), value.shape, reduced.dtype};
        }
        accumulate(value, reduced);
        return;
    }
    case Step::Kind::Contract: {
        const TensorInfo& lhs = ops[0];
        const TensorInfo& rhs = ops[1];
        accumulate(lhs, product(g, step.third, rhs, step.second, step.first, lhs.shape, lhs.dtype));
        accumulate(rhs, product(g, step.third, lhs, step.first, step.second, rhs.shape, rhs.dtype));
        return;
    }
    case Step::Kind::Slice: {
        // Each element of the slice back where it came from: the positions
        // `start + stride * i` along every axis, scattered into zeros.
        const TensorInfo& value = ops.front();
        const Dims& shape = result.shape;
        std::vector<TensorInfo> columns;
        Dims column_shape = shape;
        column_shape.push_back(1);
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            const Dims line{shape[axis]};
            TensorInfo positions{inner_.iota(shape[axis]), line, ScalarKind::I64};
            positions = {
                inner_.elementwise(
                    Elementwise::Mul,
                    {positions,
                     widen({inner_.constant(Literal::of_integer(step.third[axis]), ScalarKind::I64),
                            {},
                            ScalarKind::I64},
                           line)},
                    line,
                    ScalarKind::I64),
                line,
                ScalarKind::I64};
            positions = {
                inner_.elementwise(
                    Elementwise::Add,
                    {positions,
                     widen({inner_.constant(Literal::of_integer(step.first[axis]), ScalarKind::I64),
                            {},
                            ScalarKind::I64},
                           line)},
                    line,
                    ScalarKind::I64),
                line,
                ScalarKind::I64};
            columns.push_back(
                {inner_.broadcast(positions, {static_cast<std::int64_t>(axis)}, column_shape),
                 column_shape,
                 ScalarKind::I64});
        }
        Dims index_shape = shape;
        index_shape.push_back(static_cast<std::int64_t>(shape.size()));
        const TensorInfo indices{
            shape.empty()
                ? inner_.reshape(columns.front(), index_shape)
                : inner_.concat(columns, static_cast<std::int64_t>(shape.size()), index_shape),
            index_shape,
            ScalarKind::I64};
        const auto scattered = inner_.scatter_add(value.shape, value.dtype, indices, g);
        if (!scattered) {
            throw GradientError("this format cannot add values at indices (a slice's gradient)");
        }
        accumulate(value, {*scattered, value.shape, value.dtype});
        return;
    }
    case Step::Kind::Concat: {
        const auto axis = static_cast<std::size_t>(step.axis);
        std::int64_t offset = 0;
        for (const TensorInfo& part : ops) {
            Dims starts(result.shape.size(), 0);
            Dims limits = result.shape;
            const Dims strides(result.shape.size(), 1);
            starts[axis] = offset;
            limits[axis] = offset + part.shape[axis];
            offset = limits[axis];
            accumulate(part,
                       {inner_.slice(g, starts, limits, strides, part.shape), part.shape, g.dtype});
        }
        return;
    }
    case Step::Kind::Loop:
        return; // see `propagate_loop`
    case Step::Kind::Cumsum: {
        // Each element reaches the sums from its own position on: the
        // gradient's suffix sums, its total less the running sum before.
        const auto axis = static_cast<std::size_t>(step.axis);
        const TensorInfo total = sum_over(g, {axis});
        Dims kept;
        for (std::size_t i = 0; i < g.shape.size(); ++i) {
            if (i != axis) {
                kept.push_back(static_cast<std::int64_t>(i));
            }
        }
        const TensorInfo spread{inner_.broadcast(total, kept, g.shape), g.shape, g.dtype};
        const TensorInfo running{inner_.cumsum(g, step.axis), g.shape, g.dtype};
        accumulate(ops.front(),
                   apply(Elementwise::Add, {apply(Elementwise::Sub, {spread, running}), g}));
        return;
    }
    case Step::Kind::Custom: {
        // `pullbacks_` grows only while recording: the reference stays valid.
        const std::vector<std::optional<std::string>> gradients =
            pullbacks_[step.custom](ops, result, g);
        for (std::size_t i = 0; i < ops.size() && i < gradients.size(); ++i) {
            if (const std::optional<std::string>& gradient = gradients[i]) {
                accumulate(ops[i], {*gradient, ops[i].shape, ops[i].dtype});
            }
        }
        return;
    }
    case Step::Kind::Gather: {
        const TensorInfo& source = ops[0];
        if (!differentiable(source)) {
            return;
        }
        const auto scattered = inner_.scatter_add(source.shape, source.dtype, ops[1], g);
        if (!scattered) {
            throw GradientError("this format cannot add values at indices (a gather's gradient)");
        }
        accumulate(source, {*scattered, source.shape, source.dtype});
        return;
    }
    case Step::Kind::Reduce: {
        // The result's axes lead the operand's; the reduced ones trail.
        const TensorInfo& body = ops.front();
        Dims leading(result.shape.size());
        std::iota(leading.begin(), leading.end(), std::int64_t{0});
        const TensorInfo spread{inner_.broadcast(g, leading, body.shape), body.shape, g.dtype};
        switch (step.reduction) {
        case Reduction::Sum:
            accumulate(body, spread);
            return;
        case Reduction::Max:
        case Reduction::Min: {
            const TensorInfo reached{
                inner_.broadcast(result, leading, body.shape), body.shape, result.dtype};
            const TensorInfo is_extreme{
                inner_.compare(ir::CompareKind::Eq, body, reached, body.shape),
                body.shape,
                ScalarKind::Bool};
            accumulate(body, choose(is_extreme, spread, full(0.0, body.shape, body.dtype)));
            return;
        }
        case Reduction::Prod: {
            const TensorInfo whole{
                inner_.broadcast(result, leading, body.shape), body.shape, result.dtype};
            accumulate(body,
                       apply(Elementwise::Div, {apply(Elementwise::Mul, {spread, whole}), body}));
            return;
        }
        default:
            return; // `any`, `all`
        }
    }
    }
}

// ----------------------------------------------------------------- helpers

TensorInfo GradientTarget::full(double value, const Dims& shape, ScalarKind dtype) {
    const TensorInfo scalar{inner_.constant(Literal::of_real(value), dtype), {}, dtype};
    return widen(scalar, shape);
}

TensorInfo GradientTarget::widen(const TensorInfo& value, const Dims& shape) {
    if (value.shape == shape) {
        return value;
    }
    // Right-aligned, as elementwise operands broadcast.
    const std::size_t lead = shape.size() - value.shape.size();
    Dims dims(value.shape.size());
    std::iota(dims.begin(), dims.end(), static_cast<std::int64_t>(lead));
    return {inner_.broadcast(value, dims, shape), shape, value.dtype};
}

TensorInfo GradientTarget::apply(Elementwise kind, const std::vector<TensorInfo>& operands) {
    Dims shape;
    for (const TensorInfo& operand : operands) {
        if (operand.shape.size() > shape.size()) {
            shape = operand.shape;
        }
    }
    std::vector<TensorInfo> wide;
    wide.reserve(operands.size());
    for (const TensorInfo& operand : operands) {
        wide.push_back(widen(operand, shape));
    }
    const ScalarKind dtype = operands.front().dtype;
    return {inner_.elementwise(kind, wide, shape, dtype), shape, dtype};
}

TensorInfo GradientTarget::choose(const TensorInfo& condition,
                                  const TensorInfo& on_true,
                                  const TensorInfo& on_false) {
    const Dims& shape =
        on_true.shape.size() >= on_false.shape.size() ? on_true.shape : on_false.shape;
    return {inner_.select(widen(condition, shape),
                          widen(on_true, shape),
                          widen(on_false, shape),
                          shape,
                          on_true.dtype),
            shape,
            on_true.dtype};
}

TensorInfo GradientTarget::unbroadcast(const TensorInfo& grad, const TensorInfo& operand) {
    if (grad.shape == operand.shape) {
        return grad;
    }
    const std::size_t lead = grad.shape.size() - operand.shape.size();
    std::vector<std::size_t> axes;
    axes.reserve(grad.shape.size());
    for (std::size_t i = 0; i < lead; ++i) {
        axes.push_back(i);
    }
    for (std::size_t j = 0; j < operand.shape.size(); ++j) {
        if (operand.shape[j] == 1 && grad.shape[lead + j] != 1) {
            axes.push_back(lead + j);
        }
    }
    TensorInfo summed = sum_over(grad, axes);
    if (summed.shape != operand.shape) {
        summed = {inner_.reshape(summed, operand.shape), operand.shape, summed.dtype};
    }
    return summed;
}

TensorInfo GradientTarget::sum_over(const TensorInfo& value, const std::vector<std::size_t>& axes) {
    if (axes.empty()) {
        return value;
    }
    Dims permutation;
    Dims kept_shape;
    for (std::size_t axis = 0; axis < value.shape.size(); ++axis) {
        if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
            permutation.push_back(static_cast<std::int64_t>(axis));
            kept_shape.push_back(value.shape[axis]);
        }
    }
    const std::size_t kept = permutation.size();
    Dims moved_shape = kept_shape;
    for (const std::size_t axis : axes) {
        permutation.push_back(static_cast<std::int64_t>(axis));
        moved_shape.push_back(value.shape[axis]);
    }
    TensorInfo moved = value;
    if (!std::is_sorted(permutation.begin(), permutation.end())) {
        moved = {inner_.transpose(value, permutation, moved_shape), moved_shape, value.dtype};
    }
    Dims trailing(axes.size());
    std::iota(trailing.begin(), trailing.end(), static_cast<std::int64_t>(kept));
    return {inner_.reduce(Reduction::Sum, moved, trailing, kept_shape), kept_shape, value.dtype};
}

TensorInfo GradientTarget::product(const TensorInfo& lhs,
                                   const Dims& lhs_axes,
                                   const TensorInfo& rhs,
                                   const Dims& rhs_axes,
                                   const Dims& out_axes,
                                   const Dims& shape,
                                   ScalarKind dtype) {
    // The result's axes either operand has; any other is a broadcast after.
    Dims present_axes;
    Dims present_shape;
    Dims positions;
    for (std::size_t i = 0; i < out_axes.size(); ++i) {
        const auto axis = out_axes[i];
        const bool is_present =
            std::find(lhs_axes.begin(), lhs_axes.end(), axis) != lhs_axes.end() ||
            std::find(rhs_axes.begin(), rhs_axes.end(), axis) != rhs_axes.end();
        if (is_present) {
            present_axes.push_back(axis);
            present_shape.push_back(shape[i]);
            positions.push_back(static_cast<std::int64_t>(i));
        }
    }
    TensorInfo out;
    if (const auto contracted =
            inner_.contract(lhs, lhs_axes, rhs, rhs_axes, present_axes, present_shape, dtype)) {
        out = {*contracted, present_shape, dtype};
    } else {
        // Every axis of both on one grid, multiplied and summed down.
        Dims grid = lhs_axes;
        Dims grid_shape = lhs.shape;
        for (std::size_t i = 0; i < rhs_axes.size(); ++i) {
            if (std::find(grid.begin(), grid.end(), rhs_axes[i]) == grid.end()) {
                grid.push_back(rhs_axes[i]);
                grid_shape.push_back(rhs.shape[i]);
            }
        }
        const auto place = [&](const TensorInfo& operand, const Dims& axes) {
            Dims dims;
            dims.reserve(axes.size());
            for (const auto axis : axes) {
                dims.push_back(std::find(grid.begin(), grid.end(), axis) - grid.begin());
            }
            return TensorInfo{inner_.broadcast(operand, dims, grid_shape), grid_shape, dtype};
        };
        const TensorInfo multiplied =
            apply(Elementwise::Mul, {place(lhs, lhs_axes), place(rhs, rhs_axes)});
        std::vector<std::size_t> summed;
        Dims kept;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (std::find(present_axes.begin(), present_axes.end(), grid[i]) ==
                present_axes.end()) {
                summed.push_back(i);
            } else {
                kept.push_back(grid[i]);
            }
        }
        out = sum_over(multiplied, summed);
        if (kept != present_axes) {
            Dims permutation;
            permutation.reserve(present_axes.size());
            for (const auto axis : present_axes) {
                permutation.push_back(std::find(kept.begin(), kept.end(), axis) - kept.begin());
            }
            out = {inner_.transpose(out, permutation, present_shape), present_shape, dtype};
        }
    }
    if (present_shape == shape) {
        return out;
    }
    return {inner_.broadcast(out, positions, shape), shape, dtype};
}

} // namespace linnet::backend
