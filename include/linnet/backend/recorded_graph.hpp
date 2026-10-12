#pragma once

#include "linnet/backend/graph_export.hpp"
#include "linnet/sema/types.hpp"

#include <cstdint>
#include <deque>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace linnet::backend {

// One entry's graph as a target saw it: every call the exporter made to the
// target, in order, with the values they made. `RecordingTarget` records it
// while passing each call on to the target, so the target's own choices
// (which library calls it takes, whether it fuses a contraction) are in it;
// `replay_graph` makes the same calls again on a fresh target of the same
// kind, which then prints the same source. Calls nest between `begin_call`
// and `end_call`; a callback a target runs (an op's backward pass, a
// kernel's fallback body) is a region of the node that passed it.

inline constexpr std::uint32_t no_value = 0xFFFFFFFFU;

struct RecordedValue {
    std::string name; // the target's name for it
    Dims shape;
    sema::ScalarKind dtype = sema::ScalarKind::F32;
};

// A tensor argument: a value, with the shape and dtype the call gave it;
// `value` is `no_value` for an absent optional.
struct RecordedOperand {
    std::uint32_t value = no_value;
    Dims shape;
    sema::ScalarKind dtype = sema::ScalarKind::F32;
};

using RecordedAttr = std::variant<std::int64_t,
                                  std::string,
                                  Dims,
                                  std::vector<Dims>,
                                  std::vector<std::string>,
                                  std::map<std::string, std::int64_t>,
                                  Literal,
                                  KernelProgram>;

struct RecordedNode;

// One run of a callback: the values it was given, what it did, and what it
// returned (nothing for an argument without a gradient).
struct RecordedRegion {
    std::string role; // `pullback`, or `body` for a kernel's fallback
    std::vector<std::uint32_t> arguments;
    std::deque<RecordedNode> nodes;
    std::vector<std::uint32_t> results;
};

struct RecordedNode {
    std::string op; // the target method: `elementwise`, `native_call`, `begin_call`, ...
    std::vector<std::pair<std::string, RecordedAttr>> attrs;
    std::vector<RecordedOperand> operands;
    std::vector<std::uint32_t> results;
    bool declined = false; // a call that may decline did
    std::deque<RecordedRegion> regions;

    const RecordedAttr* attr(std::string_view key) const;
};

struct RecordedGraph {
    std::string target; // `torch`, `jax`
    std::vector<RecordedValue> values;
    std::deque<RecordedNode> nodes;
};

// A target that records every call it gets and passes it on to `inner`.
class RecordingTarget final : public GraphTarget {
public:
    RecordingTarget(GraphTarget& inner, std::string target);

    const RecordedGraph& graph() const { return graph_; }

    std::string input(const std::string& name, const Dims& shape, sema::ScalarKind dtype) override;
    std::string
    parameter(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
    std::string state(const std::string& path, const Dims& shape, sema::ScalarKind dtype) override;
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
    bool broadcasts_elementwise() const override { return inner_.broadcasts_elementwise(); }
    std::optional<std::string> contract(const TensorInfo& lhs,
                                        const Dims& lhs_axes,
                                        const TensorInfo& rhs,
                                        const Dims& rhs_axes,
                                        const Dims& out_axes,
                                        const Dims& shape,
                                        sema::ScalarKind dtype) override;
    bool supports_placement() const override { return inner_.supports_placement(); }
    void enable_placement(int slots) override;
    void set_slot(int slot) override;
    std::string transfer(const TensorInfo& value, int slot) override;
    void release(const std::vector<std::string>& names) override;
    bool supports_fully_shard() const override { return inner_.supports_fully_shard(); }
    std::string gather(const TensorInfo& value) override;
    bool supports_remat() const override { return inner_.supports_remat(); }
    void begin_remat() override;
    void end_remat() override;
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
    bool wants_calls() const override { return true; }
    void begin_call(const CallInfo& call) override;
    void end_call(const std::vector<TensorInfo>& results) override;
    std::optional<std::string> native_call(const std::string& implementation,
                                           const std::vector<std::optional<TensorInfo>>& operands,
                                           const Dims& shape,
                                           sema::ScalarKind dtype) override;
    std::optional<std::string>
    update_row(const TensorInfo& stack, const TensorInfo& index, const TensorInfo& value) override;
    bool supports_while() const override { return inner_.supports_while(); }
    bool while_needs_initial_condition() const override {
        return inner_.while_needs_initial_condition();
    }
    void while_initial_condition(const TensorInfo& predicate) override;
    std::vector<std::string> begin_while(const std::vector<TensorInfo>& initial) override;
    std::vector<std::string> while_condition(const TensorInfo& predicate) override;
    bool while_needs_trailing_condition() const override {
        return inner_.while_needs_trailing_condition();
    }
    void while_trailing_condition(const TensorInfo& predicate) override;
    std::vector<std::string> end_while(const std::vector<TensorInfo>& next) override;
    std::unique_ptr<KernelTarget> kernel_target() const override;
    std::optional<std::vector<std::string>> launch_kernel(const KernelLaunch& launch) override;
    std::vector<std::string>
    begin_custom_gradient(const std::vector<TensorInfo>& arguments) override;
    std::string end_custom_gradient(const std::vector<TensorInfo>& arguments,
                                    const TensorInfo& result,
                                    const Pullback& pullback) override;
    std::string cumsum(const TensorInfo& value, std::int64_t axis) override;
    std::optional<std::string> scatter_add(const Dims& shape,
                                           sema::ScalarKind dtype,
                                           const TensorInfo& indices,
                                           const TensorInfo& values) override;
    bool supports_counted() const override { return inner_.supports_counted(); }
    std::vector<std::string> begin_counted(std::int64_t start,
                                           std::int64_t stop,
                                           const std::vector<TensorInfo>& initial) override;
    std::vector<std::string> end_counted(const std::vector<TensorInfo>& next) override;
    std::string finish(const std::vector<TensorInfo>& results,
                       const std::vector<std::pair<std::string, TensorInfo>>& states,
                       const std::string& module_path,
                       const std::string& block_name,
                       const std::string& entry_name) override;

private:
    RecordedNode& add(std::string op);
    std::uint32_t define(const std::string& name, const Dims& shape, sema::ScalarKind dtype);
    RecordedOperand operand(const TensorInfo& value);
    RecordedOperand operand(const std::optional<TensorInfo>& value);
    std::uint32_t
    result(RecordedNode& node, const std::string& name, const Dims& shape, sema::ScalarKind dtype);
    void results(RecordedNode& node,
                 const std::vector<std::string>& names,
                 const std::vector<TensorInfo>& like);
    // A new region of `node`, of `role`, whose arguments are `arguments`; the
    // nodes recorded until `close_region` go into it.
    RecordedRegion&
    open_region(RecordedNode& node, std::string role, const std::vector<TensorInfo>& arguments);
    void close_region();

    GraphTarget& inner_;
    RecordedGraph graph_;
    std::map<std::string, std::uint32_t> ids_;
    std::vector<std::deque<RecordedNode>*> open_;
    std::vector<std::vector<TensorInfo>> loops_; // each open `while` loop's values
};

// The calls of `graph` made again on `target`, a fresh target of the kind
// that recorded it: what its `finish` returns. Fails where the target
// answers differently than the recorded one did.
std::expected<std::string, std::string> replay_graph(const RecordedGraph& graph,
                                                     GraphTarget& target);

// `graph` as JSON (`linnet graph`, docs/graph-format.md).
std::string recorded_graph_json(const RecordedGraph& graph);

} // namespace linnet::backend
