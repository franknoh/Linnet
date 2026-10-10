#include "linnet/backend/graph_export.hpp"

#include "linnet/backend/gradient.hpp"
#include "linnet/backend/kernel.hpp"
#include "linnet/support/text.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace linnet::backend {

namespace {

using namespace sema;

// A capability failure; caught at the boundary and returned as text.
struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) {
    throw Unsupported(message);
}

// A value during evaluation. Tensors (scalars are rank 0) live in the
// target's document; the other kinds are resolved statically while inlining.
struct Val {
    enum class Kind : std::uint8_t { Tensor, None, Some, Tuple, Block, Array, Index, Pack, Enum };
    Kind kind = Kind::Tensor;
    std::string name; // Tensor: the target's name for it
    Dims shape;       // Tensor
    ScalarKind dtype = ScalarKind::F32;
    std::size_t grid_rank = 0;     // Tensor: leading axes that are grid axes
    std::vector<Val> elements;     // Tuple, Array; Some holds one
    std::string path;              // Block: member path prefix, `layers.0.`; Enum: variant
    EntityId block = no_entity;    // Block: declaration
    Substitution subst;            // Block: its generic bindings
    std::vector<std::size_t> axes; // Index (one), Pack (several): grid axes
    int slot = -2;                 // Tensor, with placement: see `unplaced` and `host`
};

// A tensor made before placement assigned it a slot runs on slot 0; a
// parameter of an offloaded block lives on the host and runs nowhere until
// it is transferred.
constexpr int unplaced = -2;
constexpr int host = -1;

TensorInfo info(const Val& value) {
    return {value.name, value.shape, value.dtype};
}

class Evaluator {
public:
    Evaluator(ir::Module& module, const GraphExportOptions& options, GraphTarget& target)
        : module_(module), model_(module.model()), types_(module.types()), options_(options),
          target_(target) {
        for (const ir::Function& function : module_.functions()) {
            functions_[function.name] = &function;
        }
    }

    std::string run() {
        // No root block: the entry is a module-level one, a function of its
        // inputs alone, with no `self`, parameters, or state.
        const std::optional<EntityId> root = find_root();
        const ir::Function& entry = find_entry(root);
        const Substitution root_subst = root ? root_bindings(*root) : Substitution{};

        if (!options_.remat.empty()) {
            if (!root) {
                fail("recomputing calls a model's blocks; a function has none");
            }
            if (!target_.supports_remat()) {
                fail("this target cannot recompute blocks; recomputing is for `jax`");
            }
        }
        if (!options_.fully_shard.empty()) {
            if (!root) {
                fail("sharding splits a model's parameters; a function has none");
            }
            if (!target_.supports_fully_shard()) {
                fail("this target cannot gather sharded parameters; sharding is for `torch` and "
                     "`jax`");
            }
            if (!options_.placement.empty() || !options_.offload.empty()) {
                fail("sharded blocks run on each process's one device: not with placement or "
                     "offload");
            }
        } else if (placing()) {
            if (!root) {
                fail("placement spreads a model's blocks over devices; a function has none");
            }
            if (!target_.supports_placement()) {
                fail("this target cannot place blocks on devices; placement is for `torch`");
            }
            int slots = 1;
            for (const auto& [prefix, slot] : options_.placement) {
                if (slot < 0) {
                    fail("placement slot for `" + prefix + "` must not be negative");
                }
                slots = std::max(slots, slot + 1);
            }
            target_.enable_placement(slots);
        }

        Frame frame;
        frame.subst = entry_bindings(entry, root_subst);
        const ir::Block& body = module_.block(module_.region(entry.body).blocks.front());
        std::size_t first_input = 0;
        if (root) {
            Val self;
            self.kind = Val::Kind::Block;
            self.block = *root;
            self.subst = root_subst;
            frame.values[body.arguments.front()] = self;
            first_input = 1;
        }
        for (std::size_t i = first_input; i < body.arguments.size(); ++i) {
            const ir::ValueId argument = body.arguments[i];
            Val value = tensor_value(module_.value(argument).type, frame.subst);
            value.name = target_.input(module_.value(argument).name, value.shape, value.dtype);
            frame.values[argument] = value;
        }
        if (root) {
            collect_parameters(*root, root_subst, "");
        }
        frames_.push_back(std::move(frame));
        const std::vector<Val> results = run_block(body);
        frames_.pop_back();
        if (results.size() != 1) {
            fail("the entry must return one value");
        }
        std::vector<TensorInfo> outputs;
        const Val& result = results.front();
        for (const Val& element :
             result.kind == Val::Kind::Tuple ? result.elements : std::vector<Val>{result}) {
            if (element.kind != Val::Kind::Tensor) {
                fail("the entry must return a tensor or a tuple of tensors");
            }
            outputs.push_back(info(element));
        }
        std::vector<std::pair<std::string, TensorInfo>> final_states;
        final_states.reserve(written_states_.size());
        for (const std::string& path : written_states_) {
            final_states.emplace_back(path, info(states_.at(path)));
        }
        return target_.finish(outputs,
                              final_states,
                              model_.module_paths.at(options_.root_module),
                              root ? std::string(model_.entities[*root].name) : std::string(),
                              std::string(model_.entities[entry.entity].name));
    }

private:
    // Frames live in a deque so that references into an outer frame's
    // values survive the frames a call pushes.
    struct Frame {
        std::map<ir::ValueId, Val> values;
        Substitution subst;
    };

    // Placement state: the slot stack (calls into placed blocks push), the
    // transfers already made per loop scope, and per offloaded block the
    // parameter copies to release when it returns.
    std::vector<int> slots_{0};
    std::vector<std::map<std::pair<std::string, int>, std::string>> moved_{1};
    std::vector<std::vector<std::string>> releases_;
    std::vector<std::vector<std::pair<std::string, int>>> released_keys_;

    // ------------------------------------------------------------- roots

    // The root block, or nothing for a module-level entry: one `--entry`
    // names without `--root`, or the only one of a file whose blocks have
    // no entries.
    std::optional<EntityId> find_root() const {
        if (options_.root.empty()) {
            if (!options_.entry.empty() && is_module_entry(options_.entry)) {
                for (EntityId id = 0; id < model_.entities.size(); ++id) {
                    const Entity& entity = model_.entities[id];
                    if (entity.kind == EntityKind::Block && entity.parent == no_entity &&
                        entity.module == options_.root_module &&
                        has_entry_named(id, options_.entry)) {
                        fail("both the module and block `" + std::string(entity.name) +
                             "` have an entry `" + options_.entry + "`: pass --root " +
                             std::string(entity.name) + " for the block's, or rename one");
                    }
                }
                return std::nullopt;
            }
            if (!any_block_entry() && is_module_entry("")) {
                return std::nullopt;
            }
        }
        const auto root = find_root_block(module_, options_.root_module, options_.root);
        if (!root) {
            fail(root.error());
        }
        return *root;
    }

    bool has_entry_named(EntityId owner, std::string_view name) const {
        return declares_entry(module_, options_.root_module, owner, name);
    }

    bool is_module_entry(std::string_view name) const { return has_entry_named(no_entity, name); }

    bool any_block_entry() const {
        for (const ir::Function& function : module_.functions()) {
            const Entity& entity = model_.entities[function.entity];
            if (function.is_entry && entity.parent != no_entity &&
                entity.module == options_.root_module) {
                return true;
            }
        }
        return false;
    }

    const ir::Function& find_entry(std::optional<EntityId> root) const {
        const EntityId owner = root.value_or(no_entity);
        const char* const where = root ? "the block" : "the module";
        const ir::Function* found = nullptr;
        for (const ir::Function& function : module_.functions()) {
            const Entity& entity = model_.entities[function.entity];
            if (!function.is_entry || entity.parent != owner ||
                entity.module != options_.root_module) {
                continue;
            }
            if (!options_.entry.empty() ? entity.name == options_.entry : found == nullptr) {
                found = &function;
            } else if (options_.entry.empty()) {
                fail(std::string(where) + " has several entries; name one with --entry");
            }
        }
        if (found == nullptr) {
            fail(options_.entry.empty() ? std::string(where) + " has no entry"
                                        : "no entry named `" + options_.entry + "`");
        }
        return *found;
    }

    // --------------------------------------------------------- bindings

    void bind_generics(const std::vector<GenericInfo>& generics,
                       Substitution& subst,
                       const char* owner) {
        for (const GenericInfo& generic : generics) {
            const auto bound = options_.bindings.find(std::string(generic.name));
            const std::string* text = bound == options_.bindings.end() ? nullptr : &bound->second;
            const GenericValue fallback = generic.default_value.value_or(GenericValue{});
            if (text == nullptr && !generic.default_value) {
                fail(std::string(owner) + " generic `" + std::string(generic.name) +
                     "` needs a value: pass --bind " + std::string(generic.name) + "=...");
            }
            switch (generic.kind) {
            case GenericKind::Dim: {
                std::int64_t value = 0;
                if (text != nullptr) {
                    const auto* end = text->data() + text->size();
                    if (std::from_chars(text->data(), end, value).ptr != end) {
                        fail("`" + *text + "` is not an integer for `" + std::string(generic.name) +
                             "`");
                    }
                } else {
                    value = fallback.dim.constant().value_or(0);
                }
                subst.dims[generic.symbol] = shape::Poly(value);
                break;
            }
            case GenericKind::Pack: {
                // The pack's dimensions, comma-separated: `--bind S=2,3`, or
                // nothing for an empty pack (`--bind S=`).
                if (text == nullptr) {
                    fail(std::string(owner) + " shape pack `" + std::string(generic.name) +
                         "` needs dimensions: pass --bind " + std::string(generic.name) +
                         "=2,3 (or nothing after `=` for none)");
                }
                Shape pack;
                std::string_view rest = *text;
                while (!rest.empty()) {
                    const std::size_t comma = rest.find(',');
                    const std::string part(rest.substr(0, comma));
                    std::int64_t value = 0;
                    const char* end = part.c_str() + part.size();
                    if (part.empty() || std::from_chars(part.c_str(), end, value).ptr != end ||
                        value < 0) {
                        fail("`" + *text + "` is not a list of dimensions for `" +
                             std::string(generic.name) + "`");
                    }
                    pack.push_back(ShapeElem::of(shape::Poly(value)));
                    rest = comma == std::string_view::npos ? std::string_view{}
                                                           : rest.substr(comma + 1);
                }
                subst.packs[generic.symbol] = std::move(pack);
                break;
            }
            case GenericKind::DType: {
                DType dtype;
                if (text != nullptr) {
                    const auto kind = scalar_from_name(*text);
                    if (!kind) {
                        fail("`" + *text + "` is not a dtype for `" + std::string(generic.name) +
                             "`");
                    }
                    dtype = DType::of(*kind);
                } else {
                    dtype = fallback.dtype;
                }
                subst.dtypes[generic.dtype_var] = dtype;
                break;
            }
            }
        }
    }

    Substitution root_bindings(EntityId root) {
        Substitution subst;
        bind_generics(model_.decls.at(root).generics, subst, "root block");
        check_constraints(root, subst, "root block");
        return subst;
    }

    Substitution entry_bindings(const ir::Function& entry, const Substitution& root_subst) {
        Substitution subst = root_subst;
        bind_generics(entry.generics, subst, "entry");
        if (entry.entity != sema::no_entity && model_.decls.contains(entry.entity)) {
            check_constraints(entry.entity, subst, "entry");
        }
        return subst;
    }

    // The declaration's `where` clause under the bound values. The checker
    // proves every constraint at each call inside the program; the values
    // a caller binds from outside are checked here, before anything is
    // built from shapes the declaration rules out (`H % Heads == 0` with
    // `H = 5, Heads = 2` would split 5 features into 2 heads of 2).
    void check_constraints(EntityId decl, const Substitution& subst, const char* owner) {
        for (const ConstraintInfo& constraint : model_.decls.at(decl).constraints) {
            const auto lhs = types_.substitute(constraint.lhs, subst).constant();
            const auto rhs = types_.substitute(constraint.rhs, subst).constant();
            if (!lhs || !rhs) {
                continue;
            }
            bool holds = true;
            const char* symbol = "==";
            switch (constraint.relation) {
            case shape::Relation::Equal:
                holds = *lhs == *rhs;
                break;
            case shape::Relation::NotEqual:
                holds = *lhs != *rhs;
                symbol = "!=";
                break;
            case shape::Relation::Less:
                holds = *lhs < *rhs;
                symbol = "<";
                break;
            case shape::Relation::LessEqual:
                holds = *lhs <= *rhs;
                symbol = "<=";
                break;
            case shape::Relation::Greater:
                holds = *lhs > *rhs;
                symbol = ">";
                break;
            case shape::Relation::GreaterEqual:
                holds = *lhs >= *rhs;
                symbol = ">=";
                break;
            }
            if (!holds) {
                fail(std::string(owner) + " constraint `" + model_.dims.to_string(constraint.lhs) +
                     " " + symbol + " " + model_.dims.to_string(constraint.rhs) +
                     "` does not hold for the bound " + "values (" + std::to_string(*lhs) + " " +
                     symbol + " " + std::to_string(*rhs) + " is false)");
            }
        }
    }

    // A `while` loop: the target's loop form over the carried values, with
    // the condition and body regions emitted into it.
    void run_while(const ir::Operation& op) {
        if (!grid_.empty()) {
            fail("a `while` loop inside index notation is not supported");
        }
        std::vector<Val> own;
        own.reserve(op.operands.size());
        for (const ir::ValueId id : op.operands) {
            own.push_back(whole_tensor(value(id), "while"));
        }
        const std::vector<Val> finals = run_loop(
            own,
            [&](const std::vector<Val>& values) {
                const std::vector<Val> results = run_region(op.regions.front(), values);
                if (results.size() != 1 || results.front().kind != Val::Kind::Tensor) {
                    fail("a `while` condition must be one scalar");
                }
                return info(results.front());
            },
            [&](const std::vector<Val>& values) {
                std::vector<Val> next = run_region(op.regions.back(), values);
                if (next.size() != values.size()) {
                    fail("internal: a `while` body yielded " + std::to_string(next.size()) +
                         " values");
                }
                return next;
            });
        for (std::size_t i = 0; i < op.results.size(); ++i) {
            frame().values[op.results[i]] = finals[i];
        }
    }

    // A `for` loop: a `while` over its index, its carried values and, for a
    // loop with a value, the stacked value's parts, zeros to begin with and
    // filled a row an iteration.
    void run_for(const ir::Operation& op) {
        if (!grid_.empty()) {
            fail("a `for` loop inside index notation is not supported");
        }
        const std::int64_t start = range_bound(op.operands[0]);
        const std::int64_t stop = range_bound(op.operands[1]);
        const std::size_t carried = op.operands.size() - 2;
        const bool yields = op.results.size() > carried;
        // A counted loop keeps its own index; a `while` carries it first.
        const bool counts = target_.supports_counted();
        const std::size_t first = counts ? 0 : 1;
        std::vector<Val> own;
        own.reserve(op.operands.size() + op.results.size());
        if (!counts) {
            own.push_back(constant_int(start, ScalarKind::I64));
        }
        for (std::size_t i = 2; i < op.operands.size(); ++i) {
            own.push_back(whole_tensor(value(op.operands[i]), "for"));
        }
        bool is_tuple = false;
        if (yields) {
            const TypeId stacked =
                types_.substitute(module_.value(op.results.back()).type, frame().subst);
            const TypeData data = types_.get(stacked);
            is_tuple = data.kind == TypeKind::Tuple;
            for (const TypeId part : is_tuple ? data.elements : std::vector<TypeId>{stacked}) {
                const Val shape = tensor_value(part, frame().subst);
                own.push_back(zeros(shape.shape, shape.dtype));
            }
        }
        const std::vector<Val> finals = run_loop(
            own,
            [&](const std::vector<Val>& values) {
                const Val bound = constant_int(stop, ScalarKind::I64);
                return TensorInfo{
                    target_.compare(ir::CompareKind::Lt, info(values[0]), info(bound), {}),
                    {},
                    ScalarKind::Bool};
            },
            [&](const std::vector<Val>& values) {
                const Val& index = values[0];
                const std::vector<Val> arguments(
                    values.begin(), values.begin() + static_cast<std::ptrdiff_t>(1 + carried));
                const std::vector<Val> outputs = run_region(op.regions.front(), arguments);
                if (outputs.size() != carried + (yields ? 1 : 0)) {
                    fail("internal: a `for` body yielded " + std::to_string(outputs.size()) +
                         " values");
                }
                std::vector<Val> next{elementwise(Elementwise::Add,
                                                  {index, constant_int(1, ScalarKind::I64)},
                                                  {},
                                                  ScalarKind::I64)};
                next.reserve(values.size());
                next.insert(next.end(),
                            outputs.begin(),
                            outputs.begin() + static_cast<std::ptrdiff_t>(carried));
                if (yields) {
                    const Val row = start == 0
                                        ? index
                                        : elementwise(Elementwise::Sub,
                                                      {index, constant_int(start, ScalarKind::I64)},
                                                      {},
                                                      ScalarKind::I64);
                    const Val& element = outputs.back();
                    const std::vector<Val> parts =
                        is_tuple ? element.elements : std::vector<Val>{element};
                    for (std::size_t k = 0; k < parts.size(); ++k) {
                        next.push_back(update_row(
                            values[1 + carried + k], row, whole_tensor(parts[k], "for")));
                    }
                }
                return next;
            },
            counts ? std::optional(Counted{start, stop}) : std::nullopt);
        for (std::size_t i = 0; i < carried; ++i) {
            frame().values[op.results[i]] = finals[first + i];
        }
        if (yields) {
            Val stacked;
            if (is_tuple) {
                stacked.kind = Val::Kind::Tuple;
                stacked.elements.assign(
                    finals.begin() + static_cast<std::ptrdiff_t>(first + carried), finals.end());
            } else {
                stacked = finals.back();
            }
            frame().values[op.results.back()] = stacked;
        }
    }

    // A tensor a loop carries: whole, outside index notation.
    Val whole_tensor(const Val& carried_value, const char* loop) {
        if (carried_value.kind != Val::Kind::Tensor || carried_value.grid_rank != 0) {
            fail(std::string("a `") + loop + "` loop carries whole tensors only");
        }
        return carried_value;
    }

    Val zeros(const Dims& shape, ScalarKind dtype) {
        const Val zero = constant_int(0, dtype);
        return shape.empty() ? zero
                             : tensor(target_.broadcast(info(zero), {}, shape), shape, dtype);
    }

    // `stack` with row `index` of its leading axis replaced by `row_value`:
    // the target's own operation, or a select over every row.
    Val update_row(const Val& stack, const Val& index, const Val& row_value) {
        const Dims& shape = stack.shape;
        if (const auto name = target_.update_row(info(stack), info(index), info(row_value))) {
            return tensor(*name, shape, stack.dtype);
        }
        const Dims rows{shape.front()};
        const TensorInfo positions{target_.iota(shape.front()), rows, ScalarKind::I64};
        const TensorInfo at{target_.broadcast(info(index), {}, rows), rows, ScalarKind::I64};
        const TensorInfo is_row{
            target_.compare(ir::CompareKind::Eq, positions, at, rows), rows, ScalarKind::Bool};
        const TensorInfo mask{target_.broadcast(is_row, {0}, shape), shape, ScalarKind::Bool};
        Dims value_axes;
        value_axes.reserve(shape.size());
        for (std::size_t axis = 1; axis < shape.size(); ++axis) {
            value_axes.push_back(static_cast<std::int64_t>(axis));
        }
        const TensorInfo spread{
            target_.broadcast(info(row_value), value_axes, shape), shape, stack.dtype};
        return tensor(
            target_.select(mask, spread, info(stack), shape, stack.dtype), shape, stack.dtype);
    }

    // A range known at export time, for a target with counted loops.
    struct Counted {
        std::int64_t start = 0;
        std::int64_t stop = 0;
    };

    // The target's loop form over `own` values and every state member, so
    // writes inside the body flow out of it: `predicate` emits the condition
    // over the own values, `body` their next values. A `counted` loop is the
    // target's counted form instead: no condition, the body taking the index
    // first and returning its next value first, which is dropped. Returns
    // the final own values; the states take theirs.
    std::vector<Val> run_loop(const std::vector<Val>& own_initial,
                              const std::function<TensorInfo(const std::vector<Val>&)>& predicate,
                              const std::function<std::vector<Val>(const std::vector<Val>&)>& body,
                              std::optional<Counted> counted = std::nullopt) {
        if (!(counted ? target_.supports_counted() : target_.supports_while())) {
            fail("runtime loops are not exported to this format yet; they run in the "
                 "PyTorch interpreter");
        }
        for (const auto& [path, declared] : state_types_) {
            if (!states_.contains(path)) {
                Val value = declared;
                value.name = target_.state(path, value.shape, value.dtype);
                states_.emplace(path, value);
            }
        }
        std::vector<Val> carried = own_initial;
        std::vector<std::string> state_paths;
        for (const auto& [path, value] : states_) {
            state_paths.push_back(path);
            carried.push_back(value);
        }
        const std::size_t own = own_initial.size();
        std::vector<TensorInfo> initial;
        initial.reserve(carried.size());
        for (const Val& value : carried) {
            initial.push_back(info(value));
        }
        // Renames every carried value (and the states among them) to what a
        // region of the loop sees.
        const auto renamed = [&](const std::vector<std::string>& names) {
            if (names.size() != carried.size()) {
                fail("internal: the target named " + std::to_string(names.size()) +
                     " loop values for " + std::to_string(carried.size()));
            }
            std::vector<Val> values = carried;
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i].name = names[i];
                if (i >= own) {
                    states_[state_paths[i - own]] = values[i];
                }
            }
            return values;
        };
        const auto own_values = [own](const std::vector<Val>& values) {
            return std::vector<Val>(values.begin(),
                                    values.begin() + static_cast<std::ptrdiff_t>(own));
        };
        std::vector<Val> body_values;
        std::vector<Val> next;
        if (counted) {
            std::vector<std::string> names =
                target_.begin_counted(counted->start, counted->stop, initial);
            if (names.empty()) {
                fail("internal: the target named no index for a counted loop");
            }
            std::vector<Val> arguments{tensor(names.front(), {}, ScalarKind::I64)};
            names.erase(names.begin());
            moved_.emplace_back();
            body_values = renamed(names);
            const std::vector<Val> own_body = own_values(body_values);
            arguments.insert(arguments.end(), own_body.begin(), own_body.end());
            next = body(arguments);
            next.erase(next.begin());
        } else {
            if (target_.while_needs_initial_condition()) {
                target_.while_initial_condition(predicate(own_values(carried)));
            }
            const std::vector<Val> condition_values = renamed(target_.begin_while(initial));
            moved_.emplace_back();
            body_values = renamed(target_.while_condition(predicate(own_values(condition_values))));
            next = body(own_values(body_values));
        }
        for (std::size_t j = 0; j < state_paths.size(); ++j) {
            const Val& current = states_.at(state_paths[j]);
            // A state the body assigned left with a value other than the one
            // it entered with; it is a result of the entry from now on.
            if (current.name != body_values[own + j].name &&
                std::find(written_states_.begin(), written_states_.end(), state_paths[j]) ==
                    written_states_.end()) {
                written_states_.push_back(state_paths[j]);
            }
            next.push_back(current);
        }
        if (placing()) {
            for (std::size_t i = 0; i < next.size(); ++i) {
                const int entered_on = body_values[i].slot == unplaced ? 0 : body_values[i].slot;
                next[i] = on_slot(next[i], entered_on);
            }
        }
        if (!counted && target_.while_needs_trailing_condition()) {
            target_.while_trailing_condition(predicate(own_values(next)));
        }
        std::vector<TensorInfo> outputs;
        outputs.reserve(next.size());
        for (const Val& value : next) {
            outputs.push_back(info(value));
        }
        const std::vector<std::string> finals =
            counted ? target_.end_counted(outputs) : target_.end_while(outputs);
        moved_.pop_back();
        if (finals.size() != next.size()) {
            fail("internal: the target returned " + std::to_string(finals.size()) +
                 " loop results");
        }
        std::vector<Val> results;
        results.reserve(own);
        for (std::size_t i = 0; i < own; ++i) {
            Val value = next[i];
            value.name = finals[i];
            results.push_back(value);
        }
        for (std::size_t j = 0; j < state_paths.size(); ++j) {
            Val value = next[own + j];
            value.name = finals[own + j];
            states_[state_paths[j]] = value;
        }
        return results;
    }

    // A bound of a `static for` range: a compile-time integer, possibly
    // arithmetic on dimensions and literals.

    std::int64_t range_bound(ir::ValueId value) {
        const ir::OpId producer = module_.value(value).producer;
        if (producer == ir::no_id) {
            fail("a `static for` range bound must be a compile-time integer");
        }
        const ir::Operation& op = module_.op(producer);
        switch (op.kind) {
        case ir::OpKind::ConstInt:
            return op.attributes.integer;
        case ir::OpKind::ConstDim:
            return constant(types_.substitute(op.attributes.dim, frame().subst), "range bound");
        case ir::OpKind::Add:
            return range_bound(op.operands[0]) + range_bound(op.operands[1]);
        case ir::OpKind::Sub:
            return range_bound(op.operands[0]) - range_bound(op.operands[1]);
        case ir::OpKind::Mul:
            return range_bound(op.operands[0]) * range_bound(op.operands[1]);
        case ir::OpKind::Div: {
            const std::int64_t divisor = range_bound(op.operands[1]);
            if (divisor == 0) {
                fail("a `static for` range bound divides by zero");
            }
            return range_bound(op.operands[0]) / divisor;
        }
        default:
            fail("a `static for` range bound must be a compile-time integer");
        }
    }

    // A semantic call with a selected native implementation the target can
    // spell: whole tensors in, one tensor out, outside any grid.
    bool try_native(const ir::Operation& op) {
        const std::vector<std::string>& names = op.attributes.names;
        if (names.empty() || names.front() == "canonical decomposition" || !grid_.empty() ||
            op.results.size() != 1) {
            return false;
        }
        std::vector<std::optional<TensorInfo>> operands;
        for (const ir::ValueId id : op.operands) {
            const Val& argument = value(id);
            if (argument.kind == Val::Kind::None) {
                operands.emplace_back(std::nullopt);
            } else if (argument.kind == Val::Kind::Some &&
                       argument.elements.front().kind == Val::Kind::Tensor &&
                       argument.elements.front().grid_rank == 0) {
                operands.emplace_back(info(argument.elements.front()));
            } else if (argument.kind == Val::Kind::Tensor && argument.grid_rank == 0) {
                operands.emplace_back(info(argument));
            } else {
                return false;
            }
        }
        const TypeId type =
            types_.substitute(module_.value(op.results.front()).type, frame().subst);
        if (types_.kind(type) != TypeKind::Tensor && types_.kind(type) != TypeKind::Scalar) {
            return false;
        }
        Val result = tensor_value(type, {});
        std::map<std::string, std::int64_t> generics;
        if (const auto callee = functions_.find(op.attributes.name); callee != functions_.end()) {
            for (const sema::GenericInfo& generic : callee->second->generics) {
                const auto bound = op.attributes.substitution.dims.find(generic.symbol);
                if (generic.kind != sema::GenericKind::Dim ||
                    bound == op.attributes.substitution.dims.end()) {
                    continue;
                }
                const auto value = types_.substitute(bound->second, frame().subst).constant();
                if (value) {
                    generics.emplace(std::string(generic.name), *value);
                }
            }
        }
        target_.set_call_generics(std::move(generics));
        const auto name = target_.native_call(names.front(), operands, result.shape, result.dtype);
        if (!name) {
            return false;
        }
        result.name = *name;
        define(op, result);
        return true;
    }

    // ------------------------------------------------------- parameters

    void collect_parameters(EntityId block, const Substitution& subst, const std::string& prefix) {
        for (const auto& [entity, name] : block_members(model_, block)) {
            const TypeId type = types_.substitute(model_.entities[entity].type, subst);
            const TypeData& data = types_.get(type);
            const std::string path = prefix + std::string(name);
            const bool is_optional_sub = data.kind == TypeKind::Optional &&
                                         types_.kind(data.elements.front()) == TypeKind::Block;
            if (is_optional_sub) {
                const TypeData& inner = types_.get(data.elements.front());
                if (sub_present(inner, path)) {
                    collect_parameters(inner.decl, block_substitution(inner), path + ".");
                } else {
                    absent_subs_.insert(path);
                }
            } else if (data.kind == TypeKind::Block) {
                collect_parameters(data.decl, block_substitution(data), path + ".");
            } else if (data.kind == TypeKind::Array) {
                const TypeData& element = types_.get(data.elements.front());
                const std::int64_t length = constant(data.value, "array length");
                for (std::int64_t i = 0; i < length; ++i) {
                    collect_parameters(element.decl,
                                       block_substitution(element),
                                       path + "." + std::to_string(i) + ".");
                }
            } else if (model_.entities[entity].is_state) {
                // Becomes an input only if the entry reads it before writing.
                state_types_[path] = tensor_value(type, {});
            } else {
                const bool is_optional = data.kind == TypeKind::Optional;
                if (is_optional &&
                    (!options_.optionals_present || options_.absent.contains(path))) {
                    continue;
                }
                const TypeId tensor = is_optional ? data.elements.front() : type;
                Val value = tensor_value(tensor, {});
                value.name = target_.parameter(path, value.shape, value.dtype);
                if (placing()) {
                    value.slot = offloaded(path) ? host : slot_of(path);
                }
                parameters_[path] = value;
            }
        }
    }

    // Whether an optional `sub` at `path` is present: optionals are, the sub
    // is not named absent itself, and neither is any parameter it requires
    // (a loader lists the parameters a checkpoint lacks, so a sub whose
    // weights are missing is absent as a whole).
    bool sub_present(const TypeData& block_type, const std::string& path) const {
        if (!options_.optionals_present || options_.absent.contains(path)) {
            return false;
        }
        return !requires_absent(block_type.decl, block_substitution(block_type), path + ".");
    }

    bool
    requires_absent(EntityId block, const Substitution& subst, const std::string& prefix) const {
        for (const auto& [name, entity] : model_.decls.at(block).scope) {
            const Entity& member = model_.entities[entity];
            if (member.kind != EntityKind::Member || member.is_state) {
                continue;
            }
            const TypeData& data = types_.get(types_.substitute(member.type, subst));
            const std::string path = prefix + std::string(name);
            if (data.kind == TypeKind::Optional) {
                continue; // optional itself: its absence is its own
            }
            if (data.kind == TypeKind::Block) {
                if (requires_absent(data.decl, block_substitution(data), path + ".")) {
                    return true;
                }
            } else if (data.kind == TypeKind::Array) {
                const TypeData& element = types_.get(data.elements.front());
                const auto length = data.value.constant().value_or(0);
                for (std::int64_t i = 0; i < length; ++i) {
                    if (requires_absent(element.decl,
                                        block_substitution(element),
                                        path + "." + std::to_string(i) + ".")) {
                        return true;
                    }
                }
            } else if (options_.absent.contains(path)) {
                return true;
            }
        }
        return false;
    }

    Substitution block_substitution(const TypeData& block_type) const {
        return sema::substitution_of(model_.decls.at(block_type.decl), block_type);
    }

    // ------------------------------------------------------------ types

    std::int64_t constant(const shape::Poly& poly, const char* what) const {
        const auto value = poly.constant();
        if (!value) {
            fail(std::string(what) + " `" + model_.dims.to_string(poly) +
                 "` is not a constant; bind every generic with --bind");
        }
        return *value;
    }

    Dims concrete_shape(const Shape& shape) const {
        Dims dims;
        for (const ShapeElem& unit : shape) {
            if (unit.is_pack) {
                fail("shape pack `" + std::string(model_.dims.symbol_name(unit.pack)) +
                     "` is not bound");
            }
            dims.push_back(constant(unit.dim, "dimension"));
        }
        return dims;
    }

    ScalarKind concrete_dtype(DType dtype) const {
        if (dtype.is_var) {
            fail("dtype `" + types_.dtype_var(dtype.var).name + "` is not bound");
        }
        return dtype.scalar;
    }

    Val tensor_value(TypeId type, const Substitution& subst) {
        const TypeData& data = types_.get(types_.substitute(type, subst));
        Val value;
        if (data.kind == TypeKind::Scalar || data.kind == TypeKind::CompileInt) {
            value.dtype =
                data.kind == TypeKind::Scalar ? concrete_dtype(data.dtype) : ScalarKind::I64;
        } else if (data.kind == TypeKind::Tensor) {
            value.shape = concrete_shape(data.shape);
            value.dtype = concrete_dtype(data.dtype);
        } else {
            fail("type `" + types_.to_string(type) + "` has no tensor representation");
        }
        return value;
    }

    // --------------------------------------------------------- emission

    Val tensor(std::string name, Dims shape, ScalarKind dtype) {
        Val result;
        result.name = std::move(name);
        result.shape = std::move(shape);
        result.dtype = dtype;
        result.grid_rank = grid_.size();
        return result;
    }

    Val constant_scalar(const Literal& literal, ScalarKind dtype) {
        return tensor(target_.constant(literal, dtype), {}, dtype);
    }

    Val constant_int(std::int64_t value, ScalarKind dtype) {
        Literal literal = Literal::of_integer(value);
        if (dtype == ScalarKind::Bool) {
            literal = {Literal::Kind::Boolean, value != 0 ? 1 : 0, 0.0};
        } else if (is_float(dtype)) {
            literal = Literal::of_real(static_cast<double>(value));
        }
        return constant_scalar(literal, dtype);
    }

    Val elementwise(Elementwise kind,
                    const std::vector<Val>& operands,
                    const Dims& shape,
                    ScalarKind dtype) {
        std::vector<TensorInfo> infos;
        infos.reserve(operands.size());
        for (const Val& operand : operands) {
            infos.push_back(info(operand));
        }
        return tensor(target_.elementwise(kind, infos, shape, dtype), shape, dtype);
    }

    Val broadcast(const Val& value, const Dims& shape, const Dims& dims) {
        if (value.shape == shape) {
            return value;
        }
        return tensor(target_.broadcast(info(value), dims, shape), shape, value.dtype);
    }

    // Right-aligned broadcasting, as Linnet's elementwise operators use.
    Val broadcast_trailing(const Val& value, const Dims& shape) {
        Dims dims;
        dims.reserve(value.shape.size());
        for (std::size_t i = 0; i < value.shape.size(); ++i) {
            dims.push_back(static_cast<std::int64_t>(shape.size() - value.shape.size() + i));
        }
        return broadcast(value, shape, dims);
    }

    Val convert(const Val& value, ScalarKind dtype) {
        if (value.dtype == dtype) {
            return value;
        }
        return tensor(target_.convert(info(value), dtype), value.shape, dtype);
    }

    Val reshape(const Val& value, const Dims& shape) {
        if (value.shape == shape) {
            return value;
        }
        return tensor(target_.reshape(info(value), shape), shape, value.dtype);
    }

    // ------------------------------------------------------------- grid

    bool in_grid() const { return !grid_.empty(); }

    // The position along grid axis `axis`, as an i64 tensor over the grid.
    Val iota(std::size_t axis) {
        const Val positions = tensor(target_.iota(grid_[axis]), {grid_[axis]}, ScalarKind::I64);
        return broadcast(positions, grid_, {static_cast<std::int64_t>(axis)});
    }

    Val to_grid(const Val& value) {
        if (value.kind == Val::Kind::Index) {
            return iota(value.axes.front());
        }
        if (value.kind != Val::Kind::Tensor) {
            fail("a compile-time value was used where a tensor is needed");
        }
        if (value.shape == grid_) {
            return value;
        }
        Dims dims;
        for (std::size_t i = 0; i < value.shape.size(); ++i) {
            if (i >= value.grid_rank || i >= grid_.size() || value.shape[i] != grid_[i]) {
                fail("a tensor value of rank " + std::to_string(value.shape.size()) +
                     " was used as a scalar inside index notation");
            }
            dims.push_back(static_cast<std::int64_t>(i));
        }
        return broadcast(value, grid_, dims);
    }

    std::vector<Val> aligned(const std::vector<Val>& operands, const Dims& shape) {
        std::vector<Val> out;
        out.reserve(operands.size());
        // A target that broadcasts its own elementwise operands needs the
        // broadcasts only when no operand already has the result shape: the
        // shapes are right-aligned, which is the rule those targets use.
        const bool implicit =
            target_.broadcasts_elementwise() && !in_grid() &&
            std::any_of(operands.begin(), operands.end(), [&](const Val& operand) {
                return operand.kind == Val::Kind::Tensor && operand.shape == shape;
            });
        for (const Val& operand : operands) {
            if (implicit && operand.kind == Val::Kind::Tensor &&
                trailing_compatible(operand.shape, shape)) {
                Val kept = operand;
                kept.shape = shape; // what the operation produces, not what it reads
                out.push_back(kept);
                continue;
            }
            out.push_back(in_grid() ? to_grid(operand) : broadcast_trailing(operand, shape));
        }
        return out;
    }

    // Right-aligned axes that are equal or one: what implicit broadcasting
    // accepts without a copy.
    static bool trailing_compatible(const Dims& from, const Dims& to) {
        if (from.size() > to.size()) {
            return false;
        }
        for (std::size_t i = 0; i < from.size(); ++i) {
            const std::int64_t axis = from[from.size() - 1 - i];
            const std::int64_t target = to[to.size() - 1 - i];
            if (axis != target && axis != 1) {
                return false;
            }
        }
        return true;
    }

    // ---------------------------------------------------------- running

    Frame& frame() { return frames_.back(); }

    const Val& value(ir::ValueId id) {
        const auto found = frame().values.find(id);
        if (found == frame().values.end()) {
            fail("internal: value used before definition");
        }
        return found->second;
    }

    Dims result_shape(const ir::Operation& op) {
        if (in_grid()) {
            return grid_;
        }
        return tensor_value(module_.value(op.results.front()).type, frame().subst).shape;
    }

    ScalarKind result_dtype(const ir::Operation& op) {
        return tensor_value(module_.value(op.results.front()).type, frame().subst).dtype;
    }

    std::vector<Val> run_block(const ir::Block& block) {
        for (const ir::OpId id : block.ops) {
            const ir::Operation& op = module_.op(id);
            if (op.kind == ir::OpKind::Return || op.kind == ir::OpKind::Yield) {
                std::vector<Val> results;
                results.reserve(op.operands.size());
                for (const ir::ValueId operand : op.operands) {
                    results.push_back(value(operand));
                }
                return results;
            }
            run_op(op);
        }
        return {};
    }

    std::vector<Val> run_region(ir::RegionId region, const std::vector<Val>& arguments) {
        const ir::Block& block = module_.block(module_.region(region).blocks.front());
        for (std::size_t i = 0; i < arguments.size() && i < block.arguments.size(); ++i) {
            frame().values[block.arguments[i]] = arguments[i];
        }
        return run_block(block);
    }

    void define(const ir::Operation& op, Val value) {
        if (placing()) {
            stamp(value);
        }
        frame().values[op.results.front()] = std::move(value);
    }

    // ------------------------------------------------------- placement

    bool placing() const {
        return !options_.placement.empty() || !options_.offload.empty() ||
               !options_.fully_shard.empty();
    }

    // The slot a member path runs on: the longest placement key it starts
    // with, or slot 0.
    int slot_of(const std::string& path) const {
        int slot = 0;
        std::size_t longest = 0;
        for (const auto& [prefix, assigned] : options_.placement) {
            if (prefix.size() > longest && path.starts_with(prefix)) {
                slot = assigned;
                longest = prefix.size();
            }
        }
        return slot;
    }

    // Offloaded or sharded: the parameter is not whole on the device until
    // its block asks for it.
    bool offloaded(const std::string& path) const {
        const auto under = [&](const std::string& prefix) { return path.starts_with(prefix); };
        return std::ranges::any_of(options_.offload, under) ||
               std::ranges::any_of(options_.fully_shard, under);
    }

    int current_slot() const { return slots_.back(); }

    void stamp(Val& value) const {
        if (value.kind == Val::Kind::Tensor && value.slot == unplaced) {
            value.slot = current_slot();
        }
        for (Val& element : value.elements) {
            stamp(element);
        }
    }

    // `value` as it is on `slot`: itself if it is already there, otherwise a
    // transfer, made once per scope. A transfer of an offloaded parameter
    // belongs to the offloaded block that asked for it and is released when
    // that block returns.
    Val on_slot(const Val& value, int slot) {
        if (value.kind != Val::Kind::Tensor) {
            Val out = value;
            for (Val& element : out.elements) {
                element = on_slot(element, slot);
            }
            return out;
        }
        const int from = value.slot == unplaced ? 0 : value.slot;
        if (from == slot) {
            return value;
        }
        const std::pair<std::string, int> key{value.name, slot};
        for (auto scope = moved_.rbegin(); scope != moved_.rend(); ++scope) {
            const auto found = scope->find(key);
            if (found != scope->end()) {
                Val out = value;
                out.name = found->second;
                out.slot = slot;
                return out;
            }
        }
        Val out = value;
        out.name = from == host && !options_.fully_shard.empty()
                       ? target_.gather(info(value))
                       : target_.transfer(info(value), slot);
        out.slot = slot;
        if (from == host && !releases_.empty()) {
            releases_.back().push_back(out.name);
            released_keys_.back().push_back(key);
        }
        moved_.back()[key] = out.name;
        return out;
    }

    // Inside index notation most operations are scalar and evaluate over the
    // grid, but a tensor-valued expression there is a whole tensor computed
    // once.
    void run_op(const ir::Operation& op) {
        if (!placing() || op.kind == ir::OpKind::Call) {
            run_placed_op(op);
            return;
        }
        // Operands on another slot are replaced for the duration of this one
        // operation, so the operation and everything it defines live here.
        std::vector<std::pair<ir::ValueId, Val>> saved;
        for (const ir::ValueId id : op.operands) {
            const auto found = frame().values.find(id);
            if (found == frame().values.end() || found->second.kind == Val::Kind::Block) {
                continue;
            }
            Val moved = on_slot(found->second, current_slot());
            if (moved.name != found->second.name || !moved.elements.empty()) {
                saved.emplace_back(id, found->second);
                found->second = std::move(moved);
            }
        }
        run_placed_op(op);
        for (auto& [id, original] : saved) {
            frame().values[id] = std::move(original);
        }
    }

    void run_placed_op(const ir::Operation& op) {
        if (in_grid() && op.results.size() == 1 && op.kind != ir::OpKind::Reduce) {
            const TypeData& data = types_.get(
                types_.substitute(module_.value(op.results.front()).type, frame().subst));
            if (data.kind == TypeKind::Tensor) {
                const Dims saved = grid_;
                grid_.clear();
                run_scalar_or_tensor_op(op);
                grid_ = saved;
                return;
            }
        }
        run_scalar_or_tensor_op(op);
    }

    static std::optional<Elementwise> elementwise_kind(ir::OpKind kind) {
        switch (kind) {
        case ir::OpKind::Add:
            return Elementwise::Add;
        case ir::OpKind::Sub:
            return Elementwise::Sub;
        case ir::OpKind::Mul:
            return Elementwise::Mul;
        case ir::OpKind::Div:
            return Elementwise::Div;
        case ir::OpKind::Rem:
            return Elementwise::Rem;
        case ir::OpKind::Min:
            return Elementwise::Min;
        case ir::OpKind::Max:
            return Elementwise::Max;
        case ir::OpKind::BitAnd:
            return Elementwise::BitAnd;
        case ir::OpKind::BitOr:
            return Elementwise::BitOr;
        case ir::OpKind::BitXor:
            return Elementwise::BitXor;
        case ir::OpKind::Shl:
            return Elementwise::Shl;
        case ir::OpKind::Shr:
            return Elementwise::Shr;
        case ir::OpKind::And:
            return Elementwise::And;
        case ir::OpKind::Or:
            return Elementwise::Or;
        case ir::OpKind::Not:
            return Elementwise::Not;
        case ir::OpKind::Neg:
            return Elementwise::Neg;
        case ir::OpKind::Exp:
            return Elementwise::Exp;
        case ir::OpKind::Log:
            return Elementwise::Log;
        case ir::OpKind::Sqrt:
            return Elementwise::Sqrt;
        case ir::OpKind::Rsqrt:
            return Elementwise::Rsqrt;
        case ir::OpKind::Sin:
            return Elementwise::Sin;
        case ir::OpKind::Cos:
            return Elementwise::Cos;
        case ir::OpKind::Tanh:
            return Elementwise::Tanh;
        case ir::OpKind::Abs:
            return Elementwise::Abs;
        default:
            return std::nullopt;
        }
    }

    void run_scalar_or_tensor_op(const ir::Operation& op) {
        const ir::Attributes& a = op.attributes;
        const auto operand = [&](std::size_t i) -> const Val& { return value(op.operands[i]); };
        if (const auto kind = elementwise_kind(op.kind)) {
            const Dims shape = result_shape(op);
            std::vector<Val> operands;
            operands.reserve(op.operands.size());
            for (const ir::ValueId id : op.operands) {
                operands.push_back(value(id));
            }
            define(op, elementwise(*kind, aligned(operands, shape), shape, result_dtype(op)));
            return;
        }
        switch (op.kind) {
        case ir::OpKind::ConstInt:
            define(op, constant_int(a.integer, result_dtype(op)));
            return;
        case ir::OpKind::ConstBool:
            define(op, constant_int(a.integer, ScalarKind::Bool));
            return;
        case ir::OpKind::ConstFloat: {
            define(op, constant_scalar(Literal::of_real(a.number), result_dtype(op)));
            return;
        }
        case ir::OpKind::ConstDim:
            define(op,
                   constant_int(constant(types_.substitute(a.dim, frame().subst), "dimension"),
                                ScalarKind::I64));
            return;
        case ir::OpKind::Compare: {
            const Dims shape = result_shape(op);
            const std::vector<Val> operands = aligned({operand(0), operand(1)}, shape);
            define(op,
                   tensor(target_.compare(a.compare, info(operands[0]), info(operands[1]), shape),
                          shape,
                          ScalarKind::Bool));
            return;
        }
        case ir::OpKind::Select: {
            const Dims shape = result_shape(op);
            const std::vector<Val> operands = aligned({operand(0), operand(1), operand(2)}, shape);
            const ScalarKind dtype = result_dtype(op);
            define(
                op,
                tensor(target_.select(
                           info(operands[0]), info(operands[1]), info(operands[2]), shape, dtype),
                       shape,
                       dtype));
            return;
        }
        case ir::OpKind::Cast: {
            const Val source = in_grid() ? to_grid(operand(0)) : operand(0);
            define(op, convert(source, result_dtype(op)));
            return;
        }
        case ir::OpKind::Reshape:
            define(op, reshape(operand(0), result_shape(op)));
            return;
        case ir::OpKind::Broadcast:
            define(op, broadcast_trailing(operand(0), result_shape(op)));
            return;
        case ir::OpKind::Permute: {
            Dims permutation;
            for (const ShapeElem& axis : a.shape) {
                permutation.push_back(constant(axis.dim, "axis"));
            }
            const Val& source = operand(0);
            const Dims shape = result_shape(op);
            define(
                op,
                tensor(target_.transpose(info(source), permutation, shape), shape, source.dtype));
            return;
        }
        case ir::OpKind::Slice: {
            const Val& source = operand(0);
            Dims starts;
            Dims limits;
            Dims strides;
            Dims sliced;
            std::size_t axis = 0;
            for (std::size_t i = 0; i < a.starts.size(); ++i) {
                // A whole shape-pack entry covers every axis the pack binds to.
                const std::size_t count =
                    a.whole[i] && a.pack_units[i].is_pack
                        ? concrete_shape(types_.substitute(Shape{a.pack_units[i]}, frame().subst))
                              .size()
                        : 1;
                for (std::size_t k = 0; k < count; ++k, ++axis) {
                    const std::int64_t size = source.shape[axis];
                    const std::int64_t start =
                        a.whole[i] ? 0
                                   : constant(types_.substitute(a.starts[i], frame().subst),
                                              "slice start");
                    const std::int64_t stop =
                        a.whole[i]
                            ? size
                            : constant(types_.substitute(a.stops[i], frame().subst), "slice stop");
                    const std::int64_t step = a.whole[i] ? 1 : a.steps[i];
                    starts.push_back(start);
                    limits.push_back(stop);
                    strides.push_back(step);
                    sliced.push_back((stop - start + step - 1) / step);
                }
            }
            const Val result = tensor(
                target_.slice(info(source), starts, limits, strides, sliced), sliced, source.dtype);
            define(op, reshape(result, result_shape(op)));
            return;
        }
        case ir::OpKind::Concat: {
            std::vector<TensorInfo> parts;
            parts.reserve(op.operands.size());
            for (const ir::ValueId id : op.operands) {
                parts.push_back(info(value(id)));
            }
            const auto rank = static_cast<std::int64_t>(parts.front().shape.size());
            const std::int64_t axis = a.axis < 0 ? a.axis + rank : a.axis;
            const Dims shape = result_shape(op);
            define(op, tensor(target_.concat(parts, axis, shape), shape, parts.front().dtype));
            return;
        }
        case ir::OpKind::Cumsum: {
            const Val x = whole_tensor(operand(0), "cumsum");
            const auto rank = static_cast<std::int64_t>(x.shape.size());
            const std::int64_t axis = a.axis < 0 ? a.axis + rank : a.axis;
            define(op, tensor(target_.cumsum(info(x), axis), x.shape, x.dtype));
            return;
        }
        case ir::OpKind::Fill:
            define(op, broadcast(operand(0), result_shape(op), {}));
            return;
        case ir::OpKind::Iota: {
            const Dims shape = result_shape(op);
            const Val positions = tensor(target_.iota(shape.front()), shape, ScalarKind::I64);
            define(op, convert(positions, result_dtype(op)));
            return;
        }
        case ir::OpKind::Element:
            define(op, element(op));
            return;
        case ir::OpKind::Comprehension:
            define(op, comprehension(op));
            return;
        case ir::OpKind::Reduce:
            define(op, reduce(op));
            return;
        case ir::OpKind::TupleMake:
        case ir::OpKind::StructMake: {
            // A struct is its fields in order, as a tuple is its elements.
            Val tuple;
            tuple.kind = Val::Kind::Tuple;
            for (const ir::ValueId id : op.operands) {
                tuple.elements.push_back(value(id));
            }
            define(op, tuple);
            return;
        }
        case ir::OpKind::TupleGet:
        case ir::OpKind::StructGet: {
            const Val& tuple = operand(0);
            if (tuple.kind != Val::Kind::Tuple ||
                static_cast<std::size_t>(a.integer) >= tuple.elements.size()) {
                fail("internal: " + std::string(ir::op_spelling(op.kind)) +
                     " on a value with no such element");
            }
            define(op, tuple.elements[static_cast<std::size_t>(a.integer)]);
            return;
        }
        case ir::OpKind::OptionSome: {
            Val some;
            some.kind = Val::Kind::Some;
            some.elements.push_back(operand(0));
            define(op, some);
            return;
        }
        case ir::OpKind::OptionNone: {
            Val none;
            none.kind = Val::Kind::None;
            define(op, none);
            return;
        }
        case ir::OpKind::OptionMatch: {
            const Val& subject = operand(0);
            const std::vector<Val> results =
                subject.kind == Val::Kind::Some
                    ? run_region(op.regions[0], {subject.elements.front()})
                    : run_region(op.regions[1], {});
            define(op, results.front());
            return;
        }
        case ir::OpKind::If: {
            const Val& condition = operand(0);
            const Val then_value = run_region(op.regions[0], {}).front();
            const Val else_value = run_region(op.regions[1], {}).front();
            if (then_value.kind != Val::Kind::Tensor) {
                fail("`if` over compile-time values is not supported");
            }
            const Dims shape = result_shape(op);
            const std::vector<Val> operands = aligned({condition, then_value, else_value}, shape);
            define(op,
                   tensor(target_.select(info(operands[0]),
                                         info(operands[1]),
                                         info(operands[2]),
                                         shape,
                                         then_value.dtype),
                          shape,
                          then_value.dtype));
            return;
        }
        case ir::OpKind::Call:
        case ir::OpKind::SemanticCall:
            call(op);
            return;
        case ir::OpKind::BlockParam: {
            const Val& block = operand(0);
            const std::string path = block.path + a.name;
            const TypeData& data =
                types_.get(types_.substitute(member(block, a.name), block.subst));
            const auto found = parameters_.find(path);
            if (data.kind == TypeKind::Optional) {
                Val optional;
                optional.kind = found == parameters_.end() ? Val::Kind::None : Val::Kind::Some;
                if (found != parameters_.end()) {
                    optional.elements.push_back(found->second);
                }
                define(op, optional);
                return;
            }
            if (found == parameters_.end()) {
                fail("internal: parameter `" + path + "` was not collected");
            }
            define(op, found->second);
            return;
        }
        case ir::OpKind::BlockSub: {
            const Val& block = operand(0);
            const TypeData& data =
                types_.get(types_.substitute(member(block, a.name), block.subst));
            const std::string path = block.path + a.name;
            if (data.kind == TypeKind::Optional) {
                Val optional;
                optional.kind = absent_subs_.contains(path) ? Val::Kind::None : Val::Kind::Some;
                if (optional.kind == Val::Kind::Some) {
                    optional.elements.push_back(sub_value(types_.get(data.elements.front()), path));
                }
                define(op, optional);
                return;
            }
            define(op, sub_value(data, path));
            return;
        }
        case ir::OpKind::StateRead: {
            const std::string path = operand(0).path + a.name;
            auto found = states_.find(path);
            if (found == states_.end()) {
                // First read before any write: the value before the call.
                const auto declared = state_types_.find(path);
                if (declared == state_types_.end()) {
                    fail("internal: state `" + path + "` was not collected");
                }
                Val value = declared->second;
                value.name = target_.state(path, value.shape, value.dtype);
                if (placing()) {
                    value.slot = slot_of(path); // a cache stays where its block runs
                }
                found = states_.emplace(path, value).first;
            }
            define(op, found->second);
            return;
        }
        case ir::OpKind::StateWrite: {
            const std::string path = operand(0).path + a.name;
            const Val& value = operand(1);
            if (value.kind != Val::Kind::Tensor || value.grid_rank != 0) {
                fail("a state write needs a whole tensor");
            }
            if (std::find(written_states_.begin(), written_states_.end(), path) ==
                written_states_.end()) {
                written_states_.push_back(path);
            }
            states_[path] = value;
            return;
        }
        case ir::OpKind::ArrayGet: {
            const Val& array = operand(0);
            const std::int64_t position = constant_of(op.operands[1]);
            if (array.kind != Val::Kind::Array || position < 0 ||
                static_cast<std::size_t>(position) >= array.elements.size()) {
                fail("array index out of range while unrolling");
            }
            define(op, array.elements[static_cast<std::size_t>(position)]);
            return;
        }
        case ir::OpKind::StaticFor: {
            const Val& array = operand(0);
            if (array.kind != Val::Kind::Array) {
                fail("`static for` over a non-array");
            }
            std::vector<Val> carried;
            for (std::size_t i = 1; i < op.operands.size(); ++i) {
                carried.push_back(operand(i));
            }
            for (const Val& element : array.elements) {
                std::vector<Val> arguments{element};
                arguments.insert(arguments.end(), carried.begin(), carried.end());
                carried = run_region(op.regions.front(), arguments);
            }
            for (std::size_t i = 0; i < op.results.size(); ++i) {
                frame().values[op.results[i]] = carried[i];
            }
            return;
        }
        case ir::OpKind::While:
            run_while(op);
            return;
        case ir::OpKind::For:
            run_for(op);
            return;
        case ir::OpKind::StaticRange: {
            const std::int64_t start = range_bound(op.operands[0]);
            const std::int64_t stop = range_bound(op.operands[1]);
            std::vector<Val> carried;
            for (std::size_t i = 2; i < op.operands.size(); ++i) {
                carried.push_back(operand(i));
            }
            for (std::int64_t i = start; i < stop; ++i) {
                std::vector<Val> arguments{constant_int(i, ScalarKind::I64)};
                arguments.insert(arguments.end(), carried.begin(), carried.end());
                carried = run_region(op.regions.front(), arguments);
            }
            for (std::size_t i = 0; i < op.results.size(); ++i) {
                frame().values[op.results[i]] = carried[i];
            }
            return;
        }
        case ir::OpKind::KernelProgramId:
            define(op, tensor(kernel_target().program_id(a.integer), {}, ScalarKind::I32));
            return;
        case ir::OpKind::KernelLoad: {
            const auto rank = static_cast<std::size_t>(a.integer);
            std::vector<TensorInfo> indices;
            for (std::size_t i = 1; i <= rank; ++i) {
                indices.push_back(info(operand(i)));
            }
            std::optional<TensorInfo> mask;
            std::optional<TensorInfo> other;
            if (op.operands.size() > rank + 1) {
                mask = info(operand(rank + 1));
            }
            if (op.operands.size() > rank + 2) {
                other = info(operand(rank + 2));
            }
            const Dims shape = result_shape(op);
            const ScalarKind dtype = result_dtype(op);
            define(
                op,
                tensor(kernel_target().load(info(operand(0)), indices, mask, other, shape, dtype),
                       shape,
                       dtype));
            return;
        }
        case ir::OpKind::KernelStore: {
            const auto rank = static_cast<std::size_t>(a.integer);
            std::vector<TensorInfo> indices;
            for (std::size_t i = 1; i <= rank; ++i) {
                indices.push_back(info(operand(i)));
            }
            std::optional<TensorInfo> mask;
            if (op.operands.size() > rank + 2) {
                mask = info(operand(rank + 2));
            }
            std::optional<Reduction> atomic;
            if (a.name == "add") {
                atomic = Reduction::Sum;
            } else if (a.name == "max") {
                atomic = Reduction::Max;
            } else if (a.name == "min") {
                atomic = Reduction::Min;
            }
            kernel_target().store(info(operand(0)), indices, info(operand(rank + 1)), mask, atomic);
            return;
        }
        case ir::OpKind::EnumConst: {
            Val variant;
            variant.kind = Val::Kind::Enum;
            variant.path = a.name;
            define(op, variant);
            return;
        }
        case ir::OpKind::EnumMatch: {
            const Val& subject = operand(0);
            if (subject.kind != Val::Kind::Enum) {
                fail("`match` on a runtime enum value is not supported");
            }
            for (std::size_t i = 0; i < a.names.size() && i < op.regions.size(); ++i) {
                if (a.names[i] == subject.path || a.names[i] == "_") {
                    define(op, run_region(op.regions[i], {subject}).front());
                    return;
                }
            }
            fail("no arm matches enum variant `" + subject.path + "`");
        }
        default:
            return;
        }
    }

    std::int64_t constant_of(ir::ValueId id) {
        const ir::OpId producer = module_.value(id).producer;
        if (producer != ir::no_id) {
            const ir::Operation& op = module_.op(producer);
            if (op.kind == ir::OpKind::ConstInt) {
                return op.attributes.integer;
            }
            if (op.kind == ir::OpKind::ConstDim) {
                return constant(types_.substitute(op.attributes.dim, frame().subst), "index");
            }
        }
        fail("array indices must be compile-time constants");
    }

    TypeId member(const Val& block, const std::string& name) const {
        if (block.kind != Val::Kind::Block) {
            fail("internal: member access on a non-block value");
        }
        const DeclInfo& decl = model_.decls.at(block.block);
        const auto found = decl.scope.find(name);
        if (found == decl.scope.end()) {
            fail("internal: no member `" + name + "`");
        }
        return model_.entities[found->second].type;
    }

    Val sub_value(const TypeData& data, const std::string& path) {
        if (data.kind == TypeKind::Block) {
            Val block;
            block.kind = Val::Kind::Block;
            block.block = data.decl;
            block.subst = block_substitution(data);
            block.path = path + ".";
            return block;
        }
        if (data.kind == TypeKind::Array) {
            Val array;
            array.kind = Val::Kind::Array;
            const std::int64_t length = constant(data.value, "array length");
            for (std::int64_t i = 0; i < length; ++i) {
                array.elements.push_back(
                    sub_value(types_.get(data.elements.front()), path + "." + std::to_string(i)));
            }
            return array;
        }
        fail("internal: `sub` of a non-block type");
    }

    // ------------------------------------------------------------ calls

    void call(const ir::Operation& op) {
        const auto found = functions_.find(op.attributes.name);
        if (found == functions_.end()) {
            fail("callee `" + op.attributes.name + "` is not defined in this program");
        }
        const ir::Function& callee = *found->second;
        if (!callee.kernel.empty() && try_kernel(op, callee)) {
            return;
        }
        if (callee.gradient != ir::no_id) {
            call_with_gradient(op, callee);
            return;
        }
        if (op.kind == ir::OpKind::SemanticCall && try_native(op)) {
            return;
        }
        Frame inner;
        inner.subst = types_.substitute(op.attributes.substitution, frame().subst);
        std::vector<Val> arguments;
        arguments.reserve(op.operands.size());
        for (const ir::ValueId id : op.operands) {
            arguments.push_back(value(id));
        }
        if (!arguments.empty() && arguments.front().kind == Val::Kind::Block) {
            for (const auto& [symbol, dim] : arguments.front().subst.dims) {
                inner.subst.dims.emplace(symbol, dim);
            }
            for (const auto& [symbol, shape] : arguments.front().subst.packs) {
                inner.subst.packs.emplace(symbol, shape);
            }
            for (const auto& [var, dtype] : arguments.front().subst.dtypes) {
                inner.subst.dtypes.emplace(var, dtype);
            }
        }
        const Dims saved_grid = grid_;
        grid_.clear();
        frames_.push_back(std::move(inner));
        bool entered = false;
        bool unit = false;
        if (placing() && !arguments.empty() && arguments.front().kind == Val::Kind::Block) {
            const std::string& path = arguments.front().path;
            slots_.push_back(slot_of(path));
            target_.set_slot(current_slot());
            entered = true;
            unit = std::ranges::find(options_.offload, path) != options_.offload.end() ||
                   std::ranges::find(options_.fully_shard, path) != options_.fully_shard.end();
            if (unit) {
                releases_.emplace_back();
                released_keys_.emplace_back();
            }
        }
        const bool remat =
            !arguments.empty() && arguments.front().kind == Val::Kind::Block &&
            std::ranges::find(options_.remat, arguments.front().path) != options_.remat.end();
        if (remat) {
            target_.begin_remat();
        }
        const std::vector<Val> results = run_region(callee.body, arguments);
        if (unit) {
            if (!releases_.back().empty()) {
                target_.release(releases_.back());
            }
            for (const auto& key : released_keys_.back()) {
                for (auto& scope : moved_) {
                    scope.erase(key);
                }
            }
            releases_.pop_back();
            released_keys_.pop_back();
        }
        if (remat) {
            target_.end_remat();
        }
        if (entered) {
            slots_.pop_back();
            target_.set_slot(current_slot());
        }
        frames_.pop_back();
        grid_ = saved_grid;
        if (!op.results.empty()) {
            if (results.empty()) {
                fail("internal: call without a result");
            }
            Val result = results.front();
            result.grid_rank = grid_.size();
            define(op, result);
        }
    }

    // An op with a `grad`: its canonical body runs, bracketed for the
    // target, which takes the backward pass from `grad` instead of the body.
    void call_with_gradient(const ir::Operation& op, const ir::Function& callee) {
        const Substitution subst = types_.substitute(op.attributes.substitution, frame().subst);
        std::vector<Val> arguments;
        arguments.reserve(op.operands.size());
        for (const ir::ValueId id : op.operands) {
            arguments.push_back(value(id));
        }
        // The tensor arguments cross to the target; which of them `grad`
        // returns a gradient for, the parameter types say.
        const ir::Block& gradient = module_.block(module_.region(callee.gradient).blocks.front());
        std::vector<std::size_t> tensors;
        std::vector<TensorInfo> infos;
        std::vector<bool> takes;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            takes.push_back(
                sema::takes_gradient(types_, module_.value(gradient.arguments[i]).type));
            if (arguments[i].kind == Val::Kind::Tensor) {
                tensors.push_back(i);
                infos.push_back(info(arguments[i]));
            }
        }
        const std::vector<std::string> names = target_.begin_custom_gradient(infos);
        std::vector<Val> inside = arguments;
        for (std::size_t k = 0; k < tensors.size(); ++k) {
            inside[tensors[k]].name = names[k];
        }
        const std::vector<Val> results = run_call_region(callee.body, subst, inside);
        if (results.empty() || results.front().kind != Val::Kind::Tensor) {
            fail("internal: an op with a `grad` must return one tensor");
        }
        const GraphTarget::Pullback pullback = make_pullback(callee, subst, arguments);
        Val result = results.front();
        result.name = target_.end_custom_gradient(infos, info(result), pullback);
        result.grid_rank = grid_.size();
        define(op, result);
    }

    // The backward pass of an op's `grad`, for the tensor arguments of a
    // call: emitted from the names it is given (see `GraphTarget::Pullback`).
    GraphTarget::Pullback make_pullback(const ir::Function& callee,
                                        const Substitution& subst,
                                        const std::vector<Val>& arguments) {
        const ir::Block& gradient = module_.block(module_.region(callee.gradient).blocks.front());
        std::vector<std::size_t> tensors;
        std::vector<bool> takes;
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            takes.push_back(
                sema::takes_gradient(types_, module_.value(gradient.arguments[i]).type));
            if (arguments[i].kind == Val::Kind::Tensor) {
                tensors.push_back(i);
            }
        }
        const ir::RegionId region = callee.gradient;
        return [this, region, subst, arguments, tensors, takes](
                   const std::vector<TensorInfo>& given,
                   const TensorInfo& result,
                   const TensorInfo& grad) -> std::vector<std::optional<std::string>> {
            std::vector<Val> bound = arguments;
            for (std::size_t k = 0; k < tensors.size(); ++k) {
                bound[tensors[k]].name = given[k].name;
            }
            Val y;
            y.name = result.name;
            y.shape = result.shape;
            y.dtype = result.dtype;
            Val dy = y;
            dy.name = grad.name;
            bound.push_back(y);
            bound.push_back(dy);
            const std::vector<Val> returned = run_call_region(region, subst, bound);
            if (returned.empty()) {
                fail("internal: a `grad` without a result");
            }
            const std::vector<Val> gradients = returned.front().kind == Val::Kind::Tuple
                                                   ? returned.front().elements
                                                   : std::vector<Val>{returned.front()};
            std::vector<std::optional<std::string>> out(tensors.size());
            std::size_t next = 0;
            for (std::size_t k = 0; k < tensors.size(); ++k) {
                if (takes[tensors[k]] && next < gradients.size()) {
                    out[k] = gradients[next++].name;
                }
            }
            return out;
        };
    }

    // An op with a `kernel`, launched in place of its body where the target
    // runs kernels and the kernel's `where` clause holds; false otherwise.
    bool try_kernel(const ir::Operation& op, const ir::Function& callee) {
        if (!options_.kernels) {
            return false;
        }
        const std::unique_ptr<KernelTarget> writer = target_.kernel_target();
        const auto found = functions_.find(callee.kernel);
        if (!writer || !grid_.empty() || found == functions_.end()) {
            return false;
        }
        const ir::Function& kernel = *found->second;
        const Substitution subst = types_.substitute(op.attributes.substitution, frame().subst);
        Substitution bound;
        for (std::size_t i = 0; i < kernel.generics.size() && i < callee.kernel_args.size(); ++i) {
            sema::bind(bound, kernel.generics[i], types_.substitute(callee.kernel_args[i], subst));
        }
        for (const sema::ConstraintInfo& constraint : kernel.constraints) {
            const std::int64_t lhs =
                constant(types_.substitute(constraint.lhs, bound), "a kernel's constraint");
            const std::int64_t rhs =
                constant(types_.substitute(constraint.rhs, bound), "a kernel's constraint");
            if (!relation_holds(constraint.relation, lhs, rhs)) {
                return false;
            }
        }
        std::vector<Val> arguments;
        for (const ir::ValueId id : op.operands) {
            arguments.push_back(value(id));
            if (arguments.back().kind != Val::Kind::Tensor) {
                return false;
            }
        }

        KernelLaunch launch;
        launch.op = callee.name;
        launch.name = kernel.name;
        launch.warps = kernel.warps;
        launch.stages = kernel.stages;
        // The grid's sizes, compile-time integers of the kernel's generics.
        Frame sizes;
        sizes.subst = bound;
        frames_.push_back(std::move(sizes));
        const ir::Block& grid = module_.block(module_.region(kernel.grid).blocks.front());
        for (const ir::ValueId size : module_.op(grid.ops.back()).operands) {
            launch.grid.push_back(range_bound(size));
        }
        frames_.pop_back();
        for (const Val& argument : arguments) {
            launch.arguments.push_back(info(argument));
        }
        const ir::Block& body = module_.block(module_.region(kernel.body).blocks.front());
        for (std::size_t i = body.arguments.size() - kernel.outputs; i < body.arguments.size();
             ++i) {
            const TypeData& data =
                types_.get(types_.substitute(module_.value(body.arguments[i]).type, bound));
            launch.results.push_back({"", concrete_shape(data.shape), concrete_dtype(data.dtype)});
        }
        launch.program = Evaluator(module_, options_, *writer).run_kernel(kernel, bound);
        launch.body = [this, &callee, subst, arguments](const std::vector<TensorInfo>& given) {
            std::vector<Val> bound_arguments = arguments;
            for (std::size_t i = 0; i < given.size() && i < bound_arguments.size(); ++i) {
                bound_arguments[i].name = given[i].name;
            }
            const std::vector<Val> returned = run_call_region(callee.body, subst, bound_arguments);
            std::vector<std::string> names;
            for (const Val& result : returned.empty() || returned.front().kind != Val::Kind::Tuple
                                         ? returned
                                         : returned.front().elements) {
                names.push_back(result.name);
            }
            return names;
        };
        if (callee.gradient != ir::no_id) {
            launch.pullback = make_pullback(callee, subst, arguments);
        }
        const auto names = target_.launch_kernel(launch);
        if (!names || names->size() != launch.results.size()) {
            return false;
        }
        Val result;
        if (names->size() == 1) {
            result =
                tensor(names->front(), launch.results.front().shape, launch.results.front().dtype);
        } else {
            result.kind = Val::Kind::Tuple;
            for (std::size_t i = 0; i < names->size(); ++i) {
                result.elements.push_back(
                    tensor((*names)[i], launch.results[i].shape, launch.results[i].dtype));
            }
        }
        define(op, result);
        return true;
    }

    static bool relation_holds(shape::Relation relation, std::int64_t lhs, std::int64_t rhs) {
        switch (relation) {
        case shape::Relation::Equal:
            return lhs == rhs;
        case shape::Relation::NotEqual:
            return lhs != rhs;
        case shape::Relation::Less:
            return lhs < rhs;
        case shape::Relation::LessEqual:
            return lhs <= rhs;
        case shape::Relation::Greater:
            return lhs > rhs;
        case shape::Relation::GreaterEqual:
            return lhs >= rhs;
        }
        return false;
    }

    KernelTarget& kernel_target() {
        auto* kernel = dynamic_cast<KernelTarget*>(&target_);
        if (kernel == nullptr) {
            fail("internal: a kernel's operation outside a kernel");
        }
        return *kernel;
    }

public:
    // A kernel's body with its generics `subst`, written by the kernel
    // target this evaluator writes to.
    KernelProgram run_kernel(const ir::Function& kernel, const Substitution& subst) {
        KernelTarget& writer = kernel_target();
        Frame frame;
        frame.subst = subst;
        const ir::Block& body = module_.block(module_.region(kernel.body).blocks.front());
        const std::size_t results = body.arguments.size() - kernel.outputs;
        for (std::size_t i = 0; i < body.arguments.size(); ++i) {
            const ir::Value& argument = module_.value(body.arguments[i]);
            const TypeData& data = types_.get(types_.substitute(argument.type, subst));
            const ScalarKind dtype = concrete_dtype(data.dtype);
            if (data.kind == TypeKind::Tensor) {
                const Dims shape = concrete_shape(data.shape);
                frame.values[body.arguments[i]] =
                    tensor(writer.memory(argument.name, shape, dtype, i >= results), shape, dtype);
            } else {
                frame.values[body.arguments[i]] =
                    tensor(writer.scalar(argument.name, dtype), {}, dtype);
            }
        }
        frames_.push_back(std::move(frame));
        run_block(body);
        frames_.pop_back();
        return writer.program();
    }

private:
    // A region run as a call's: a frame of its own, outside any grid.
    std::vector<Val> run_call_region(ir::RegionId region,
                                     const Substitution& subst,
                                     const std::vector<Val>& arguments) {
        Frame inner;
        inner.subst = subst;
        const Dims saved_grid = grid_;
        grid_.clear();
        frames_.push_back(std::move(inner));
        std::vector<Val> results = run_region(region, arguments);
        frames_.pop_back();
        grid_ = saved_grid;
        return results;
    }

    // ------------------------------------------------- index notation

    std::vector<Val> index_arguments(const ir::Operation& op) {
        const ir::Block& body = module_.block(module_.region(op.regions.front()).blocks.front());
        std::vector<Val> arguments;
        for (std::size_t i = 0; i < body.arguments.size(); ++i) {
            const TypeData& data =
                types_.get(types_.substitute(module_.value(body.arguments[i]).type, frame().subst));
            Val index;
            if (data.kind == TypeKind::ShapeValue) {
                index.kind = Val::Kind::Pack;
                for (const std::int64_t dim : concrete_shape(data.shape)) {
                    index.axes.push_back(grid_.size());
                    grid_.push_back(dim);
                }
            } else {
                index.kind = Val::Kind::Index;
                const Shape unit = types_.substitute(Shape{op.attributes.shape[i]}, frame().subst);
                const Dims dims = concrete_shape(unit);
                if (dims.size() != 1) {
                    fail("an index domain must be one dimension");
                }
                index.axes.push_back(grid_.size());
                grid_.push_back(dims.front());
            }
            arguments.push_back(index);
        }
        return arguments;
    }

    Val comprehension(const ir::Operation& op) {
        if (in_grid()) {
            fail("internal: comprehension inside a region");
        }
        const std::vector<Val> arguments = index_arguments(op);
        Val result = to_grid(run_region(op.regions.front(), arguments).front());
        result = convert(result, result_dtype(op));
        grid_.clear();
        result.grid_rank = 0;
        return result;
    }

    // One factor of a contraction: a whole tensor, and the grid axis each of
    // its own axes runs along.
    struct Factor {
        TensorInfo tensor;
        Dims axes;
    };

    // `sum[k] a[i, k] * b[k, j]`, recognized before it is evaluated. Taken
    // literally over the grid it builds the whole product `[i, j, k]` and then
    // sums it: 51 GB for SAM's relative-position term, 85 GiB for a
    // mixture-of-experts projection at 160 positions. As a contraction the
    // target never materializes it (`einsum`, `dot_general`, ONNX `Einsum`).
    //
    // The body must be the product of two element reads indexed directly by
    // grid positions (each axis once), each optionally cast, in the dtype the
    // sum accumulates in. Anything else -- a gather, a third factor, a
    // product in a narrower type than its sum -- is not a plain contraction
    // and takes the general path. Nothing is emitted before the match is
    // certain, so falling back costs nothing.
    std::optional<Val> try_contract(const ir::Operation& op) {
        if (op.attributes.reduce != ir::ReduceKind::Sum) {
            return std::nullopt;
        }
        const ir::Region& region = module_.region(op.regions.front());
        if (region.blocks.size() != 1) {
            return std::nullopt;
        }
        const ir::Block& body = module_.block(region.blocks.front());
        // Pass one: the shape of the body, from op kinds alone.
        std::set<ir::ValueId> params;
        std::map<ir::ValueId, ir::ValueId> element_of; // cast/element result -> element result
        std::optional<ir::ValueId> product;
        std::optional<std::pair<ir::ValueId, ir::ValueId>> factors;
        for (const ir::OpId id : body.ops) {
            const ir::Operation& inner = module_.op(id);
            switch (inner.kind) {
            case ir::OpKind::BlockParam:
                params.insert(inner.results.front());
                break;
            case ir::OpKind::Element:
                element_of[inner.results.front()] = inner.results.front();
                break;
            case ir::OpKind::Cast: {
                const auto found = element_of.find(inner.operands.front());
                if (found == element_of.end()) {
                    return std::nullopt;
                }
                element_of[inner.results.front()] = found->second;
                break;
            }
            case ir::OpKind::Mul:
                if (product || inner.operands.size() != 2 ||
                    !element_of.contains(inner.operands[0]) ||
                    !element_of.contains(inner.operands[1])) {
                    return std::nullopt;
                }
                product = inner.results.front();
                factors = {inner.operands[0], inner.operands[1]};
                break;
            case ir::OpKind::Yield:
                if (!product || inner.operands.size() != 1 || inner.operands.front() != *product) {
                    return std::nullopt;
                }
                break;
            default:
                return std::nullopt;
            }
        }
        if (!factors) {
            return std::nullopt;
        }
        // Pass two: bind the grid and read each factor's tensor and axes.
        const Dims outer = grid_;
        const std::vector<Val> arguments = index_arguments(op);
        for (std::size_t i = 0; i < arguments.size() && i < body.arguments.size(); ++i) {
            frame().values[body.arguments[i]] = arguments[i];
        }
        const auto give_up = [&]() -> std::optional<Val> {
            grid_ = outer;
            return std::nullopt;
        };
        const ScalarKind dtype = result_dtype(op);
        std::vector<Factor> read;
        for (const ir::ValueId factor : {factors->first, factors->second}) {
            const ir::Operation& element =
                module_.op(module_.value(element_of.at(factor)).producer);
            const ir::ValueId source_id = element.operands.front();
            if (params.contains(source_id)) {
                run_op(module_.op(module_.value(source_id).producer));
            }
            const Val& source = value(source_id);
            if (source.kind != Val::Kind::Tensor || source.grid_rank != 0) {
                return give_up();
            }
            Dims axes;
            for (std::size_t i = 1; i < element.operands.size(); ++i) {
                // An index computed in the body (a gather: `t[idx[b, k], h]`)
                // has no value yet, and is not a grid axis anyway.
                if (!frame().values.contains(element.operands[i])) {
                    return give_up();
                }
                const Val& index = value(element.operands[i]);
                if (index.kind != Val::Kind::Index && index.kind != Val::Kind::Pack) {
                    return give_up();
                }
                for (const std::size_t axis : index.axes) {
                    axes.push_back(static_cast<std::int64_t>(axis));
                }
            }
            Dims sorted = axes;
            std::sort(sorted.begin(), sorted.end());
            if (axes.size() != source.shape.size() ||
                std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
                return give_up();
            }
            // The factor's own dtype: the element read's, or its cast's.
            const ScalarKind factor_dtype =
                tensor_value(module_.value(factor).type, frame().subst).dtype;
            if (factor_dtype != dtype) {
                return give_up(); // a product narrower than its sum rounds first
            }
            TensorInfo tensor = info(source);
            if (tensor.dtype != factor_dtype) {
                tensor = {target_.convert(tensor, factor_dtype), tensor.shape, factor_dtype};
            }
            read.push_back({tensor, axes});
        }
        Dims out_axes;
        for (std::size_t axis = 0; axis < outer.size(); ++axis) {
            const auto uses = [&](const Factor& f) {
                return std::find(f.axes.begin(), f.axes.end(), static_cast<std::int64_t>(axis)) !=
                       f.axes.end();
            };
            if (!uses(read[0]) && !uses(read[1])) {
                return give_up(); // an output axis neither factor spans
            }
            out_axes.push_back(static_cast<std::int64_t>(axis));
        }
        const auto name = target_.contract(
            read[0].tensor, read[0].axes, read[1].tensor, read[1].axes, out_axes, outer, dtype);
        grid_ = outer;
        if (!name) {
            return std::nullopt;
        }
        return tensor(*name, outer, dtype);
    }

    Val reduce(const ir::Operation& op) {
        if (auto contracted = try_contract(op)) {
            return *contracted;
        }
        const Dims outer = grid_;
        const std::vector<Val> arguments = index_arguments(op);
        Val body = to_grid(run_region(op.regions.front(), arguments).front());
        const ScalarKind dtype = result_dtype(op);
        body = convert(body, dtype);
        Dims dimensions;
        for (std::size_t i = outer.size(); i < grid_.size(); ++i) {
            dimensions.push_back(static_cast<std::int64_t>(i));
        }
        grid_ = outer;
        const Reduction kind = op.attributes.reduce;
        return tensor(target_.reduce(kind, info(body), dimensions, outer), outer, dtype);
    }

    // `x[i, j]` over the grid: a broadcast when every index is a distinct
    // grid position, a gather otherwise.
    Val element(const ir::Operation& op) {
        const Val& source = value(op.operands.front());
        if (source.kind != Val::Kind::Tensor) {
            fail("internal: element access on a non-tensor");
        }
        Dims axes;
        bool is_positional = true;
        for (std::size_t i = 1; i < op.operands.size(); ++i) {
            const Val& index = value(op.operands[i]);
            if (index.kind == Val::Kind::Index || index.kind == Val::Kind::Pack) {
                for (const std::size_t axis : index.axes) {
                    axes.push_back(static_cast<std::int64_t>(axis));
                }
            } else {
                is_positional = false;
            }
        }
        Dims sorted = axes;
        std::sort(sorted.begin(), sorted.end());
        is_positional = is_positional && axes.size() == source.shape.size() &&
                        std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
        if (is_positional) {
            bool fits = true;
            for (std::size_t i = 0; i < axes.size(); ++i) {
                fits = fits && grid_[static_cast<std::size_t>(axes[i])] == source.shape[i];
            }
            if (fits) {
                // The grid's own leading axes, in order: the tensor as it
                // is, broadcast only where something needs the whole grid
                // (`to_grid`); an index stays as small as it is.
                bool leading = true;
                for (std::size_t i = 0; i < axes.size(); ++i) {
                    leading = leading && axes[i] == static_cast<std::int64_t>(i);
                }
                if (leading && source.shape.size() < grid_.size()) {
                    Val lazy = source;
                    lazy.grid_rank = source.shape.size();
                    return lazy;
                }
                return broadcast(source, grid_, axes);
            }
        }
        if (auto partial = gather_leading(op, source)) {
            return *partial;
        }
        // General case: one i64 position per operand axis, gathered.
        std::vector<TensorInfo> columns;
        for (std::size_t i = 1; i < op.operands.size(); ++i) {
            const Val& index = value(op.operands[i]);
            std::vector<Val> positions;
            if (index.kind == Val::Kind::Pack) {
                for (const std::size_t axis : index.axes) {
                    positions.push_back(iota(axis));
                }
            } else {
                positions.push_back(convert(to_grid(index), ScalarKind::I64));
            }
            for (const Val& position : positions) {
                Dims column_shape = grid_;
                column_shape.push_back(1);
                columns.push_back(info(reshape(position, column_shape)));
            }
        }
        if (columns.size() != source.shape.size()) {
            fail("internal: element access with the wrong number of indices");
        }
        Dims indices_shape = grid_;
        indices_shape.push_back(static_cast<std::int64_t>(columns.size()));
        const TensorInfo indices =
            columns.size() == 1
                ? columns.front()
                : TensorInfo{target_.concat(
                                 columns, static_cast<std::int64_t>(grid_.size()), indices_shape),
                             indices_shape,
                             ScalarKind::I64};
        return tensor(target_.gather(info(source), indices, grid_), grid_, source.dtype);
    }

    // `experts[top[k], o, i]`: when the trailing indices are the grid's own
    // trailing positions over whole source axes, only the leading axes are
    // gathered, by indices over the leading grid, and the rest taken whole.
    // The general case would build an index for every element of the
    // result: for a mixture of experts, a [TopK, Out, In, 3] tensor per
    // layer per step.
    std::optional<Val> gather_leading(const ir::Operation& op, const Val& source) {
        const std::size_t rank = source.shape.size();
        if (op.operands.size() != rank + 1) {
            return std::nullopt;
        }
        std::size_t tail = 0;
        while (tail < rank && tail < grid_.size()) {
            const Val& index = value(op.operands[rank - tail]);
            const std::size_t axis = grid_.size() - 1 - tail;
            if (index.kind != Val::Kind::Index || index.axes.front() != axis ||
                grid_[axis] != source.shape[rank - 1 - tail]) {
                break;
            }
            ++tail;
        }
        const std::size_t gathered = rank - tail;
        if (tail == 0 || gathered == 0) {
            return std::nullopt;
        }
        const std::size_t prefix = grid_.size() - tail;
        for (std::size_t i = 1; i <= gathered; ++i) {
            const Val& index = value(op.operands[i]);
            const bool fits = (index.kind == Val::Kind::Index && index.axes.front() < prefix) ||
                              (index.kind == Val::Kind::Tensor && index.shape.size() <= prefix &&
                               index.grid_rank >= index.shape.size());
            if (!fits) {
                return std::nullopt;
            }
        }
        const Dims whole = grid_;
        grid_ = Dims(whole.begin(), whole.begin() + static_cast<std::ptrdiff_t>(prefix));
        std::vector<TensorInfo> columns;
        for (std::size_t i = 1; i <= gathered; ++i) {
            const Val& index = value(op.operands[i]);
            const Val position = index.kind == Val::Kind::Index
                                     ? iota(index.axes.front())
                                     : convert(to_grid(index), ScalarKind::I64);
            Dims column_shape = grid_;
            column_shape.push_back(1);
            columns.push_back(info(reshape(position, column_shape)));
        }
        Dims indices_shape = grid_;
        indices_shape.push_back(static_cast<std::int64_t>(gathered));
        const TensorInfo indices =
            columns.size() == 1
                ? columns.front()
                : TensorInfo{target_.concat(
                                 columns, static_cast<std::int64_t>(grid_.size()), indices_shape),
                             indices_shape,
                             ScalarKind::I64};
        grid_ = whole;
        return tensor(target_.gather(info(source), indices, grid_), grid_, source.dtype);
    }

    ir::Module& module_;
    const Model& model_;
    TypeStore& types_;
    const GraphExportOptions& options_;
    GraphTarget& target_;
    std::map<std::string, const ir::Function*> functions_;
    std::map<std::string, Val> parameters_;
    std::set<std::string> absent_subs_;       // optional subs left out, by path
    std::map<std::string, Val> state_types_;  // state members by path, unnamed
    std::map<std::string, Val> states_;       // current state values by path
    std::vector<std::string> written_states_; // assigned states in first-write order
    std::deque<Frame> frames_;
    Dims grid_;
};

} // namespace

std::expected<std::string, std::string>
export_graph(ir::Module& module, const GraphExportOptions& options, GraphTarget& target) {
    try {
        if (!options.gradient) {
            return Evaluator(module, options, target).run();
        }
        if (!options.placement.empty() || !options.offload.empty() ||
            !options.fully_shard.empty() || options.prepare) {
            return std::unexpected("a gradient export runs on one device, with every parameter "
                                   "an argument: not with placement, offload, sharding or "
                                   "--prepare");
        }
        GradientTarget gradient(target);
        return Evaluator(module, options, gradient).run();
    } catch (const Unsupported& error) {
        return std::unexpected(error.what());
    } catch (const GradientError& error) {
        return std::unexpected(std::string("gradient: ") + error.what());
    } catch (const KernelError& error) {
        return std::unexpected(std::string("kernel: ") + error.what());
    }
}

bool word_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool numbered(std::string_view word, char prefix) {
    return word.size() > 1 && word[0] == prefix &&
           word.find_first_not_of("0123456789", 1) == std::string_view::npos;
}

std::vector<std::pair<std::size_t, std::string>> words_of(std::string_view text) {
    std::vector<std::pair<std::size_t, std::string>> words;
    std::size_t i = 0;
    while (i < text.size()) {
        if (word_char(text[i]) && (i == 0 || !word_char(text[i - 1]))) {
            std::size_t j = i;
            while (j < text.size() && word_char(text[j])) {
                ++j;
            }
            if (std::isdigit(static_cast<unsigned char>(text[i])) == 0) {
                words.emplace_back(i, text.substr(i, j - i));
            }
            i = j;
            continue;
        }
        ++i;
    }
    return words;
}

std::set<std::string> value_names(std::string_view text) {
    std::set<std::string> names;
    for (auto& [at, word] : words_of(text)) {
        if (numbered(word, 'v')) {
            names.insert(std::move(word));
        }
    }
    return names;
}

namespace {

// The `vN` names a generated line mentions on its right-hand side.
std::set<std::string> mentioned_values(const std::string& line) {
    const std::size_t assignment = line.find(" = ");
    return value_names(assignment == std::string::npos
                           ? std::string_view(line)
                           : std::string_view(line).substr(assignment + 3));
}

} // namespace

bool glob_match(std::string_view pattern, std::string_view text) {
    std::size_t p = 0;
    std::size_t t = 0;
    std::size_t star = std::string_view::npos;
    std::size_t resume = 0;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
            ++p;
            ++t;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            resume = t;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            t = ++resume;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

std::string release_dead_values(const std::string& body, const std::string& live_tail) {
    std::vector<std::string> lines;
    std::string current;
    for (const char c : body) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    const auto indent = [](const std::string& line) {
        const std::size_t first = line.find_first_not_of(' ');
        return first == std::string::npos ? line.size() : first;
    };
    // A top-level statement and the lines nested under it.
    std::vector<std::pair<std::size_t, std::size_t>> groups;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (groups.empty() || indent(lines[i]) <= 4) {
            groups.emplace_back(i, i);
        } else {
            groups.back().second = i;
        }
    }
    std::set<std::string> assigned; // by a top-level `vN = ...`
    std::set<std::string> released; // already by a `del` in the body
    for (const auto& [first, last] : groups) {
        const std::string& line = lines[first];
        const std::size_t start = indent(line);
        const std::size_t equals = line.find(" = ", start);
        if (start == 4 && line[start] == 'v' && equals != std::string::npos &&
            line.find_first_not_of("0123456789", start + 1) == equals) {
            assigned.insert(line.substr(start, equals - start));
        }
        if (line.compare(start, 4, "del ") == 0) {
            for (const std::string& name : mentioned_values(line)) {
                released.insert(name);
            }
        }
        (void)last;
    }
    std::set<std::string> live = mentioned_values(live_tail);
    std::vector<std::vector<std::string>> after(groups.size());
    for (std::size_t g = groups.size(); g-- > 0;) {
        std::set<std::string> uses;
        for (std::size_t i = groups[g].first; i <= groups[g].second; ++i) {
            for (const std::string& name : mentioned_values(lines[i])) {
                uses.insert(name);
            }
        }
        for (const std::string& name : uses) {
            if (!live.contains(name) && assigned.contains(name) && !released.contains(name)) {
                after[g].push_back(name);
            }
            live.insert(name);
        }
    }
    std::string out;
    for (std::size_t g = 0; g < groups.size(); ++g) {
        for (std::size_t i = groups[g].first; i <= groups[g].second; ++i) {
            out += lines[i] + "\n";
        }
        if (!after[g].empty()) {
            std::string line = "    del ";
            line += join(after[g], ", ");
            out += line + "\n";
        }
    }
    return out;
}

namespace {

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::string current;
    for (const char c : text) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        lines.push_back(current);
    }
    return lines;
}

// `name = expr` of a top-level generated assignment, if `line` is one.
std::optional<std::pair<std::string, std::string>> top_level_assignment(const std::string& line) {
    if (line.size() < 6 || line.compare(0, 4, "    ") != 0 || line[4] != 'v') {
        return std::nullopt;
    }
    const std::size_t equals = line.find(" = ", 4);
    if (equals == std::string::npos) {
        return std::nullopt;
    }
    const std::string name = line.substr(4, equals - 4);
    if (!numbered(name, 'v')) {
        return std::nullopt;
    }
    return std::pair{name, line.substr(equals + 3)};
}

// A parameter or value seen through views and casts only: `p3.permute(1, 0)`,
// `v9.float()`, `p1.reshape(4, 5).to(torch.bfloat16)`, `p2[0]`.
// Whether the words of `arguments` are only shapes, axes, and dtypes.
bool literal_arguments(const std::string& arguments) {
    static const std::set<std::string> argument_words{"torch",
                                                      "jnp",
                                                      "float32",
                                                      "float16",
                                                      "bfloat16",
                                                      "float64",
                                                      "int32",
                                                      "int64",
                                                      "bool",
                                                      "None",
                                                      "dtype"};
    for (const auto& [at, word] : words_of(arguments)) {
        if (!argument_words.contains(word)) {
            return false;
        }
    }
    return true;
}

// Whether `expression` is a chain of `methods` calls on one value, their
// arguments shapes, axes, and dtypes: `v3.reshape((2, 4)).t()` as PyTorch
// writes it, `jnp.transpose(v3, (1, 0))` as JAX does.
bool method_chain(const std::string& expression, const std::set<std::string>& methods) {
    std::size_t i = 0;
    if (expression.starts_with("jnp.")) {
        std::size_t j = 4;
        while (j < expression.size() && word_char(expression[j])) {
            ++j;
        }
        if (!methods.contains(expression.substr(4, j - 4)) || j >= expression.size() ||
            expression[j] != '(') {
            return false;
        }
        // The call's first argument is the value; the rest are literals.
        int depth = 0;
        std::size_t comma = std::string::npos;
        std::size_t close = std::string::npos;
        for (std::size_t k = j; k < expression.size(); ++k) {
            const char c = expression[k];
            if (c == '(' || c == '[') {
                ++depth;
            } else if (c == ')' || c == ']') {
                if (--depth == 0) {
                    close = k;
                    break;
                }
            } else if (c == ',' && depth == 1 && comma == std::string::npos) {
                comma = k;
            }
        }
        if (close == std::string::npos) {
            return false;
        }
        const std::size_t end = comma == std::string::npos ? close : comma;
        std::string first = expression.substr(j + 1, end - j - 1);
        while (!first.empty() && first.front() == ' ') {
            first.erase(first.begin());
        }
        while (!first.empty() && first.back() == ' ') {
            first.pop_back();
        }
        if (!method_chain(first, methods) ||
            (comma != std::string::npos &&
             !literal_arguments(expression.substr(comma, close - comma)))) {
            return false;
        }
        i = close + 1;
    } else {
        while (i < expression.size() && word_char(expression[i])) {
            ++i;
        }
        const std::string head = expression.substr(0, i);
        if (!numbered(head, 'v') && !numbered(head, 'p')) {
            return false;
        }
    }
    while (i < expression.size()) {
        if (expression[i] == '[') {
            const std::size_t close = expression.find(']', i);
            if (close == std::string::npos) {
                return false;
            }
            for (const auto& [at, word] : words_of(expression.substr(i, close - i))) {
                if (word != "None") {
                    return false; // an index computed from something
                }
            }
            i = close + 1;
            continue;
        }
        if (expression[i] != '.') {
            return false;
        }
        std::size_t j = i + 1;
        while (j < expression.size() && word_char(expression[j])) {
            ++j;
        }
        if (!methods.contains(expression.substr(i + 1, j - i - 1)) || j >= expression.size() ||
            expression[j] != '(') {
            return false;
        }
        std::size_t close = j;
        int depth = 0;
        do {
            depth += expression[close] == '(' ? 1 : expression[close] == ')' ? -1 : 0;
            ++close;
        } while (close < expression.size() && depth > 0);
        if (depth != 0 || !literal_arguments(expression.substr(j, close - j))) {
            return false;
        }
        i = close;
    }
    return true;
}

// A view of one value: the same memory, read another way.
bool pure_view(const std::string& expression) {
    static const std::set<std::string> methods{"t",
                                               "permute",
                                               "reshape",
                                               "view",
                                               "transpose",
                                               "expand",
                                               "unsqueeze",
                                               "squeeze",
                                               "flatten",
                                               "broadcast_to",
                                               "expand_dims",
                                               "swapaxes",
                                               "moveaxis"};
    return method_chain(expression, methods);
}

bool view_or_cast(const std::string& expression) {
    static const std::set<std::string> methods{
        "float",   "half",    "bfloat16",     "contiguous",  "t",         "to",      "permute",
        "reshape", "view",    "transpose",    "expand",      "unsqueeze", "squeeze", "flatten",
        "astype",  "asarray", "broadcast_to", "expand_dims", "swapaxes",  "moveaxis"};
    return method_chain(expression, methods);
}

std::string fnv1a(const std::string& text) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const char c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    static const char* digits = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[static_cast<std::size_t>(i)] = digits[hash & 15U];
        hash >>= 4U;
    }
    return out;
}

} // namespace

PreparedSplit split_prepared(const std::string& body,
                             const std::string& live_tail,
                             const std::vector<std::string>& parameter_paths,
                             const std::string& constants) {
    PreparedSplit split;
    const std::vector<std::string> lines = split_lines(body);
    // Constants the target hoisted: `name -> expression`, in order.
    std::vector<std::pair<std::string, std::string>> constant_lines;
    std::set<std::string> constant_names;
    for (const std::string& line : split_lines(constants)) {
        if (const auto assignment = top_level_assignment(line)) {
            constant_lines.push_back(*assignment);
            constant_names.insert(assignment->first);
        }
    }
    const auto reads = [](const std::string& expression) {
        std::vector<std::string> names;
        for (const auto& [at, word] : words_of(expression)) {
            if (numbered(word, 'v') || numbered(word, 'p')) {
                names.push_back(word);
            }
        }
        return names;
    };
    // Pass one: the weight-only lines, and which of them compute something.
    std::map<std::string, std::size_t> defined_at; // prepared name -> line
    std::map<std::string, bool> computes;
    // Whether a value reads a parameter at all: one that reads only literals
    // is a constant, which the target folds or hoists itself.
    std::map<std::string, bool> reads_weights;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto assignment = top_level_assignment(lines[i]);
        if (!assignment) {
            continue;
        }
        const auto& [name, expression] = *assignment;
        bool weight_only = true;
        bool heavy = !view_or_cast(expression);
        bool weights = false;
        for (const auto& [at, word] : words_of(expression)) {
            if (numbered(word, 'p')) {
                weights = true;
                continue;
            }
            if (numbered(word, 'v')) {
                if (defined_at.contains(word)) {
                    heavy = heavy || computes[word];
                    weights = weights || reads_weights[word];
                } else if (!constant_names.contains(word)) {
                    weight_only = false;
                }
                continue;
            }
            if (numbered(word, 's') || numbered(word, 'w') || word.starts_with("in_") ||
                word == "_dev") {
                weight_only = false;
            }
        }
        if (weight_only) {
            defined_at[name] = i;
            computes[name] = heavy;
            reads_weights[name] = weights;
        }
    }
    // Pass two: keep what computes something and is read outside, with
    // everything it reads; the rest goes back to the body where it was.
    std::set<std::string> read_outside = mentioned_values(live_tail);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto assignment = top_level_assignment(lines[i]);
        if (assignment && defined_at.contains(assignment->first)) {
            continue;
        }
        for (const std::string& name : reads(lines[i])) {
            read_outside.insert(name);
        }
    }
    std::set<std::string> keep;
    std::vector<std::string> pending;
    for (const auto& [name, line] : defined_at) {
        if (read_outside.contains(name) && computes[name] && reads_weights[name]) {
            // A view of a prepared value stays in the body: the value is
            // what is kept, so entries that read it another way (flattened
            // for one, whole for another) share it.
            std::string root = name;
            while (true) {
                const std::string expression =
                    top_level_assignment(lines[defined_at[root]])->second;
                const std::vector<std::string> read = reads(expression);
                if (!pure_view(expression) || read.size() != 1 || !defined_at.contains(read[0])) {
                    break;
                }
                root = read[0];
            }
            pending.push_back(root);
        }
    }
    while (!pending.empty()) {
        const std::string name = pending.back();
        pending.pop_back();
        if (!keep.insert(name).second) {
            continue;
        }
        for (const std::string& read :
             reads(top_level_assignment(lines[defined_at[name]])->second)) {
            if (defined_at.contains(read)) {
                pending.push_back(read);
            }
        }
    }
    std::vector<std::string> body_lines;
    std::vector<std::string> kept_lines;
    for (const std::string& line : lines) {
        const auto assignment = top_level_assignment(line);
        if (assignment && keep.contains(assignment->first)) {
            kept_lines.push_back(line);
            // A constant both sides read is made on both: as a prepared
            // value it would key one entry's set apart from another's.
            if (!reads_weights[assignment->first]) {
                body_lines.push_back(line);
            }
        } else {
            body_lines.push_back(line);
        }
    }
    std::set<std::string> read_by_body = mentioned_values(live_tail);
    for (const std::string& line : body_lines) {
        for (const std::string& name : reads(line)) {
            read_by_body.insert(name);
        }
    }
    for (const std::string& line : kept_lines) {
        const std::string name = top_level_assignment(line)->first;
        if (read_by_body.contains(name) && reads_weights[name]) {
            split.outputs.push_back(name);
        }
    }
    for (const std::string& line : body_lines) {
        split.body += line + "\n";
    }
    if (split.outputs.empty()) {
        split.body = body;
        split.outputs.clear();
        return split;
    }
    std::set<std::string> seen_inputs;
    for (const std::string& line : kept_lines) {
        split.prepare += line + "\n";
        for (const std::string& name : reads(top_level_assignment(line)->second)) {
            if ((numbered(name, 'p') || constant_names.contains(name)) &&
                seen_inputs.insert(name).second) {
                split.inputs.push_back(name);
            }
        }
    }
    // Keys: each output's computation over parameter paths, its values
    // renamed in order of definition, so equal computations hash equal.
    std::map<std::string, std::string> expressions;
    for (const auto& [name, expression] : constant_lines) {
        expressions[name] = expression;
    }
    for (const std::string& line : kept_lines) {
        const auto assignment = top_level_assignment(line);
        expressions[assignment->first] = assignment->second;
    }
    for (const std::string& output : split.outputs) {
        std::vector<std::string> order; // definitions, dependencies first
        std::set<std::string> visited;
        std::function<void(const std::string&)> visit = [&](const std::string& name) {
            if (!visited.insert(name).second) {
                return;
            }
            for (const std::string& read : reads(expressions[name])) {
                if (expressions.contains(read)) {
                    visit(read);
                }
            }
            order.push_back(name);
        };
        visit(output);
        std::map<std::string, std::string> renamed;
        for (const std::string& name : order) {
            renamed[name] = "t" + std::to_string(renamed.size());
        }
        std::string canonical;
        for (const std::string& name : order) {
            const std::string& expression = expressions[name];
            std::string text;
            std::size_t last = 0;
            for (const auto& [at, word] : words_of(expression)) {
                text += expression.substr(last, at - last);
                if (renamed.contains(word)) {
                    text += renamed[word];
                } else if (numbered(word, 'p')) {
                    const std::size_t index = std::stoul(word.substr(1));
                    text += "P<" +
                            (index < parameter_paths.size() ? parameter_paths[index] : word) + ">";
                } else {
                    text += word;
                }
                last = at + word.size();
            }
            text += expression.substr(last);
            canonical += renamed[name] + " = " + text + "\n";
        }
        split.keys.push_back(fnv1a(canonical + "-> " + renamed[output]));
    }
    return split;
}

std::string einsum_equation(const Dims& lhs_axes, const Dims& rhs_axes, const Dims& out_axes) {
    std::map<std::int64_t, char> letters;
    const auto word = [&](const Dims& axes) {
        std::string out;
        for (const std::int64_t axis : axes) {
            const auto found = letters.find(axis);
            if (found != letters.end()) {
                out += found->second;
                continue;
            }
            const char letter = static_cast<char>('a' + letters.size());
            letters.emplace(axis, letter);
            out += letter;
        }
        return out;
    };
    const std::string lhs = word(lhs_axes);
    const std::string rhs = word(rhs_axes);
    return lhs + "," + rhs + "->" + word(out_axes);
}

std::string prune_python_assignments(const std::string& body, const std::string& live_tail) {
    std::vector<std::string> lines;
    std::string current;
    for (const char c : body) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    std::set<std::string> live = mentioned_values(live_tail);
    std::vector<bool> keep(lines.size(), true);
    for (std::size_t i = lines.size(); i-- > 0;) {
        const std::string& line = lines[i];
        const std::size_t start = line.find_first_not_of(' ');
        if (start != std::string::npos && line[start] == 'v') {
            const std::size_t end = line.find(" = ", start);
            if (end != std::string::npos &&
                line.find_first_not_of("0123456789", start + 1) == end &&
                !live.contains(line.substr(start, end - start))) {
                keep[i] = false;
                continue;
            }
        }
        for (const std::string& name : mentioned_values(line)) {
            live.insert(name);
        }
    }
    std::string out;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (keep[i]) {
            out += lines[i] + "\n";
        }
    }
    return out;
}

std::unique_ptr<KernelTarget> GraphTarget::kernel_target() const {
    return nullptr;
}

std::optional<std::vector<std::string>> GraphTarget::launch_kernel(const KernelLaunch& launch) {
    (void)launch;
    return std::nullopt;
}

std::string python_tuple(const Dims& dims) {
    std::string out = "(";
    out += join(dims, ", ", [&](const auto& item) { return std::to_string(item); });
    return out + (dims.size() == 1 ? ",)" : ")");
}

const DTypeNames& dtype_names(sema::ScalarKind dtype) {
    // In `ScalarKind` order.
    static constexpr std::array<DTypeNames, 13> names{{
        {"i1", "bool", 9, "bool"},
        {"i8", "int8", 3, "int8"},
        {"i16", "int16", 5, "int16"},
        {"i32", "int32", 6, "int32"},
        {"i64", "int64", 7, "int64"},
        {"ui8", "uint8", 2, "uint8"},
        {"ui16", "uint16", 4, "uint16"},
        {"ui32", "uint32", 12, "uint32"},
        {"ui64", "uint64", 13, "uint64"},
        {"f16", "float16", 10, "float16"},
        {"bf16", "bfloat16", 16, "bfloat16"},
        {"f32", "float", 1, "float32"},
        {"f64", "double", 11, "float64"},
    }};
    return names[static_cast<std::size_t>(dtype)];
}

TensorInfo GraphTarget::place_axes(const TensorInfo& value, const Dims& dims, const Dims& shape) {
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
    return source;
}

CallName call_name(std::string_view implementation) {
    CallName name;
    const auto strip = [&](std::string_view suffix) {
        const bool found = implementation.ends_with(suffix);
        if (found) {
            implementation.remove_suffix(suffix.size());
        }
        return found;
    };
    name.fast = strip("(input dtype)");
    name.grouped = strip("(enable_gqa)");
    name.base = std::string(implementation);
    return name;
}

std::vector<const TensorInfo*>
operand_pointers(const std::vector<std::optional<TensorInfo>>& operands) {
    std::vector<const TensorInfo*> at;
    at.reserve(operands.size());
    for (const std::optional<TensorInfo>& operand : operands) {
        at.push_back(operand.has_value() ? &*operand : nullptr);
    }
    return at;
}

bool declares_entry(const ir::Module& module,
                    std::uint32_t root_module,
                    sema::EntityId owner,
                    std::string_view name) {
    const sema::Model& model = module.model();
    return std::ranges::any_of(module.functions(), [&](const ir::Function& function) {
        const sema::Entity& entity = model.entities[function.entity];
        return function.is_entry && entity.parent == owner && entity.module == root_module &&
               (name.empty() || entity.name == name);
    });
}

std::expected<sema::EntityId, std::string> find_root_block(const ir::Module& module,
                                                           std::uint32_t root_module,
                                                           std::string_view root,
                                                           std::string_view no_entry_hint) {
    const sema::Model& model = module.model();
    std::vector<sema::EntityId> candidates;
    for (sema::EntityId id = 0; id < model.entities.size(); ++id) {
        const sema::Entity& entity = model.entities[id];
        if (entity.kind != sema::EntityKind::Block || entity.parent != sema::no_entity ||
            entity.module != root_module) {
            continue;
        }
        if (!root.empty()) {
            if (entity.name == root) {
                return id;
            }
            continue;
        }
        if (declares_entry(module, root_module, id)) {
            candidates.push_back(id);
        }
    }
    if (!root.empty()) {
        return std::unexpected("no block named `" + std::string(root) + "` in this file");
    }
    if (candidates.size() != 1) {
        return std::unexpected(
            candidates.empty() ? "no block with an `entry`; " + std::string(no_entry_hint)
                               : std::string("several blocks have entries; name one with --root"));
    }
    return candidates.front();
}

std::string python_float(double value) {
    return shortest_float(value);
}

} // namespace linnet::backend
