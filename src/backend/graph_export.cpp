#include "linnet/backend/graph_export.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <deque>
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
        const EntityId root = find_root();
        const ir::Function& entry = find_entry(root);
        const Substitution root_subst = root_bindings(root);

        Val self;
        self.kind = Val::Kind::Block;
        self.block = root;
        self.subst = root_subst;

        if (placing()) {
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
        frame.values[body.arguments.front()] = self;
        for (std::size_t i = 1; i < body.arguments.size(); ++i) {
            const ir::ValueId argument = body.arguments[i];
            Val value = tensor_value(module_.value(argument).type, frame.subst);
            value.name = target_.input(module_.value(argument).name, value.shape, value.dtype);
            frame.values[argument] = value;
        }
        collect_parameters(root, root_subst, "");
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
                              std::string(model_.entities[root].name),
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

    EntityId find_root() const {
        std::vector<EntityId> candidates;
        for (EntityId id = 0; id < model_.entities.size(); ++id) {
            const Entity& entity = model_.entities[id];
            if (entity.kind != EntityKind::Block || entity.parent != no_entity ||
                entity.module != options_.root_module) {
                continue;
            }
            if (!options_.root.empty()) {
                if (entity.name == options_.root) {
                    return id;
                }
                continue;
            }
            for (const ir::Function& function : module_.functions()) {
                if (function.is_entry && model_.entities[function.entity].parent == id) {
                    candidates.push_back(id);
                    break;
                }
            }
        }
        if (!options_.root.empty()) {
            fail("no block named `" + options_.root + "` in this file");
        }
        if (candidates.size() != 1) {
            fail(candidates.empty() ? "no block with an `entry`; name one with --root"
                                    : "several blocks have entries; name one with --root");
        }
        return candidates.front();
    }

    const ir::Function& find_entry(EntityId root) const {
        const ir::Function* found = nullptr;
        for (const ir::Function& function : module_.functions()) {
            const Entity& entity = model_.entities[function.entity];
            if (!function.is_entry || entity.parent != root) {
                continue;
            }
            if (!options_.entry.empty() ? entity.name == options_.entry : found == nullptr) {
                found = &function;
            } else if (options_.entry.empty()) {
                fail("the block has several entries; name one with --entry");
            }
        }
        if (found == nullptr) {
            fail(options_.entry.empty() ? "the block has no entry"
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
            case GenericKind::Pack:
                fail("shape-pack generics of the root block or entry are not supported");
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
        return subst;
    }

    Substitution entry_bindings(const ir::Function& entry, const Substitution& root_subst) {
        Substitution subst = root_subst;
        bind_generics(entry.generics, subst, "entry");
        return subst;
    }

    // A `while` loop: the target's loop form over the carried values — the
    // loop's own, then every state member, so writes inside the body flow
    // out of it — with the condition and body regions emitted into it.
    void run_while(const ir::Operation& op) {
        if (!target_.supports_while()) {
            fail("runtime `while` loops are not exported to this format yet; they run in the "
                 "PyTorch interpreter");
        }
        if (!grid_.empty()) {
            fail("a `while` loop inside index notation is not supported");
        }
        for (const auto& [path, declared] : state_types_) {
            if (!states_.contains(path)) {
                Val value = declared;
                value.name = target_.state(path, value.shape, value.dtype);
                states_.emplace(path, value);
            }
        }
        std::vector<Val> carried;
        for (const ir::ValueId id : op.operands) {
            const Val& carried_value = value(id);
            if (carried_value.kind != Val::Kind::Tensor || carried_value.grid_rank != 0) {
                fail("a `while` loop carries whole tensors only");
            }
            carried.push_back(carried_value);
        }
        std::vector<std::string> state_paths;
        for (const auto& [path, value] : states_) {
            state_paths.push_back(path);
            carried.push_back(value);
        }
        const std::size_t own = op.operands.size();
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
        const auto predicate = [&](const std::vector<Val>& arguments) {
            const std::vector<Val> results = run_region(op.regions.front(), arguments);
            if (results.size() != 1 || results.front().kind != Val::Kind::Tensor) {
                fail("a `while` condition must be one scalar");
            }
            return info(results.front());
        };
        if (target_.while_needs_initial_condition()) {
            target_.while_initial_condition(predicate(own_values(carried)));
        }
        const std::vector<Val> condition_values = renamed(target_.begin_while(initial));
        moved_.emplace_back();
        const std::vector<Val> body_values =
            renamed(target_.while_condition(predicate(own_values(condition_values))));
        std::vector<Val> next = run_region(op.regions.back(), own_values(body_values));
        if (next.size() != own) {
            fail("internal: a `while` body yielded " + std::to_string(next.size()) + " values");
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
        if (target_.while_needs_trailing_condition()) {
            target_.while_trailing_condition(predicate(own_values(next)));
        }
        std::vector<TensorInfo> outputs;
        outputs.reserve(next.size());
        for (const Val& value : next) {
            outputs.push_back(info(value));
        }
        const std::vector<std::string> finals = target_.end_while(outputs);
        moved_.pop_back();
        if (finals.size() != next.size()) {
            fail("internal: the target returned " + std::to_string(finals.size()) +
                 " loop results");
        }
        for (std::size_t i = 0; i < own; ++i) {
            Val value = next[i];
            value.name = finals[i];
            frame().values[op.results[i]] = value;
        }
        for (std::size_t j = 0; j < state_paths.size(); ++j) {
            Val value = next[own + j];
            value.name = finals[own + j];
            states_[state_paths[j]] = value;
        }
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
        const DeclInfo& decl = model_.decls.at(block);
        std::vector<std::pair<EntityId, std::string_view>> members;
        for (const auto& [name, entity] : decl.scope) {
            if (model_.entities[entity].kind == EntityKind::Member) {
                members.emplace_back(entity, name);
            }
        }
        std::sort(members.begin(), members.end());
        for (const auto& [entity, name] : members) {
            const TypeId type = types_.substitute(model_.entities[entity].type, subst);
            const TypeData& data = types_.get(type);
            const std::string path = prefix + std::string(name);
            if (data.kind == TypeKind::Block) {
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

    Substitution block_substitution(const TypeData& block_type) const {
        Substitution subst;
        const DeclInfo& decl = model_.decls.at(block_type.decl);
        for (std::size_t i = 0; i < decl.generics.size() && i < block_type.args.size(); ++i) {
            const GenericInfo& generic = decl.generics[i];
            switch (generic.kind) {
            case GenericKind::Dim:
                subst.dims[generic.symbol] = block_type.args[i].dim;
                break;
            case GenericKind::Pack:
                subst.packs[generic.symbol] = block_type.args[i].shape;
                break;
            case GenericKind::DType:
                subst.dtypes[generic.dtype_var] = block_type.args[i].dtype;
                break;
            }
        }
        return subst;
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
        Literal literal;
        if (dtype == ScalarKind::Bool) {
            literal.kind = Literal::Kind::Boolean;
            literal.integer = value != 0 ? 1 : 0;
        } else if (is_float(dtype)) {
            literal.kind = Literal::Kind::Real;
            literal.real = static_cast<double>(value);
        } else {
            literal.integer = value;
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

    bool placing() const { return !options_.placement.empty() || !options_.offload.empty(); }

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

    bool offloaded(const std::string& path) const {
        return std::ranges::any_of(
            options_.offload, [&](const std::string& prefix) { return path.starts_with(prefix); });
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
        out.name = target_.transfer(info(value), slot);
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
            Literal literal;
            literal.kind = Literal::Kind::Real;
            literal.real = a.number;
            define(op, constant_scalar(literal, result_dtype(op)));
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
        case ir::OpKind::TupleMake: {
            Val tuple;
            tuple.kind = Val::Kind::Tuple;
            for (const ir::ValueId id : op.operands) {
                tuple.elements.push_back(value(id));
            }
            define(op, tuple);
            return;
        }
        case ir::OpKind::TupleGet: {
            const Val& tuple = operand(0);
            if (tuple.kind != Val::Kind::Tuple ||
                static_cast<std::size_t>(a.integer) >= tuple.elements.size()) {
                fail("internal: tuple.get on a non-tuple");
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
            define(op, sub_value(data, block.path + a.name));
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
        case ir::OpKind::StructMake:
        case ir::OpKind::StructGet:
            fail(std::string("`") + std::string(ir::op_spelling(op.kind)) +
                 "` is not supported by graph exporters");
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
        if (op.kind == ir::OpKind::SemanticCall && try_native(op)) {
            return;
        }
        const ir::Function& callee = *found->second;
        Frame inner;
        inner.subst = op.attributes.substitution;
        for (auto& [symbol, dim] : inner.subst.dims) {
            dim = types_.substitute(dim, frame().subst);
        }
        for (auto& [symbol, shape] : inner.subst.packs) {
            shape = types_.substitute(shape, frame().subst);
        }
        for (auto& [var, dtype] : inner.subst.dtypes) {
            dtype = types_.substitute(dtype, frame().subst);
        }
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
            unit = std::ranges::find(options_.offload, path) != options_.offload.end();
            if (unit) {
                releases_.emplace_back();
                released_keys_.emplace_back();
            }
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
        Reduction kind = Reduction::Sum;
        switch (op.attributes.reduce) {
        case ir::ReduceKind::Sum:
            kind = Reduction::Sum;
            break;
        case ir::ReduceKind::Prod:
            kind = Reduction::Prod;
            break;
        case ir::ReduceKind::Max:
            kind = Reduction::Max;
            break;
        case ir::ReduceKind::Min:
            kind = Reduction::Min;
            break;
        case ir::ReduceKind::Any:
            kind = Reduction::Any;
            break;
        case ir::ReduceKind::All:
            kind = Reduction::All;
            break;
        }
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
                return broadcast(source, grid_, axes);
            }
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

    ir::Module& module_;
    const Model& model_;
    TypeStore& types_;
    const GraphExportOptions& options_;
    GraphTarget& target_;
    std::map<std::string, const ir::Function*> functions_;
    std::map<std::string, Val> parameters_;
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
        return Evaluator(module, options, target).run();
    } catch (const Unsupported& error) {
        return std::unexpected(error.what());
    }
}

namespace {

// The `vN` names a generated line mentions on its right-hand side.
std::set<std::string> mentioned_values(const std::string& line) {
    std::set<std::string> names;
    const std::size_t assignment = line.find(" = ");
    const std::size_t begin = assignment == std::string::npos ? 0 : assignment + 3;
    for (std::size_t i = begin; i < line.size(); ++i) {
        const bool boundary = i == 0 || (!std::isalnum(static_cast<unsigned char>(line[i - 1])) &&
                                         line[i - 1] != '_');
        if (line[i] != 'v' || !boundary) {
            continue;
        }
        std::size_t j = i + 1;
        while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) {
            ++j;
        }
        const bool ends = j == line.size() ||
                          (!std::isalnum(static_cast<unsigned char>(line[j])) && line[j] != '_');
        if (j > i + 1 && ends) {
            names.insert(line.substr(i, j - i));
        }
        i = j;
    }
    return names;
}

} // namespace

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
            for (std::size_t i = 0; i < after[g].size(); ++i) {
                line += (i == 0 ? "" : ", ") + after[g][i];
            }
            out += line + "\n";
        }
    }
    return out;
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

} // namespace linnet::backend
