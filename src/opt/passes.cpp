#include "linnet/opt/passes.hpp"

#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

namespace linnet::opt {

namespace {

using namespace ir;

// Uses of every value across the module, including uses inside nested regions.
std::map<ValueId, std::size_t> count_uses(const Module& module) {
    std::map<ValueId, std::size_t> uses;
    for (const BlockId block : module.all_blocks()) {
        for (const OpId id : module.block(block).ops) {
            for (const ValueId operand : module.op(id).operands) {
                ++uses[operand];
            }
        }
    }
    return uses;
}

bool is_terminator(OpKind kind) {
    return kind == OpKind::Return || kind == OpKind::Yield;
}

// State reads and writes are ordered effects: never removed as dead, never
// merged with one another, never moved past each other.
bool has_effects(OpKind kind) {
    return kind == OpKind::StateRead || kind == OpKind::StateWrite;
}

// Integer arithmetic that refuses to overflow, so that folding never changes
// a result the program would have computed differently.
std::optional<std::int64_t> fold_integers(OpKind kind, std::int64_t x, std::int64_t y) {
    constexpr std::int64_t low = std::numeric_limits<std::int64_t>::min();
    constexpr std::int64_t high = std::numeric_limits<std::int64_t>::max();
    switch (kind) {
    case OpKind::Add:
        if ((y > 0 && x > high - y) || (y < 0 && x < low - y)) {
            return std::nullopt;
        }
        return x + y;
    case OpKind::Sub:
        if ((y < 0 && x > high + y) || (y > 0 && x < low + y)) {
            return std::nullopt;
        }
        return x - y;
    case OpKind::Mul: {
        if (x == 0 || y == 0) {
            return 0;
        }
        if ((x == -1 && y == low) || (y == -1 && x == low)) {
            return std::nullopt;
        }
        const std::int64_t product = static_cast<std::int64_t>(static_cast<std::uint64_t>(x) *
                                                               static_cast<std::uint64_t>(y));
        return product / y == x ? std::optional(product) : std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

// Structural identity of two types: identical shape expressions, not merely
// provably equal ones. Types are interned per use, so ids differ for equal
// types and cannot be compared directly.
bool same_type(const Module& module, sema::TypeId a, sema::TypeId b) {
    if (a == b) {
        return true;
    }
    const sema::TypeData& x = module.types().get(a);
    const sema::TypeData& y = module.types().get(b);
    if (x.kind != y.kind || x.dtype != y.dtype || x.shape != y.shape || x.value != y.value ||
        x.decl != y.decl || x.args != y.args || x.elements.size() != y.elements.size()) {
        return false;
    }
    for (std::size_t i = 0; i < x.elements.size(); ++i) {
        if (!same_type(module, x.elements[i], y.elements[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

// ------------------------------------------------------------------------ DCE

bool eliminate_dead_code(Module& module) {
    bool has_changed = false;
    bool has_progress = true;
    while (has_progress) {
        has_progress = false;
        const std::map<ValueId, std::size_t> uses = count_uses(module);
        for (const BlockId block : module.all_blocks()) {
            // Iterate over a copy: erasing changes the block's op list.
            const std::vector<OpId> ops = module.block(block).ops;
            for (const OpId id : ops) {
                const Operation& op = module.op(id);
                if (is_terminator(op.kind) || op.kind == OpKind::StateWrite ||
                    op.kind == OpKind::While) {
                    continue;
                }
                bool is_used = false;
                for (const ValueId result : op.results) {
                    is_used = is_used || uses.contains(result);
                }
                if (!is_used) {
                    module.erase_op(id);
                    has_progress = true;
                    has_changed = true;
                }
            }
        }
    }
    return has_changed;
}

// ------------------------------------------------------------------------ CSE

bool eliminate_common_subexpressions(Module& module) {
    bool has_changed = false;
    for (const BlockId block : module.all_blocks()) {
        // Only region-free operations are merged: two comprehensions with
        // equal attributes may still differ in their bodies.
        std::vector<OpId> seen;
        const std::vector<OpId> ops = module.block(block).ops;
        for (const OpId id : ops) {
            const Operation& op = module.op(id);
            if (is_terminator(op.kind) || has_effects(op.kind) || !op.regions.empty() ||
                op.results.size() != 1) {
                continue;
            }
            bool is_duplicate = false;
            for (const OpId earlier_id : seen) {
                const Operation& earlier = module.op(earlier_id);
                const bool is_same_type = same_type(module,
                                                    module.value(earlier.results.front()).type,
                                                    module.value(op.results.front()).type);
                if (earlier.kind == op.kind && earlier.operands == op.operands &&
                    earlier.attributes == op.attributes && is_same_type) {
                    module.replace_uses(op.results.front(), earlier.results.front());
                    module.erase_op(id);
                    is_duplicate = true;
                    has_changed = true;
                    break;
                }
            }
            if (!is_duplicate) {
                seen.push_back(id);
            }
        }
    }
    return has_changed;
}

// ------------------------------------------------------------- canonicalize

namespace {

bool is_identity_permutation(const sema::Shape& axes) {
    for (std::size_t i = 0; i < axes.size(); ++i) {
        const auto constant = axes[i].dim.constant();
        if (!constant || *constant != static_cast<std::int64_t>(i)) {
            return false;
        }
    }
    return true;
}

} // namespace

bool canonicalize(Module& module) {
    bool has_changed = false;
    for (const BlockId block : module.all_blocks()) {
        const std::vector<OpId> ops = module.block(block).ops;
        for (const OpId id : ops) {
            Operation& op = module.op(id);
            if (op.results.size() != 1) {
                continue;
            }
            const ValueId result = op.results.front();
            const auto replace_with = [&](ValueId value) {
                module.replace_uses(result, value);
                module.erase_op(id);
                has_changed = true;
            };
            const auto operand_type = [&](std::size_t i) {
                return module.value(op.operands[i]).type;
            };
            switch (op.kind) {
            case OpKind::Reshape:
            case OpKind::Broadcast:
            case OpKind::Cast:
                // A view or conversion to the operand's own type is a copy.
                if (same_type(module, operand_type(0), module.value(result).type)) {
                    replace_with(op.operands.front());
                }
                break;
            case OpKind::Permute:
                if (is_identity_permutation(op.attributes.shape)) {
                    replace_with(op.operands.front());
                    break;
                }
                // permute(permute(x, p), q) == permute(x, p[q]).
                if (const OpId producer = module.value(op.operands.front()).producer;
                    producer != no_id && module.op(producer).kind == OpKind::Permute) {
                    const Operation& inner = module.op(producer);
                    sema::Shape composed;
                    for (const sema::ShapeElem& axis : op.attributes.shape) {
                        const auto index = axis.dim.constant();
                        if (!index || *index < 0 ||
                            *index >= static_cast<std::int64_t>(inner.attributes.shape.size())) {
                            composed.clear();
                            break;
                        }
                        composed.push_back(
                            inner.attributes.shape[static_cast<std::size_t>(*index)]);
                    }
                    if (composed.size() == op.attributes.shape.size()) {
                        op.operands.front() = inner.operands.front();
                        op.attributes.shape = std::move(composed);
                        has_changed = true;
                    }
                }
                break;
            case OpKind::Neg: {
                // neg(neg(x)) == x, and the negation of a literal is a literal
                // (exact: negation only flips the sign bit).
                const ValueId inner = op.operands.front();
                const OpId producer = module.value(inner).producer;
                if (producer == no_id) {
                    break;
                }
                const Operation& source = module.op(producer);
                if (source.kind == OpKind::Neg) {
                    replace_with(source.operands.front());
                } else if (source.kind == OpKind::ConstFloat ||
                           (source.kind == OpKind::ConstInt &&
                            source.attributes.integer !=
                                std::numeric_limits<std::int64_t>::min())) {
                    const Attributes negated = source.attributes;
                    op.kind = source.kind;
                    op.operands.clear();
                    op.attributes = {};
                    op.attributes.integer = -negated.integer;
                    op.attributes.number = -negated.number;
                    has_changed = true;
                }
                break;
            }
            case OpKind::Not:
                if (const ValueId inner = op.operands.front();
                    module.value(inner).producer != no_id &&
                    module.op(module.value(inner).producer).kind == OpKind::Not) {
                    replace_with(module.op(module.value(inner).producer).operands.front());
                }
                break;
            case OpKind::Add:
            case OpKind::Sub:
            case OpKind::Mul: {
                // Integer constant folding; floating-point values are left
                // alone so that no rounding decision is made here.
                const OpId a = module.value(op.operands[0]).producer;
                const OpId b = module.value(op.operands[1]).producer;
                if (a == no_id || b == no_id || module.op(a).kind != OpKind::ConstInt ||
                    module.op(b).kind != OpKind::ConstInt) {
                    break;
                }
                const auto folded = fold_integers(
                    op.kind, module.op(a).attributes.integer, module.op(b).attributes.integer);
                if (!folded) {
                    break;
                }
                op.kind = OpKind::ConstInt;
                op.operands.clear();
                op.attributes = {};
                op.attributes.integer = *folded;
                has_changed = true;
                break;
            }
            default:
                break;
            }
        }
    }
    return has_changed;
}

// ------------------------------------------------------------------ pipeline

std::vector<Pass> canonical_passes() {
    return {
        {"canonicalize", Legality::Exact, canonicalize},
        {"cse", Legality::Exact, eliminate_common_subexpressions},
        {"dce", Legality::Exact, eliminate_dead_code},
    };
}

std::vector<PassResult>
run_pipeline(Module& module, const std::vector<Pass>& passes, const PipelineOptions& options) {
    std::vector<PassResult> results;
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        bool has_changed = false;
        for (const Pass& pass : passes) {
            if (pass.legality > options.allowed) {
                continue;
            }
            const bool changed = pass.run(module);
            results.push_back({pass.name, changed});
            has_changed = has_changed || changed;
            if (options.verify && changed) {
                const std::vector<std::string> problems = verify(module);
                if (!problems.empty()) {
                    throw std::logic_error("pass `" + pass.name +
                                           "` produced invalid IR: " + problems.front());
                }
            }
        }
        if (!has_changed) {
            break;
        }
    }
    return results;
}

} // namespace linnet::opt
