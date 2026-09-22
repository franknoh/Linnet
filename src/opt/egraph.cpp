#include "linnet/opt/egraph.hpp"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace linnet::opt {

namespace {

using namespace ir;

using ClassId = std::uint32_t;

// A term: an operation over e-classes, or a leaf standing for a value the
// graph does not look into.
struct Node {
    bool is_leaf = false;
    ValueId leaf = no_id;
    OpKind kind = OpKind::Return;
    std::vector<ClassId> children;
    Attributes attrs;
    sema::TypeId type = sema::no_type;
    bool is_original = false; // came from the IR, not from a rewrite
    OpId op = no_id;          // the IR operation it came from, when original
};

std::string attributes_key(const Module& module, const Attributes& a) {
    const sema::TypeStore& types = module.types();
    char number[32];
    std::snprintf(number, sizeof number, "%.17g", a.number);
    std::string key = std::to_string(a.integer) + "|" + number + "|" +
                      module.model().dims.to_string(a.dim) + "|" + a.name + "|";
    for (const std::string& name : a.names) {
        key += name + ",";
    }
    key += "|" + std::to_string(static_cast<int>(a.compare)) + "|" +
           std::to_string(static_cast<int>(a.reduce)) + "|" + types.to_string(a.shape) + "|";
    for (std::size_t i = 0; i < a.starts.size(); ++i) {
        key += module.model().dims.to_string(a.starts[i]) + ":" +
               module.model().dims.to_string(a.stops[i]) + ":" + std::to_string(a.steps[i]) + ":" +
               (a.squeezed[i] ? "s" : "") + (a.whole[i] ? "w" : "") + ",";
    }
    key += "|" + std::to_string(a.axis) + "|";
    for (const auto& [symbol, dim] : a.substitution.dims) {
        key += std::to_string(symbol) + "=" + module.model().dims.to_string(dim) + ",";
    }
    for (const auto& [symbol, shape] : a.substitution.packs) {
        key += std::to_string(symbol) + "=[" + types.to_string(shape) + "],";
    }
    for (const auto& [var, dtype] : a.substitution.dtypes) {
        key += std::to_string(var) + "=" + types.to_string(dtype) + ",";
    }
    return key;
}

class EGraph {
public:
    explicit EGraph(Module& module) : module_(module) {}

    ClassId find(ClassId id) const {
        while (parent_[id] != id) {
            id = parent_[id];
        }
        return id;
    }

    // Adds a term, returning the class that holds it.
    ClassId add(Node node) {
        for (ClassId& child : node.children) {
            child = find(child);
        }
        const std::string key = node_key(node);
        if (const auto found = hashcons_.find(key); found != hashcons_.end()) {
            return find(found->second);
        }
        const auto id = static_cast<ClassId>(parent_.size());
        parent_.push_back(id);
        type_keys_.push_back(module_.types().to_string(node.type));
        nodes_.emplace_back();
        nodes_.back().push_back(std::move(node));
        hashcons_[key] = id;
        return id;
    }

    // Declares two classes equal; their nodes are brought together by the
    // next `rebuild`. False when the types differ (never merged) or when the
    // classes were already one.
    bool merge(ClassId a, ClassId b) {
        a = find(a);
        b = find(b);
        if (a == b || type_keys_[a] != type_keys_[b]) {
            return false;
        }
        parent_[std::max(a, b)] = std::min(a, b);
        is_dirty_ = true;
        return true;
    }

    // Restores the invariants: every node lives in its root class with
    // canonical children, and equal terms share one class.
    void rebuild() {
        while (is_dirty_) {
            is_dirty_ = false;
            std::vector<std::pair<ClassId, Node>> all;
            for (ClassId id = 0; id < parent_.size(); ++id) {
                for (Node& node : nodes_[id]) {
                    for (ClassId& child : node.children) {
                        child = find(child);
                    }
                    all.emplace_back(find(id), std::move(node));
                }
                nodes_[id].clear();
            }
            hashcons_.clear();
            std::map<std::string, std::size_t> first;
            std::vector<bool> is_duplicate(all.size(), false);
            for (std::size_t i = 0; i < all.size(); ++i) {
                const std::string key = node_key(all[i].second);
                const auto found = first.find(key);
                if (found == first.end()) {
                    first[key] = i;
                    continue;
                }
                // The same term twice: its classes are equal, and one copy
                // is enough.
                Node& kept = all[found->second].second;
                kept.is_original = kept.is_original || all[i].second.is_original;
                kept.op = kept.op == no_id ? all[i].second.op : kept.op;
                is_duplicate[i] = true;
                if (find(all[found->second].first) != find(all[i].first)) {
                    parent_[std::max(find(all[found->second].first), find(all[i].first))] =
                        std::min(find(all[found->second].first), find(all[i].first));
                    is_dirty_ = true;
                }
            }
            for (std::size_t i = 0; i < all.size(); ++i) {
                if (is_duplicate[i]) {
                    continue;
                }
                const ClassId root = find(all[i].first);
                hashcons_[node_key(all[i].second)] = root;
                nodes_[root].push_back(std::move(all[i].second));
            }
        }
    }

    std::size_t class_count() const { return parent_.size(); }
    std::size_t node_count() const {
        std::size_t total = 0;
        for (const auto& nodes : nodes_) {
            total += nodes.size();
        }
        return total;
    }
    const std::vector<Node>& nodes(ClassId id) const { return nodes_[find(id)]; }
    const std::string& type_key(ClassId id) const { return type_keys_[find(id)]; }
    bool is_root(ClassId id) const { return find(id) == id; }

private:
    std::string node_key(const Node& node) const {
        if (node.is_leaf) {
            return "leaf:" + std::to_string(node.leaf);
        }
        std::string key = std::string(op_spelling(node.kind)) + "(";
        for (const ClassId child : node.children) {
            key += std::to_string(find(child)) + ",";
        }
        return key + ")" + attributes_key(module_, node.attrs) + "#" +
               module_.types().to_string(node.type);
    }

    Module& module_;
    std::vector<ClassId> parent_;
    std::deque<std::vector<Node>> nodes_; // deque: references survive growth
    std::vector<std::string> type_keys_;
    std::map<std::string, ClassId> hashcons_;
    bool is_dirty_ = false;
};

// ------------------------------------------------------------------ rules

const std::vector<Rewrite>& rule_table() {
    static const std::vector<Rewrite> table = {
        {"compose reshapes", Legality::Exact},
        {"compose permutes", Legality::Exact},
        {"compose broadcasts", Legality::Exact},
        {"double negation", Legality::Exact},
        {"commutativity", Legality::Exact},
        {"multiplicative identity", Legality::Exact},
        {"select of equal branches", Legality::Exact},
        {"additive identity", Legality::IEEEEquivalent},
        {"reassociation", Legality::NumericallyEquivalent},
        {"factoring", Legality::NumericallyEquivalent},
    };
    return table;
}

Legality rule_legality(const char* name) {
    for (const Rewrite& rewrite : rule_table()) {
        if (rewrite.name == name) {
            return rewrite.legality;
        }
    }
    return Legality::Approximate;
}

bool is_commutative(OpKind kind) {
    return kind == OpKind::Add || kind == OpKind::Mul || kind == OpKind::Min ||
           kind == OpKind::Max || kind == OpKind::And || kind == OpKind::Or;
}

bool is_constant_number(const Node& node, double value) {
    if (node.is_leaf) {
        return false;
    }
    if (node.kind == OpKind::ConstInt) {
        return static_cast<double>(node.attrs.integer) == value;
    }
    if (node.kind == OpKind::ConstFloat) {
        return node.attrs.number == value;
    }
    return false;
}

// One saturation over the terms of a block.
class Saturator {
public:
    Saturator(Module& module, const SaturationOptions& options)
        : module_(module), options_(options), graph_(module) {}

    bool run(BlockId block_id) {
        const Block& block = module_.block(block_id);
        // Every value used or defined in the block gets a class; operations
        // without regions become terms, everything else leaves.
        for (const OpId id : block.ops) {
            const Operation& op = module_.op(id);
            if (op.regions.empty() && op.results.size() == 1 && op.kind != OpKind::Return &&
                op.kind != OpKind::Yield) {
                Node node;
                node.kind = op.kind;
                node.attrs = op.attributes;
                node.type = module_.value(op.results.front()).type;
                node.is_original = true;
                node.op = id;
                for (const ValueId operand : op.operands) {
                    node.children.push_back(class_of(operand));
                }
                const ClassId klass = graph_.add(std::move(node));
                classes_[op.results.front()] = klass;
                originals_.emplace_back(id, klass);
            }
        }
        if (originals_.empty()) {
            return false;
        }
        for (int round = 0; round < options_.max_rounds; ++round) {
            if (!saturate_round() || graph_.node_count() > options_.max_nodes) {
                break;
            }
        }
        return extract(block_id);
    }

private:
    ClassId class_of(ValueId value) {
        if (const auto found = classes_.find(value); found != classes_.end()) {
            return found->second;
        }
        Node leaf;
        leaf.is_leaf = true;
        leaf.leaf = value;
        leaf.type = module_.value(value).type;
        const ClassId klass = graph_.add(std::move(leaf));
        classes_[value] = klass;
        return klass;
    }

    bool allows(const char* rule) const { return rule_legality(rule) <= options_.allowed; }

    // Finds every match, then applies the additions and merges together.
    bool saturate_round() {
        std::vector<std::pair<ClassId, Node>> additions;
        std::vector<std::pair<ClassId, ClassId>> merges;
        const auto count = static_cast<ClassId>(graph_.class_count());
        for (ClassId id = 0; id < count; ++id) {
            if (!graph_.is_root(id)) {
                continue;
            }
            for (const Node& node : graph_.nodes(id)) {
                if (!node.is_leaf) {
                    match(id, node, additions, merges);
                }
            }
        }
        bool has_changed = false;
        for (auto& [klass, node] : additions) {
            const ClassId added = graph_.add(std::move(node));
            has_changed = graph_.merge(klass, added) || has_changed;
        }
        for (const auto& [a, b] : merges) {
            has_changed = graph_.merge(a, b) || has_changed;
        }
        graph_.rebuild();
        return has_changed;
    }

    Node derived(OpKind kind, std::vector<ClassId> children, Attributes attrs, sema::TypeId type) {
        Node node;
        node.kind = kind;
        node.children = std::move(children);
        node.attrs = std::move(attrs);
        node.type = type;
        return node;
    }

    bool same_type(ClassId a, ClassId b) const { return graph_.type_key(a) == graph_.type_key(b); }

    void match(ClassId id,
               const Node& node,
               std::vector<std::pair<ClassId, Node>>& additions,
               std::vector<std::pair<ClassId, ClassId>>& merges) {
        const auto child_nodes = [&](std::size_t i) -> const std::vector<Node>& {
            return graph_.nodes(node.children[i]);
        };
        switch (node.kind) {
        case OpKind::Reshape:
            if (allows("compose reshapes")) {
                for (const Node& inner : child_nodes(0)) {
                    if (!inner.is_leaf && inner.kind == OpKind::Reshape) {
                        additions.emplace_back(
                            id,
                            derived(OpKind::Reshape, {inner.children[0]}, node.attrs, node.type));
                    }
                }
            }
            break;
        case OpKind::Broadcast:
            if (allows("compose broadcasts")) {
                for (const Node& inner : child_nodes(0)) {
                    if (!inner.is_leaf && inner.kind == OpKind::Broadcast) {
                        additions.emplace_back(
                            id,
                            derived(OpKind::Broadcast, {inner.children[0]}, node.attrs, node.type));
                    }
                }
            }
            break;
        case OpKind::Permute:
            if (allows("compose permutes")) {
                for (const Node& inner : child_nodes(0)) {
                    if (inner.is_leaf || inner.kind != OpKind::Permute ||
                        inner.attrs.shape.size() != node.attrs.shape.size()) {
                        continue;
                    }
                    // permute(permute(x, p), q) = permute(x, p ∘ q)
                    Attributes composed = node.attrs;
                    bool is_constant = true;
                    for (std::size_t i = 0; i < node.attrs.shape.size(); ++i) {
                        const auto q = node.attrs.shape[i].dim.constant();
                        if (!q || *q < 0 ||
                            static_cast<std::size_t>(*q) >= inner.attrs.shape.size()) {
                            is_constant = false;
                            break;
                        }
                        composed.shape[i] = inner.attrs.shape[static_cast<std::size_t>(*q)];
                    }
                    if (is_constant) {
                        additions.emplace_back(
                            id, derived(OpKind::Permute, {inner.children[0]}, composed, node.type));
                    }
                }
            }
            break;
        case OpKind::Neg:
        case OpKind::Not:
            if (allows("double negation")) {
                for (const Node& inner : child_nodes(0)) {
                    if (!inner.is_leaf && inner.kind == node.kind &&
                        same_type(id, inner.children[0])) {
                        merges.emplace_back(id, inner.children[0]);
                    }
                }
            }
            break;
        case OpKind::Select:
            if (allows("select of equal branches") && node.children[1] == node.children[2] &&
                same_type(id, node.children[1])) {
                merges.emplace_back(id, node.children[1]);
            }
            break;
        case OpKind::Compare:
            if (allows("commutativity") &&
                (node.attrs.compare == CompareKind::Eq || node.attrs.compare == CompareKind::Ne)) {
                additions.emplace_back(
                    id,
                    derived(
                        node.kind, {node.children[1], node.children[0]}, node.attrs, node.type));
            }
            break;
        case OpKind::Sub:
        case OpKind::Div:
            if (allows(node.kind == OpKind::Sub ? "additive identity"
                                                : "multiplicative identity") &&
                same_type(id, node.children[0])) {
                for (const Node& right : child_nodes(1)) {
                    if (is_constant_number(right, node.kind == OpKind::Sub ? 0.0 : 1.0)) {
                        merges.emplace_back(id, node.children[0]);
                    }
                }
            }
            break;
        default:
            break;
        }
        if (!is_commutative(node.kind)) {
            return;
        }
        if (allows("commutativity")) {
            additions.emplace_back(
                id,
                derived(node.kind, {node.children[1], node.children[0]}, node.attrs, node.type));
        }
        const bool is_add = node.kind == OpKind::Add;
        const bool is_mul = node.kind == OpKind::Mul;
        if ((is_add && allows("additive identity")) ||
            (is_mul && allows("multiplicative identity"))) {
            for (std::size_t side = 0; side < 2; ++side) {
                if (!same_type(id, node.children[1 - side])) {
                    continue;
                }
                for (const Node& other : child_nodes(side)) {
                    if (is_constant_number(other, is_add ? 0.0 : 1.0)) {
                        merges.emplace_back(id, node.children[1 - side]);
                    }
                }
            }
        }
        if ((is_add || is_mul) && allows("reassociation") && same_type(id, node.children[0]) &&
            same_type(id, node.children[1])) {
            // (a ∘ b) ∘ c = a ∘ (b ∘ c) when every term has the result's type.
            for (const Node& inner : child_nodes(0)) {
                if (inner.is_leaf || inner.kind != node.kind || !same_type(id, inner.children[0]) ||
                    !same_type(id, inner.children[1])) {
                    continue;
                }
                const ClassId right = graph_.add(derived(
                    node.kind, {inner.children[1], node.children[1]}, node.attrs, node.type));
                additions.emplace_back(
                    id, derived(node.kind, {inner.children[0], right}, node.attrs, node.type));
            }
        }
        if (is_add && allows("factoring") && same_type(id, node.children[0]) &&
            same_type(id, node.children[1])) {
            // a*c + b*c = (a + b)*c
            for (const Node& left : child_nodes(0)) {
                if (left.is_leaf || left.kind != OpKind::Mul) {
                    continue;
                }
                for (const Node& right : child_nodes(1)) {
                    if (right.is_leaf || right.kind != OpKind::Mul) {
                        continue;
                    }
                    for (std::size_t i = 0; i < 2; ++i) {
                        for (std::size_t j = 0; j < 2; ++j) {
                            if (left.children[i] != right.children[j] ||
                                !same_type(id, left.children[1 - i]) ||
                                !same_type(id, right.children[1 - j]) ||
                                !same_type(id, left.children[i])) {
                                continue;
                            }
                            const ClassId sum =
                                graph_.add(derived(OpKind::Add,
                                                   {left.children[1 - i], right.children[1 - j]},
                                                   {},
                                                   node.type));
                            additions.emplace_back(
                                id, derived(OpKind::Mul, {sum, left.children[i]}, {}, node.type));
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------- extraction

    std::int64_t element_count(sema::TypeId type) const {
        const sema::TypeData& data = module_.types().get(type);
        std::int64_t count = 1;
        for (const sema::ShapeElem& unit : data.shape) {
            const auto constant = unit.is_pack ? std::nullopt : unit.dim.constant();
            count *= constant ? std::max<std::int64_t>(*constant, 1) : options_.symbolic_extent;
        }
        return count;
    }

    static std::int64_t weight(OpKind kind) {
        switch (kind) {
        case OpKind::ConstInt:
        case OpKind::ConstFloat:
        case OpKind::ConstBool:
        case OpKind::ConstDim:
        case OpKind::EnumConst:
        case OpKind::TupleMake:
        case OpKind::TupleGet:
        case OpKind::OptionSome:
        case OpKind::OptionNone:
        case OpKind::BlockParam:
        case OpKind::BlockSub:
        case OpKind::ArrayGet:
            return 0;
        case OpKind::Reshape:
            return 1;
        case OpKind::Broadcast:
        case OpKind::Cast:
        case OpKind::Compare:
        case OpKind::Select:
        case OpKind::Add:
        case OpKind::Sub:
        case OpKind::Mul:
        case OpKind::Min:
        case OpKind::Max:
        case OpKind::And:
        case OpKind::Or:
        case OpKind::Not:
        case OpKind::Neg:
        case OpKind::Abs:
            return 2;
        case OpKind::Div:
        case OpKind::Rem:
        case OpKind::Permute:
        case OpKind::Slice:
        case OpKind::Concat:
        case OpKind::Fill:
        case OpKind::Iota:
            return 4;
        default:
            return 8; // transcendental functions and calls
        }
    }

    std::int64_t node_cost(const Node& node) const {
        return node.is_leaf ? 0 : weight(node.kind) * element_count(node.type);
    }

    // Cheapest term per class, by fixpoint; originals win ties.
    struct Choice {
        std::optional<std::int64_t> cost;
        const Node* node = nullptr;
    };

    std::vector<Choice> choose() {
        std::vector<Choice> best(graph_.class_count());
        bool has_progress = true;
        while (has_progress) {
            has_progress = false;
            for (ClassId id = 0; id < graph_.class_count(); ++id) {
                if (!graph_.is_root(id)) {
                    continue;
                }
                for (const Node& node : graph_.nodes(id)) {
                    std::int64_t total = node_cost(node);
                    bool is_ready = true;
                    for (const ClassId child : node.children) {
                        const Choice& choice = best[graph_.find(child)];
                        if (!choice.cost) {
                            is_ready = false;
                            break;
                        }
                        total += *choice.cost;
                    }
                    if (!is_ready) {
                        continue;
                    }
                    Choice& current = best[id];
                    const bool is_better =
                        !current.cost || total < *current.cost ||
                        (total == *current.cost && node.is_original && !current.node->is_original);
                    if (is_better) {
                        current = {total, &node};
                        has_progress = true;
                    }
                }
            }
        }
        return best;
    }

    bool extract(BlockId block_id) {
        const std::vector<Choice> best = choose();
        std::map<ClassId, ValueId> materialized;
        bool has_changed = false;
        for (const auto& [op_id, klass] : originals_) {
            const ValueId original = module_.op(op_id).results.front();
            const ValueId replacement =
                materialize(block_id, graph_.find(klass), best, materialized);
            if (replacement != original) {
                module_.replace_uses(original, replacement);
                has_changed = true;
            }
        }
        if (has_changed) {
            reorder(block_id);
        }
        return has_changed;
    }

    ValueId materialize(BlockId block_id,
                        ClassId klass,
                        const std::vector<Choice>& best,
                        std::map<ClassId, ValueId>& materialized) {
        if (const auto found = materialized.find(klass); found != materialized.end()) {
            return found->second;
        }
        const Choice& choice = best[klass];
        if (choice.node == nullptr) {
            throw std::logic_error("saturation: a class has no extractable term");
        }
        const Node& node = *choice.node;
        ValueId value = no_id;
        if (node.is_leaf) {
            value = node.leaf;
        } else if (node.op != no_id) {
            value = module_.op(node.op).results.front();
        } else {
            std::vector<ValueId> operands;
            operands.reserve(node.children.size());
            for (const ClassId child : node.children) {
                operands.push_back(materialize(block_id, graph_.find(child), best, materialized));
            }
            const OpId op =
                module_.add_op(block_id, node.kind, std::move(operands), {node.type}, node.attrs);
            value = module_.op(op).results.front();
        }
        materialized[klass] = value;
        return value;
    }

    // Puts the block's operations back in dependency order, keeping the
    // original relative order where dependencies allow, terminator last.
    void reorder(BlockId block_id) {
        Block& block = module_.block(block_id);
        std::map<ValueId, OpId> producers;
        for (const OpId id : block.ops) {
            for (const ValueId result : module_.op(id).results) {
                producers[result] = id;
            }
        }
        std::map<OpId, std::set<OpId>> dependencies;
        for (const OpId id : block.ops) {
            std::set<OpId>& deps = dependencies[id];
            collect_dependencies(module_.op(id), producers, deps);
            deps.erase(id);
        }
        std::vector<OpId> order;
        std::set<OpId> placed;
        std::vector<OpId> pending = block.ops;
        OpId terminator = no_id;
        while (!pending.empty()) {
            bool has_progress = false;
            std::vector<OpId> remaining;
            for (const OpId id : pending) {
                const Operation& op = module_.op(id);
                if (op.kind == OpKind::Return || op.kind == OpKind::Yield) {
                    terminator = id;
                    continue;
                }
                const std::set<OpId>& deps = dependencies[id];
                const bool is_ready = std::all_of(
                    deps.begin(), deps.end(), [&](OpId dep) { return placed.contains(dep); });
                if (is_ready) {
                    order.push_back(id);
                    placed.insert(id);
                    has_progress = true;
                } else {
                    remaining.push_back(id);
                }
            }
            pending = std::move(remaining);
            if (!has_progress) {
                throw std::logic_error("saturation: cyclic dependencies after extraction");
            }
        }
        if (terminator != no_id) {
            order.push_back(terminator);
        }
        block.ops = std::move(order);
    }

    void collect_dependencies(const Operation& op,
                              const std::map<ValueId, OpId>& producers,
                              std::set<OpId>& deps) const {
        for (const ValueId operand : op.operands) {
            if (const auto found = producers.find(operand); found != producers.end()) {
                deps.insert(found->second);
            }
        }
        for (const RegionId region : op.regions) {
            for (const BlockId inner : module_.region(region).blocks) {
                for (const OpId id : module_.block(inner).ops) {
                    collect_dependencies(module_.op(id), producers, deps);
                }
            }
        }
    }

    Module& module_;
    const SaturationOptions& options_;
    EGraph graph_;
    std::map<ValueId, ClassId> classes_;
    std::vector<std::pair<OpId, ClassId>> originals_;
};

} // namespace

std::vector<Rewrite> rewrites() {
    return rule_table();
}

bool saturate(Module& module, const SaturationOptions& options) {
    bool has_changed = false;
    for (const BlockId block : module.all_blocks()) {
        Saturator saturator(module, options);
        has_changed = saturator.run(block) || has_changed;
    }
    return has_changed;
}

Pass saturation_pass(Legality allowed) {
    return {"saturate", Legality::Exact, [allowed](Module& module) {
                SaturationOptions options;
                options.allowed = allowed;
                return saturate(module, options);
            }};
}

std::vector<Pass> optimizing_passes(Legality allowed) {
    std::vector<Pass> passes = canonical_passes();
    passes.push_back(saturation_pass(allowed));
    passes.push_back({"dce", Legality::Exact, eliminate_dead_code});
    return passes;
}

} // namespace linnet::opt
