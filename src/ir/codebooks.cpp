#include "linnet/ir/codebooks.hpp"

#include <cstdint>
#include <vector>

namespace linnet::ir {

namespace {

using sema::DType;
using sema::ScalarKind;
using sema::TypeData;
using sema::TypeId;
using sema::TypeKind;

// Appends operations to a block, to be moved where the cast they replace
// stands.
class Expansion {
public:
    Expansion(Module& module, const Operation& cast)
        : module_(module), block_(cast.block), span_(cast.span) {
        const TypeData data = module.types().get(module.value(cast.results.front()).type);
        is_tensor_ = data.kind == TypeKind::Tensor;
        shape_ = data.shape;
    }

    ValueId emit(OpKind kind, std::vector<ValueId> operands, TypeId type, Attributes a = {}) {
        const OpId id =
            module_.add_op(block_, kind, std::move(operands), {type}, std::move(a), span_);
        added_.push_back(id);
        return module_.op(id).results.front();
    }

    // A value of `kind` shaped like the cast's result.
    TypeId typed(ScalarKind kind) {
        return is_tensor_ ? module_.types().tensor(shape_, DType::of(kind))
                          : module_.types().scalar(kind);
    }

    ValueId number(double value) {
        Attributes a;
        a.number = value;
        return emit(OpKind::ConstFloat, {}, module_.types().scalar(ScalarKind::F32), a);
    }

    ValueId integer(std::int64_t value) {
        Attributes a;
        a.integer = value;
        return emit(OpKind::ConstInt, {}, module_.types().scalar(ScalarKind::I32), a);
    }

    ValueId compare(CompareKind kind, ValueId lhs, ValueId rhs) {
        Attributes a;
        a.compare = kind;
        return emit(OpKind::Compare, {lhs, rhs}, typed(ScalarKind::Bool), a);
    }

    // Moves the appended operations to just before `cast` in its block.
    void place_before(OpId cast) {
        std::vector<OpId>& ops = module_.block(block_).ops;
        ops.resize(ops.size() - added_.size());
        std::size_t at = 0;
        while (ops[at] != cast) {
            ++at;
        }
        ops.insert(ops.begin() + static_cast<std::ptrdiff_t>(at), added_.begin(), added_.end());
    }

private:
    Module& module_;
    BlockId block_;
    SourceSpan span_;
    bool is_tensor_ = false;
    sema::Shape shape_;
    std::vector<OpId> added_;
};

ValueId expand(Module& module, OpId id) {
    const Operation& cast = module.op(id);
    const sema::TypeStore& types = module.types();
    const ValueId source = cast.operands.front();
    const TypeId result = module.value(cast.results.front()).type;
    const sema::Codebook* from = types.codebook_of(types.get(module.value(source).type).dtype);
    const sema::Codebook* to = types.codebook_of(types.get(result).dtype);
    Expansion out(module, cast);
    ValueId value = source;
    if (from != nullptr) {
        const ValueId codes = out.emit(OpKind::Cast, {source}, out.typed(ScalarKind::I32));
        value = out.number(from->values.front());
        for (std::size_t k = 1; k < from->values.size(); ++k) {
            const ValueId hit =
                out.compare(CompareKind::Eq, codes, out.integer(static_cast<std::int64_t>(k)));
            value = out.emit(OpKind::Select,
                             {hit, out.number(from->values[k]), value},
                             out.typed(ScalarKind::F32));
        }
        if (to == nullptr) {
            const ValueId widened = out.emit(OpKind::Cast, {value}, result);
            out.place_before(id);
            return widened;
        }
    }
    const ValueId wide = out.emit(OpKind::Cast, {value}, out.typed(ScalarKind::F32));
    ValueId code = out.integer(0);
    for (std::size_t k = 1; k < to->values.size(); ++k) {
        const double midpoint = (to->values[k - 1] + to->values[k]) / 2;
        const ValueId above = out.compare(CompareKind::Gt, wide, out.number(midpoint));
        code = out.emit(OpKind::Add,
                        {code, out.emit(OpKind::Cast, {above}, out.typed(ScalarKind::I32))},
                        out.typed(ScalarKind::I32));
    }
    const ValueId narrowed = out.emit(OpKind::Cast, {code}, result);
    out.place_before(id);
    return narrowed;
}

} // namespace

void expand_codebook_casts(Module& module) {
    const sema::TypeStore& types = module.types();
    const auto is_codebook = [&](ValueId value) {
        return types.codebook_of(types.get(module.value(value).type).dtype) != nullptr;
    };
    std::vector<OpId> casts;
    for (const BlockId block : module.all_blocks()) {
        for (const OpId id : module.block(block).ops) {
            const Operation& op = module.op(id);
            if (op.kind == OpKind::Cast &&
                (is_codebook(op.operands.front()) || is_codebook(op.results.front()))) {
                casts.push_back(id);
            }
        }
    }
    for (const OpId id : casts) {
        const ValueId replaced = module.op(id).results.front();
        const ValueId expanded = expand(module, id);
        module.replace_uses(replaced, expanded);
        module.erase_op(id);
    }
}

} // namespace linnet::ir
