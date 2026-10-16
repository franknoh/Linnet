#include "linnet/ir/formats.hpp"

#include <cstdint>
#include <limits>
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

    ValueId integer(std::int64_t value, ScalarKind kind = ScalarKind::I32) {
        Attributes a;
        a.integer = value;
        return emit(OpKind::ConstInt, {}, module_.types().scalar(kind), a);
    }

    ValueId compare(CompareKind kind, ValueId lhs, ValueId rhs) {
        Attributes a;
        a.compare = kind;
        return emit(OpKind::Compare, {lhs, rhs}, typed(ScalarKind::Bool), a);
    }

    // `kind(lhs, rhs)`, of `dtype` shaped like the cast's result.
    ValueId binary(OpKind kind, ValueId lhs, ValueId rhs, ScalarKind dtype) {
        return emit(kind, {lhs, rhs}, typed(dtype));
    }

    ValueId cast(ValueId value, ScalarKind dtype) {
        return emit(OpKind::Cast, {value}, typed(dtype));
    }

    ValueId select(ValueId condition, ValueId yes, ValueId no, ScalarKind dtype) {
        return emit(OpKind::Select, {condition, yes, no}, typed(dtype));
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

// A format whose casts this pass rewrites: a codebook's, or E8M0's.
struct Format {
    const sema::Codebook* codebook = nullptr;
    bool is_e8m0 = false;
    bool any() const { return codebook != nullptr || is_e8m0; }
};

Format format_of(const sema::TypeStore& types, TypeId type) {
    const DType dtype = types.get(type).dtype;
    const sema::Codebook* codebook = types.codebook_of(dtype);
    return {codebook, codebook == nullptr && !dtype.is_var && dtype.scalar == ScalarKind::F8E8M0};
}

// The value of each code of `codebook`: a `select` per code.
ValueId codebook_values(Expansion& out, ValueId source, const sema::Codebook& codebook) {
    const ValueId codes = out.cast(source, ScalarKind::I32);
    ValueId value = out.number(codebook.values.front());
    for (std::size_t k = 1; k < codebook.values.size(); ++k) {
        const ValueId hit =
            out.compare(CompareKind::Eq, codes, out.integer(static_cast<std::int64_t>(k)));
        value = out.select(hit, out.number(codebook.values[k]), value, ScalarKind::F32);
    }
    return value;
}

// The code of the value of `codebook` nearest `wide`: the count of the
// midpoints below it.
ValueId codebook_codes(Expansion& out, ValueId wide, const sema::Codebook& codebook) {
    ValueId code = out.integer(0);
    for (std::size_t k = 1; k < codebook.values.size(); ++k) {
        const double midpoint = (codebook.values[k - 1] + codebook.values[k]) / 2;
        const ValueId above = out.compare(CompareKind::Gt, wide, out.number(midpoint));
        code = out.binary(OpKind::Add, code, out.cast(above, ScalarKind::I32), ScalarKind::I32);
    }
    return code;
}

// `2 ** k` in f32 for an i32 `k` of -149 to 127, exact: the product of three
// powers of two of at most 62 each (from i64 shifts), or of their
// reciprocals for a negative `k`, so no partial product leaves f32's range.
ValueId power_of_two(Expansion& out, ValueId k) {
    const ValueId zero = out.integer(0);
    const ValueId nonnegative = out.compare(CompareKind::Ge, k, zero);
    const ValueId magnitude = out.emit(OpKind::Abs, {k}, out.typed(ScalarKind::I32));
    const auto factor = [&](std::int64_t skip, bool is_last) {
        ValueId count = magnitude;
        if (skip != 0) {
            count = out.binary(OpKind::Sub, magnitude, out.integer(skip), ScalarKind::I32);
            count = out.binary(OpKind::Max, count, zero, ScalarKind::I32);
        }
        if (!is_last) {
            count = out.binary(OpKind::Min, count, out.integer(62), ScalarKind::I32);
        }
        const ValueId shifted = out.binary(OpKind::Shl,
                                           out.integer(1, ScalarKind::I64),
                                           out.cast(count, ScalarKind::I64),
                                           ScalarKind::I64);
        const ValueId up = out.cast(shifted, ScalarKind::F32);
        const ValueId down = out.binary(OpKind::Div, out.number(1.0), up, ScalarKind::F32);
        return out.select(nonnegative, up, down, ScalarKind::F32);
    };
    const ValueId low = factor(0, false);
    const ValueId middle = factor(62, false);
    const ValueId high = factor(124, true);
    return out.binary(
        OpKind::Mul, out.binary(OpKind::Mul, low, middle, ScalarKind::F32), high, ScalarKind::F32);
}

// An E8M0 byte's value, `2 ** (byte - 127)`, and NaN for 255.
ValueId e8m0_values(Expansion& out, ValueId source) {
    const ValueId bits = out.cast(source, ScalarKind::I32);
    const ValueId power =
        power_of_two(out, out.binary(OpKind::Sub, bits, out.integer(127), ScalarKind::I32));
    const ValueId is_nan = out.compare(CompareKind::Eq, bits, out.integer(255));
    return out.select(
        is_nan, out.number(std::numeric_limits<double>::quiet_NaN()), power, ScalarKind::F32);
}

// The E8M0 byte of `wide` (f32), as PyTorch rounds to `float8_e8m0fnu`: its
// magnitude's power of two, the next one up from 1.5 times it (and at it);
// below `2 ** -127`, zero included, to `2 ** -127`; NaN, infinities and from
// `1.5 * 2 ** 127` up, to NaN. The power below the magnitude is the base-2
// logarithm cut toward zero, within one of it, and moved by one where the
// exact power says it is off.
ValueId e8m0_codes(Expansion& out, ValueId wide) {
    const ValueId magnitude = out.emit(OpKind::Abs, {wide}, out.typed(ScalarKind::F32));
    const ValueId finite =
        out.compare(CompareKind::Le, magnitude, out.number(std::numeric_limits<float>::max()));
    const ValueId logarithm =
        out.binary(OpKind::Mul,
                   out.emit(OpKind::Log, {magnitude}, out.typed(ScalarKind::F32)),
                   out.number(1.4426950408889634),
                   ScalarKind::F32);
    const auto clamp = [&](ValueId value, ScalarKind dtype, ValueId low, ValueId high) {
        return out.binary(OpKind::Min, out.binary(OpKind::Max, value, low, dtype), high, dtype);
    };
    // Bounded where the code is 0 below and the power of two finite above;
    // NaN casts to some integer, which the second clamp bounds, and the
    // result is NaN then anyway.
    const ValueId guess =
        clamp(out.cast(clamp(logarithm, ScalarKind::F32, out.number(-129.0), out.number(127.0)),
                       ScalarKind::I32),
              ScalarKind::I32,
              out.integer(-129),
              out.integer(127));
    const ValueId power = power_of_two(out, guess);
    const ValueId above = out.compare(CompareKind::Gt, power, magnitude);
    const ValueId below =
        out.compare(CompareKind::Le,
                    out.binary(OpKind::Mul, power, out.number(2.0), ScalarKind::F32),
                    magnitude);
    const ValueId exponent = out.binary(
        OpKind::Add,
        out.binary(OpKind::Sub, guess, out.cast(above, ScalarKind::I32), ScalarKind::I32),
        out.cast(below, ScalarKind::I32),
        ScalarKind::I32);
    const ValueId floor =
        out.binary(OpKind::Mul,
                   power,
                   out.select(above,
                              out.number(0.5),
                              out.select(below, out.number(2.0), out.number(1.0), ScalarKind::F32),
                              ScalarKind::F32),
                   ScalarKind::F32);
    const ValueId round_up =
        out.compare(CompareKind::Ge,
                    magnitude,
                    out.binary(OpKind::Mul, floor, out.number(1.5), ScalarKind::F32));
    ValueId code = out.binary(OpKind::Add,
                              out.binary(OpKind::Add, exponent, out.integer(127), ScalarKind::I32),
                              out.cast(round_up, ScalarKind::I32),
                              ScalarKind::I32);
    code = clamp(code, ScalarKind::I32, out.integer(0), out.integer(255));
    return out.select(finite, code, out.integer(255), ScalarKind::I32);
}

ValueId expand(Module& module, OpId id) {
    const Operation& cast = module.op(id);
    const sema::TypeStore& types = module.types();
    const ValueId source = cast.operands.front();
    const TypeId result = module.value(cast.results.front()).type;
    const Format from = format_of(types, module.value(source).type);
    const Format to = format_of(types, result);
    Expansion out(module, cast);
    ValueId value = source;
    if (from.any()) {
        value = from.codebook != nullptr ? codebook_values(out, source, *from.codebook)
                                         : e8m0_values(out, source);
        if (!to.any()) {
            const ValueId widened = out.emit(OpKind::Cast, {value}, result);
            out.place_before(id);
            return widened;
        }
    }
    const ValueId wide = out.cast(value, ScalarKind::F32);
    const ValueId code =
        to.codebook != nullptr ? codebook_codes(out, wide, *to.codebook) : e8m0_codes(out, wide);
    const ValueId narrowed = out.emit(OpKind::Cast, {code}, result);
    out.place_before(id);
    return narrowed;
}

} // namespace

void expand_format_casts(Module& module) {
    const sema::TypeStore& types = module.types();
    const auto is_expanded = [&](ValueId value) {
        return format_of(types, module.value(value).type).any();
    };
    std::vector<OpId> casts;
    for (const BlockId block : module.all_blocks()) {
        for (const OpId id : module.block(block).ops) {
            const Operation& op = module.op(id);
            if (op.kind == OpKind::Cast &&
                (is_expanded(op.operands.front()) || is_expanded(op.results.front()))) {
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
