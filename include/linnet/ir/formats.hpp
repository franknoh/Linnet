#pragma once

#include "linnet/ir/ir.hpp"

namespace linnet::ir {

// Rewrites each cast from or to a codebook format or E8M0 (spec §3.1), whose
// tensors hold their codes (E8M0, its byte), as operations every target
// runs. A codebook widened selects each code's value (a `select` per code,
// so each value is exact); narrowed, it takes the code of the nearest value,
// the count of the midpoints below it (a tie to the lower code, NaN to the
// first). E8M0 widened is `2 ** (byte - 127)` from integer shifts, and NaN
// for 255; narrowed, the power of two PyTorch rounds to. Only casts of the
// codes remain, which targets and the interpreter take as integers. A plan
// or a target reads the module after this; emitted source keeps the casts,
// which source spells only by value.
void expand_format_casts(Module& module);

} // namespace linnet::ir
