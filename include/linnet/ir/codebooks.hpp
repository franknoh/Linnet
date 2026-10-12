#pragma once

#include "linnet/ir/ir.hpp"

namespace linnet::ir {

// Rewrites each cast from or to a codebook format (spec §3.1), whose tensors
// hold their carrier's codes, as operations every target runs: widened,
// each code to its value (a `select` per code, so each value is exact);
// narrowed, to the code of the nearest value, the count of the midpoints
// below it (a tie to the lower code, NaN to the first). Only casts of the
// codes remain, which targets and the interpreter take as integers. A plan
// or a target reads the module after this; emitted source keeps the casts,
// which source spells only by value.
void expand_codebook_casts(Module& module);

} // namespace linnet::ir
