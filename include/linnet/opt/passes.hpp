#pragma once

#include "linnet/ir/ir.hpp"

#include <functional>
#include <string>
#include <vector>

namespace linnet::opt {

// How far a rewrite may change floating-point results. Only Exact rewrites
// run by default.
enum class Legality : std::uint8_t {
    Exact,                 // bit-identical results
    IEEEEquivalent,        // identical under IEEE semantics, e.g. removing a copy
    NumericallyEquivalent, // reassociation and the like
    Approximate,           // changes results within a tolerance
};

// A pass changes no result: a rewrite that may (`Legality` above Exact) is
// one a pass applies only when its options allow it.
struct Pass {
    std::string name;
    // Returns true when the module was changed.
    std::function<bool(ir::Module&)> run;
};

// Runs the passes repeatedly, in order, until a full round changes nothing
// (eight rounds at most), verifying the IR after each pass that changed
// it. A verifier failure throws std::logic_error naming the pass: a
// miscompiled module must never be used.
void run_pipeline(ir::Module& module, const std::vector<Pass>& passes);

// The canonical pipeline: canonicalization, common-subexpression
// elimination, and dead-code elimination.
std::vector<Pass> canonical_passes();

// Individual passes.
bool eliminate_dead_code(ir::Module& module);
bool eliminate_common_subexpressions(ir::Module& module);
bool canonicalize(ir::Module& module);

} // namespace linnet::opt
