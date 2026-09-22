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

struct Pass {
    std::string name;
    Legality legality = Legality::Exact;
    // Returns true when the module was changed.
    std::function<bool(ir::Module&)> run;
};

struct PassResult {
    std::string name;
    bool has_changed = false;
};

struct PipelineOptions {
    Legality allowed = Legality::Exact; // passes above this strength are skipped
    bool verify = true;                 // run the IR verifier after each pass
    int max_iterations = 8;             // repeat until nothing changes
};

// Runs the passes repeatedly, in order, until a full round changes nothing.
// A verifier failure throws std::logic_error naming the pass: a miscompiled
// module must never be used.
std::vector<PassResult> run_pipeline(ir::Module& module,
                                     const std::vector<Pass>& passes,
                                     const PipelineOptions& options = {});

// The canonical pipeline: canonicalization, common-subexpression
// elimination, and dead-code elimination. All are Exact.
std::vector<Pass> canonical_passes();

// Individual passes.
bool eliminate_dead_code(ir::Module& module);
bool eliminate_common_subexpressions(ir::Module& module);
bool canonicalize(ir::Module& module);

} // namespace linnet::opt
