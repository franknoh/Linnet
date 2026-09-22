#pragma once

#include "linnet/ir/ir.hpp"
#include "linnet/opt/passes.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace linnet::opt {

// Equality saturation over the region-free operations of each block.
//
// Every value becomes an e-class; rewrites add equivalent terms to classes
// until nothing new appears or the budget is spent; extraction then picks the
// cheapest term of each class under a shape-based cost model and rebuilds
// the block. Operations with regions (comprehensions, reductions, matches)
// and block arguments are leaves: the rewrites never look inside them.
//
// Each rule carries a legality class, and `saturate` applies only the rules
// within `allowed`, so the default (`Exact`) never changes floating-point
// results; reassociation and distribution need `NumericallyEquivalent`.
struct Rewrite {
    std::string name;
    Legality legality;
};

// The rules the saturation pass knows, for `linnet explain` and tests.
std::vector<Rewrite> rewrites();

struct SaturationOptions {
    Legality allowed = Legality::Exact;
    std::size_t max_nodes = 20000; // stop growing the graph beyond this
    int max_rounds = 8;            // rounds of matching and merging
    // Symbolic dimensions count this much in the cost model.
    std::int64_t symbolic_extent = 256;
};

// Returns true when the module changed.
bool saturate(ir::Module& module, const SaturationOptions& options = {});

// A pass wrapping `saturate` with the given policy; Exact for the pipeline,
// since it only applies rules within `allowed`.
Pass saturation_pass(Legality allowed);

// The canonical passes followed by saturation under `allowed`.
std::vector<Pass> optimizing_passes(Legality allowed);

} // namespace linnet::opt
