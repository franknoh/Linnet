#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/ir/ir.hpp"

#include <expected>
#include <span>
#include <string>

namespace linnet::backend {

// Serializes a checked program as a plan for a materializer: the root block,
// its parameter manifest, and the Core IR of every function, with dimensions
// left as symbolic expressions that the materializer evaluates once the root
// block's generic arguments and the entry inputs are known.
//
// The format is documented in docs/plan-format.md; it is versioned separately
// from the language because it is consumed by code outside this repository.
struct PlanOptions {
    std::string root; // name of the root block; empty selects the only block with entries
    std::uint32_t root_module = 0;
    std::span<const ast::Ast* const> modules; // the syntax trees the IR was lowered from
};

std::expected<std::string, std::string> export_plan(ir::Module& module, const PlanOptions& options);

} // namespace linnet::backend
