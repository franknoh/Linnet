#pragma once

#include "linnet/ir/ir.hpp"

#include <expected>
#include <string>
#include <string_view>

namespace linnet::backend {

// Reads a plan document (docs/plan-format.md) back into a Core IR module
// with a synthesized analysis model: blocks with their members and methods,
// functions with their generics and constraints, and every operation of
// their bodies. Callees that the document does not define stay external
// calls by qualified name. The result passes the IR verifier.
//
// This is how a graph captured from another framework reaches the source
// emitter: the adapter writes a plan, `linnet emit` reads it and prints the
// `.linnet` module.
std::expected<ir::Module, std::string> import_plan(std::string_view text);

} // namespace linnet::backend
