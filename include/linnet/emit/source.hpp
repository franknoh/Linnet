#pragma once

#include "linnet/ir/ir.hpp"

#include <expected>
#include <string>

namespace linnet::emit {

// Reconstructs Linnet source for the declarations of a Core IR module's
// first source module (the one a plan describes): its functions, ops, entries, and blocks with
// their members and methods. The output is canonical `.linnet` text that parses and checks; callers
// run the formatter on it for exact repository style.
//
// SSA values become `let` bindings named after their source names where
// known, region bodies become expressions (values used several times inside a
// region are repeated, which is always valid because every operation is
// pure), and index notation is reconstructed from `comprehension`/`reduce`
// regions.
std::expected<std::string, std::string> emit_source(const ir::Module& module);

} // namespace linnet::emit
