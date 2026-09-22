#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/ir/ir.hpp"
#include "linnet/sema/model.hpp"
#include "linnet/source/source_manager.hpp"

#include <memory>
#include <span>

namespace linnet::ir {

// Lowers every function, op, entry, and block method of the given modules to
// Core IR. Requires a model from an analysis without errors; the result is
// verified by the caller as needed.
Module lower(const SourceManager& sources,
             std::span<const ast::Ast* const> modules,
             std::shared_ptr<sema::Model> model);

} // namespace linnet::ir
