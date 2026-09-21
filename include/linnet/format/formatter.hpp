#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/format/doc.hpp"
#include "linnet/source/source_manager.hpp"

#include <string>

namespace linnet::format {

// Renders a syntax tree in the canonical Linnet style. The tree must come from
// a parse that reported no errors. Comments are preserved; those written in
// positions the formatter does not track are moved to the next line boundary
// rather than dropped.
//
// Formatting is idempotent: formatting the output again reproduces it.
std::string
format(const ast::Ast& ast, const SourceManager& sources, const LayoutOptions& options = {});

} // namespace linnet::format
