#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/source/source_manager.hpp"

#include <string>

namespace linnet::ast {

// Indented tree rendering of a parsed file for `linnet inspect --ast` and
// golden tests. The format is a debugging aid and is not stable.
std::string dump(const Ast& ast, const SourceManager& sources);

} // namespace linnet::ast
