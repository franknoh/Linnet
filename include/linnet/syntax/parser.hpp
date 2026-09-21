#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/source/source_manager.hpp"

namespace linnet {

// Lexes and parses one file. Always returns a tree: syntax errors are reported
// to `sink`, the parser resynchronizes, and damaged regions become Error
// nodes so that later stages can skip them without cascading diagnostics.
ast::Ast parse(const SourceManager& sources, FileId file, DiagnosticSink& sink);

} // namespace linnet
