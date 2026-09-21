#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/source/source_manager.hpp"

#include <span>
#include <string>
#include <vector>

namespace linnet::sema {

// The checked type of one local binding, rendered for tools and tests.
struct BindingInfo {
    SourceSpan span;   // the bound name
    std::string owner; // enclosing function, qualified by its block if any
    std::string name;
    std::string type;
};

struct AnalysisResult {
    std::vector<BindingInfo> bindings;
};

// Resolves names and checks types, shapes, and tensor algebra for a set of
// modules that were parsed without errors. Imports are resolved among the
// given modules by their declared module paths. All findings go to `sink`;
// analysis itself never fails.
AnalysisResult analyze(const SourceManager& sources,
                       std::span<const ast::Ast* const> modules,
                       DiagnosticSink& sink);

} // namespace linnet::sema
