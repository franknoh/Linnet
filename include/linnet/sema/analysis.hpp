#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/source/source_manager.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <utility>
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

// Where each `use` declaration leads, as decided by the module loader: the key
// is (importing module, index of the `use`), the value the imported module.
using ImportTable = std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>;

// Resolves names and checks types, shapes, and tensor algebra for a set of
// modules that were parsed without errors. Imports follow `imports` when it is
// given; otherwise they are resolved among the given modules by their declared
// module paths. All findings go to `sink`; analysis itself never fails.
AnalysisResult analyze(const SourceManager& sources,
                       std::span<const ast::Ast* const> modules,
                       DiagnosticSink& sink,
                       const ImportTable* imports = nullptr);

} // namespace linnet::sema
