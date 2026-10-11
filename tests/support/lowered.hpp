#pragma once

// One module of source taken through the compiler's front half, for tests
// of what comes after: parsed, checked and lowered to Core IR.

#include "linnet/ir/ir.hpp"
#include "linnet/ir/lower.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

namespace linnet::test {

// `use(module)` for `source` (one module, `m.linnet`) lowered, while
// everything the module refers to is alive. Fails the test, and returns an
// empty result, when the source does not check.
template <typename Use>
std::invoke_result_t<Use, ir::Module&> with_lowered(const std::string& source, Use use) {
    SourceManager sources;
    DiagnosticSink sink;
    const FileId file = sources.add_file("m.linnet", source).value();
    const ast::Ast tree = parse(sources, file, sink);
    const ast::Ast* const modules[] = {&tree};
    const sema::AnalysisResult analysis = sema::analyze(sources, modules, sink);
    CHECK(!sink.has_errors());
    if (sink.has_errors()) {
        return {};
    }
    ir::Module module = ir::lower(sources, modules, analysis.model);
    return use(module);
}

// How many times `needle` occurs in `text`, overlapping ones included.
inline std::size_t count(const std::string& text, const std::string& needle) {
    std::size_t total = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + 1)) {
        ++total;
    }
    return total;
}

} // namespace linnet::test
