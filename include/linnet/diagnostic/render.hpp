#pragma once

#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/source/source_manager.hpp"

#include <string>

namespace linnet {

struct RenderOptions {
    bool color = false;
    std::uint32_t tab_width = 4;
};

// Renders one diagnostic as human-readable text with source excerpts:
//
//   error E0001: message
//
//     --> path.linnet:3:5
//      |
//    3 | let x = foo
//      |     ^ primary label
//      |         --- secondary label
//      |
//      = note: ...
//      = help: ...
//
// Reported columns are 1-based and count Unicode code points. Spans covering
// several lines are underlined on their first line only.
std::string render_diagnostic(const SourceManager& sources,
                              const Diagnostic& diagnostic,
                              const RenderOptions& options = {});

} // namespace linnet
