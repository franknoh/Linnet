#pragma once

#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/source/source_manager.hpp"

#include <span>
#include <string>
#include <string_view>

namespace linnet {

// Machine-readable diagnostics for `--json`, documented in docs/tooling.md:
//
//   { "version": 1,
//     "diagnostics": [ { "code", "severity", "message", "label",
//                        "location": { "file", "start", "end" } | null,
//                        "related": [ { "location", "message" } ],
//                        "notes": [...], "help": [...] } ],
//     "summary": { "errors": n, "warnings": n } }
//
// Positions are { "line", "column", "offset" }: lines and columns are 1-based,
// columns count Unicode code points, offsets are 0-based bytes. `end` is
// exclusive. The output is a single line terminated by a newline.
std::string render_json(const SourceManager& sources, std::span<const Diagnostic> diagnostics);

// `text` as a JSON string literal, including the quotes.
std::string json_string(std::string_view text);

} // namespace linnet
