#include "linnet/diagnostic/diagnostic.hpp"

#include <algorithm>
#include <tuple>
#include <utility>

namespace linnet {

std::string_view severity_name(Severity severity) {
    switch (severity) {
    case Severity::Error:
        return "error";
    case Severity::Warning:
        return "warning";
    case Severity::Note:
        return "note";
    }
    return "error";
}

void DiagnosticSink::report(Diagnostic diagnostic) {
    if (diagnostic.severity == Severity::Error) {
        ++error_count_;
    }
    diagnostics_.push_back(std::move(diagnostic));
}

void DiagnosticSink::sort_by_location() {
    const auto key = [](const Diagnostic& diagnostic) {
        const SourceSpan& span = diagnostic.primary.span;
        // invalid_file_id + 1 wraps to 0, which sorts location-free entries first.
        return std::tuple(static_cast<FileId>(span.file + 1), span.begin, span.end);
    };
    std::stable_sort(diagnostics_.begin(),
                     diagnostics_.end(),
                     [&](const Diagnostic& a, const Diagnostic& b) { return key(a) < key(b); });
}

} // namespace linnet
