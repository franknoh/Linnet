#pragma once

#include "linnet/source/source_manager.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace linnet {

enum class Severity : std::uint8_t { Error, Warning, Note };

std::string_view severity_name(Severity severity);

// A source range with an explanatory message, shown beneath the source line.
struct Label {
    SourceSpan span;
    std::string message;
};

struct Diagnostic {
    Severity severity = Severity::Error;
    std::string code; // stable identifier such as "E2201"; may be empty
    std::string message;
    Label primary; // primary.span.file may be invalid_file_id for location-free diagnostics
    std::vector<Label> secondary;
    std::vector<std::string> notes;
    std::vector<std::string> help;

    bool has_location() const { return primary.span.file != invalid_file_id; }
};

// Collects diagnostics produced by compiler stages. Stages report and carry
// on; nothing here throws or aborts compilation.
class DiagnosticSink {
public:
    void report(Diagnostic diagnostic);

    const std::vector<Diagnostic>& diagnostics() const { return diagnostics_; }
    std::size_t error_count() const { return error_count_; }
    bool has_errors() const { return error_count_ != 0; }

    // Orders diagnostics by file and position, keeping report order for ties
    // and placing location-free diagnostics first.
    void sort_by_location();

private:
    std::vector<Diagnostic> diagnostics_;
    std::size_t error_count_ = 0;
};

} // namespace linnet
