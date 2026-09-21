#include "linnet/diagnostic/json.hpp"

#include "linnet/source/utf8.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

namespace linnet {

std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (std::size_t offset = 0; offset < text.size();) {
        const DecodedCodePoint decoded = decode_utf8(text, offset);
        const char c = text[offset];
        if (!decoded.valid) {
            out += "\\ufffd"; // JSON must be valid Unicode
        } else if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\r') {
            out += "\\r";
        } else if (decoded.value < 0x20) {
            std::array<char, 8> escape{};
            std::snprintf(
                escape.data(), escape.size(), "\\u%04x", static_cast<unsigned>(decoded.value));
            out += escape.data();
        } else {
            out += text.substr(offset, decoded.length);
        }
        offset += decoded.length;
    }
    return out + "\"";
}

namespace {

std::string position(const SourceManager& sources, FileId file, std::uint32_t offset) {
    const LineColumn where = sources.line_column(file, offset);
    const std::string_view line = sources.line_text(file, where.line);
    std::uint32_t column = 1;
    for (std::size_t at = 0; at < line.size() && at + 1 < where.column;) {
        at += decode_utf8(line, at).length;
        ++column;
    }
    return "{\"line\":" + std::to_string(where.line) + ",\"column\":" + std::to_string(column) +
           ",\"offset\":" + std::to_string(std::min(offset, sources.size(file))) + "}";
}

std::string location(const SourceManager& sources, SourceSpan span) {
    if (span.file == invalid_file_id) {
        return "null";
    }
    return "{\"file\":" + json_string(sources.path(span.file)) +
           ",\"start\":" + position(sources, span.file, span.begin) +
           ",\"end\":" + position(sources, span.file, span.end) + "}";
}

std::string string_array(const std::vector<std::string>& values) {
    std::string out = "[";
    for (const std::string& value : values) {
        out += out.size() == 1 ? "" : ",";
        out += json_string(value);
    }
    return out + "]";
}

} // namespace

std::string render_json(const SourceManager& sources, std::span<const Diagnostic> diagnostics) {
    std::size_t errors = 0;
    std::size_t warnings = 0;
    std::string out = "{\"version\":1,\"diagnostics\":[";
    bool is_first = true;
    for (const Diagnostic& diagnostic : diagnostics) {
        errors += diagnostic.severity == Severity::Error ? 1 : 0;
        warnings += diagnostic.severity == Severity::Warning ? 1 : 0;
        out += is_first ? "" : ",";
        is_first = false;
        out += "{\"code\":" + json_string(diagnostic.code);
        out += ",\"severity\":" + json_string(severity_name(diagnostic.severity));
        out += ",\"message\":" + json_string(diagnostic.message);
        out += ",\"label\":" + json_string(diagnostic.primary.message);
        out += ",\"location\":" + location(sources, diagnostic.primary.span);
        out += ",\"related\":[";
        for (std::size_t i = 0; i < diagnostic.secondary.size(); ++i) {
            out += i == 0 ? "" : ",";
            out += "{\"location\":" + location(sources, diagnostic.secondary[i].span) +
                   ",\"message\":" + json_string(diagnostic.secondary[i].message) + "}";
        }
        out += "],\"notes\":" + string_array(diagnostic.notes);
        out += ",\"help\":" + string_array(diagnostic.help) + "}";
    }
    out += "],\"summary\":{\"errors\":" + std::to_string(errors) +
           ",\"warnings\":" + std::to_string(warnings) + "}}\n";
    return out;
}

} // namespace linnet
