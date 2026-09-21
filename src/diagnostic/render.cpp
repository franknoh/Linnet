#include "linnet/diagnostic/render.hpp"

#include "linnet/source/utf8.hpp"

#include <algorithm>
#include <string_view>
#include <tuple>
#include <vector>

namespace linnet {

namespace {

struct Style {
    std::string_view severity;
    std::string_view gutter;
    std::string_view bold;
    std::string_view reset;
};

Style make_style(Severity severity, bool color) {
    if (!color) {
        return {};
    }
    std::string_view severity_color = "\x1b[1;31m";
    if (severity == Severity::Warning) {
        severity_color = "\x1b[1;33m";
    } else if (severity == Severity::Note) {
        severity_color = "\x1b[1;36m";
    }
    return {severity_color, "\x1b[1;34m", "\x1b[1m", "\x1b[0m"};
}

struct PlacedLabel {
    const Label* label;
    bool is_primary;
    std::uint32_t line; // 1-based
};

// A source line prepared for display: tabs expanded, undecodable bytes and
// control characters replaced, plus the display column of every byte offset.
struct DisplayLine {
    std::string text;
    std::vector<std::uint32_t> columns; // size = line bytes + 1
};

DisplayLine make_display_line(std::string_view line, std::uint32_t tab_width) {
    DisplayLine display;
    display.columns.reserve(line.size() + 1);
    std::uint32_t column = 0;
    std::size_t offset = 0;
    while (offset < line.size()) {
        const DecodedCodePoint decoded = decode_utf8(line, offset);
        for (std::uint32_t i = 0; i < decoded.length; ++i) {
            display.columns.push_back(column);
        }
        if (decoded.value == U'\t') {
            display.text.append(tab_width, ' ');
            column += tab_width;
        } else if (!decoded.valid || decoded.value < 0x20 || decoded.value == 0x7F) {
            display.text += "\xEF\xBF\xBD";
            ++column;
        } else {
            display.text += line.substr(offset, decoded.length);
            ++column;
        }
        offset += decoded.length;
    }
    display.columns.push_back(column);
    return display;
}

std::uint32_t code_point_column(std::string_view line, std::uint32_t byte_offset) {
    std::uint32_t column = 1;
    std::size_t offset = 0;
    while (offset < line.size() && offset < byte_offset) {
        offset += decode_utf8(line, offset).length;
        ++column;
    }
    return column;
}

std::uint32_t digit_count(std::uint32_t value) {
    std::uint32_t digits = 1;
    while (value >= 10) {
        value /= 10;
        ++digits;
    }
    return digits;
}

class Renderer {
public:
    Renderer(const SourceManager& sources,
             const Diagnostic& diagnostic,
             const RenderOptions& options)
        : sources_(sources), diagnostic_(diagnostic), options_(options),
          style_(make_style(diagnostic.severity, options.color)) {}

    std::string render() {
        render_header();
        if (diagnostic_.has_location()) {
            place_labels();
            render_excerpts();
        }
        render_footers("note", diagnostic_.notes);
        render_footers("help", diagnostic_.help);
        return std::move(out_);
    }

private:
    void render_header() {
        out_ += style_.severity;
        out_ += severity_name(diagnostic_.severity);
        if (!diagnostic_.code.empty()) {
            out_ += '[';
            out_ += diagnostic_.code;
            out_ += ']';
        }
        out_ += style_.reset;
        out_ += style_.bold;
        out_ += ": ";
        out_ += diagnostic_.message;
        out_ += style_.reset;
        out_ += '\n';
    }

    void place(const Label& label, bool is_primary) {
        if (label.span.file == invalid_file_id) {
            return;
        }
        const std::uint32_t line = sources_.line_column(label.span.file, label.span.begin).line;
        labels_.push_back({&label, is_primary, line});
        gutter_width_ = std::max(gutter_width_, digit_count(line));
    }

    void place_labels() {
        place(diagnostic_.primary, true);
        for (const Label& label : diagnostic_.secondary) {
            place(label, false);
        }
        const FileId primary_file = diagnostic_.primary.span.file;
        std::stable_sort(
            labels_.begin(), labels_.end(), [&](const PlacedLabel& a, const PlacedLabel& b) {
                const auto key = [&](const PlacedLabel& placed) {
                    const SourceSpan& span = placed.label->span;
                    return std::tuple(
                        span.file != primary_file, span.file, placed.line, span.begin);
                };
                return key(a) < key(b);
            });
    }

    void gutter(std::string_view number, char separator) {
        out_ += style_.gutter;
        out_.append(gutter_width_ - std::min<std::size_t>(gutter_width_, number.size()), ' ');
        out_ += number;
        out_ += ' ';
        out_ += separator;
        out_ += style_.reset;
    }

    void render_location(const PlacedLabel& placed, bool is_primary_file) {
        const SourceSpan& span = placed.label->span;
        const std::string_view line = sources_.line_text(span.file, placed.line);
        const std::uint32_t byte_column = span.begin - sources_.line_start(span.file, placed.line);

        out_.append(gutter_width_, ' ');
        out_ += style_.gutter;
        out_ += is_primary_file ? "-->" : ":::";
        out_ += style_.reset;
        out_ += ' ';
        out_ += sources_.path(span.file);
        out_ += ':';
        out_ += std::to_string(placed.line);
        out_ += ':';
        out_ += std::to_string(code_point_column(line, byte_column));
        out_ += '\n';
    }

    void render_marker(const PlacedLabel& placed, const DisplayLine& display) {
        const SourceSpan& span = placed.label->span;
        const std::uint32_t line_begin = sources_.line_start(span.file, placed.line);
        const auto line_size = static_cast<std::uint32_t>(display.columns.size() - 1);
        const std::uint32_t begin = std::min(span.begin - line_begin, line_size);
        const std::uint32_t end = std::clamp(span.end - line_begin, begin, line_size);
        const std::uint32_t first = display.columns[begin];
        const std::uint32_t width = std::max<std::uint32_t>(1, display.columns[end] - first);

        gutter("", '|');
        out_ += ' ';
        out_.append(first, ' ');
        out_ += placed.is_primary ? style_.severity : style_.gutter;
        out_.append(width, placed.is_primary ? '^' : '-');
        if (!placed.label->message.empty()) {
            out_ += ' ';
            out_ += placed.label->message;
        }
        out_ += style_.reset;
        out_ += '\n';
    }

    void render_excerpts() {
        FileId current_file = invalid_file_id;
        std::uint32_t current_line = 0;
        DisplayLine display;

        for (const PlacedLabel& placed : labels_) {
            const FileId file = placed.label->span.file;
            if (file != current_file) {
                // The primary file is headed by the primary label's position even
                // when a secondary label precedes it.
                const auto primary = std::find_if(
                    labels_.begin(), labels_.end(), [](const auto& l) { return l.is_primary; });
                const bool is_primary_file = file == diagnostic_.primary.span.file;
                render_location(is_primary_file ? *primary : placed, is_primary_file);
                gutter("", '|');
                out_ += '\n';
                current_file = file;
                current_line = 0;
            }
            if (placed.line != current_line) {
                if (current_line != 0 && placed.line > current_line + 1) {
                    out_ += style_.gutter;
                    out_ += "...";
                    out_ += style_.reset;
                    out_ += '\n';
                }
                display =
                    make_display_line(sources_.line_text(file, placed.line), options_.tab_width);
                gutter(std::to_string(placed.line), '|');
                if (!display.text.empty()) {
                    out_ += ' ';
                    out_ += display.text;
                }
                out_ += '\n';
                current_line = placed.line;
            }
            render_marker(placed, display);
        }
    }

    void render_footers(std::string_view kind, const std::vector<std::string>& entries) {
        for (const std::string& entry : entries) {
            gutter("", '=');
            out_ += ' ';
            out_ += style_.bold;
            out_ += kind;
            out_ += style_.reset;
            out_ += ": ";
            out_ += entry;
            out_ += '\n';
        }
    }

    const SourceManager& sources_;
    const Diagnostic& diagnostic_;
    const RenderOptions& options_;
    Style style_;
    std::vector<PlacedLabel> labels_;
    std::uint32_t gutter_width_ = 1;
    std::string out_;
};

} // namespace

std::string render_diagnostic(const SourceManager& sources,
                              const Diagnostic& diagnostic,
                              const RenderOptions& options) {
    return Renderer(sources, diagnostic, options).render();
}

} // namespace linnet
