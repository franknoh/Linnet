#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/diagnostic/render.hpp"

#include "test.hpp"

using namespace linnet;

namespace {

Diagnostic error_at(SourceSpan span, std::string message, std::string label = {}) {
    Diagnostic diagnostic;
    diagnostic.message = std::move(message);
    diagnostic.primary = {span, std::move(label)};
    return diagnostic;
}

} // namespace

TEST("sink: counts errors only") {
    DiagnosticSink sink;
    CHECK(!sink.has_errors());
    Diagnostic warning;
    warning.severity = Severity::Warning;
    sink.report(warning);
    CHECK(!sink.has_errors());
    sink.report(Diagnostic{});
    CHECK_EQ(sink.error_count(), 1U);
    CHECK_EQ(sink.diagnostics().size(), 2U);
}

TEST("sink: sorts by location with location-free diagnostics first") {
    DiagnosticSink sink;
    sink.report(error_at({1, 5, 6}, "c"));
    sink.report(error_at({0, 9, 10}, "b"));
    sink.report(error_at({0, 2, 3}, "a"));
    sink.report(error_at({}, "global"));
    sink.sort_by_location();
    const auto& sorted = sink.diagnostics();
    CHECK_EQ(sorted[0].message, "global");
    CHECK_EQ(sorted[1].message, "a");
    CHECK_EQ(sorted[2].message, "b");
    CHECK_EQ(sorted[3].message, "c");
}

TEST("render: primary label with code, note, and help") {
    SourceManager sources;
    const FileId file = sources.add_file("m.linnet", "fn f() {\n    let x = foo\n}\n").value();
    Diagnostic diagnostic = error_at({file, 21, 24}, "unknown name `foo`", "not found");
    diagnostic.code = "E0001";
    diagnostic.notes.push_back("names must be declared before use");
    diagnostic.help.push_back("did you mean `for`?");

    CHECK_EQ(render_diagnostic(sources, diagnostic),
             "error E0001: unknown name `foo`\n"
             "\n"
             "  --> m.linnet:2:13\n"
             "   |\n"
             " 2 |     let x = foo\n"
             "   |             ^^^ not found\n"
             "   |\n"
             "   = note: names must be declared before use\n"
             "   = help: did you mean `for`?\n");
}

TEST("render: secondary labels, line gaps, and gutter width") {
    SourceManager sources;
    std::string text = "let a = 1\n";
    for (int i = 0; i < 9; ++i) {
        text += "\n";
    }
    text += "let a = 2\n";
    const FileId file = sources.add_file("m.linnet", text).value();
    Diagnostic diagnostic = error_at({file, 23, 24}, "duplicate definition", "redefined here");
    diagnostic.secondary.push_back({{file, 4, 5}, "first defined here"});

    CHECK_EQ(render_diagnostic(sources, diagnostic),
             "error: duplicate definition\n"
             "\n"
             "  --> m.linnet:11:5\n"
             "   |\n"
             " 1 | let a = 1\n"
             "   |     - first defined here\n"
             "...\n"
             "11 | let a = 2\n"
             "   |     ^ redefined here\n");
}

TEST("render: labels in another file") {
    SourceManager sources;
    const FileId a = sources.add_file("a.linnet", "use b\n").value();
    const FileId b = sources.add_file("b.linnet", "op x\n").value();
    Diagnostic diagnostic = error_at({a, 4, 5}, "cannot import");
    diagnostic.severity = Severity::Warning;
    diagnostic.secondary.push_back({{b, 0, 2}, "declared here"});

    CHECK_EQ(render_diagnostic(sources, diagnostic),
             "warning: cannot import\n"
             "\n"
             "  --> a.linnet:1:5\n"
             "   |\n"
             " 1 | use b\n"
             "   |     ^\n"
             "  ::: b.linnet:1:1\n"
             "   |\n"
             " 1 | op x\n"
             "   | -- declared here\n");
}

TEST("render: tabs, multibyte text, and end-of-file spans") {
    SourceManager sources;
    const FileId file = sources.add_file("m.linnet", "\t\xC3\xA9 = \xFFz").value();
    // Span covers the invalid byte; the column counts code points.
    CHECK_EQ(render_diagnostic(sources, error_at({file, 6, 7}, "invalid UTF-8")),
             "error: invalid UTF-8\n"
             "\n"
             "  --> m.linnet:1:6\n"
             "   |\n"
             " 1 |     \xC3\xA9 = \xEF\xBF\xBDz\n"
             "   |         ^\n");
    // Empty span at end of file still gets one caret.
    CHECK_EQ(render_diagnostic(sources, error_at({file, 8, 8}, "unexpected end of file")),
             "error: unexpected end of file\n"
             "\n"
             "  --> m.linnet:1:8\n"
             "   |\n"
             " 1 |     \xC3\xA9 = \xEF\xBF\xBDz\n"
             "   |           ^\n");
}

TEST("render: multi-line span underlines its first line") {
    SourceManager sources;
    const FileId file = sources.add_file("m.linnet", "/* open\nnever closed").value();
    CHECK_EQ(render_diagnostic(sources, error_at({file, 0, 20}, "unterminated block comment")),
             "error: unterminated block comment\n"
             "\n"
             "  --> m.linnet:1:1\n"
             "   |\n"
             " 1 | /* open\n"
             "   | ^^^^^^^\n");
}

TEST("render: location-free diagnostic") {
    const SourceManager sources;
    Diagnostic diagnostic = error_at({}, "no package manifest found");
    diagnostic.help.push_back("run `linnet init`");
    CHECK_EQ(render_diagnostic(sources, diagnostic),
             "error: no package manifest found\n"
             "   = help: run `linnet init`\n");
}

TEST("render: color wraps output in ANSI sequences") {
    const SourceManager sources;
    const std::string rendered =
        render_diagnostic(sources, error_at({}, "boom"), RenderOptions{.color = true});
    CHECK_EQ(rendered, "\x1b[1;31merror\x1b[0m\x1b[1m: boom\x1b[0m\n");
}
