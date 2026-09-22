// Runs the syntax-level portion of the executable specification: every case
// must lex and parse, and cases whose expected diagnostic is lexical or
// syntactic (E10xx, E11xx) must report exactly that code. Cases expecting
// semantic errors must still parse cleanly.

#include "linnet/format/formatter.hpp"
#include "linnet/package/spec_manifest.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <fstream>
#include <string>
#include <vector>

using namespace linnet;

namespace {

bool is_syntax_code(const std::string& code) {
    return code.starts_with("E10") || code.starts_with("E11");
}

} // namespace

TEST("spec: syntax conformance") {
    const std::string root = LINNET_SPEC_TESTS_DIR;
    const std::vector<SpecCase> cases = read_spec_manifest(root + "/manifest.toml").value();
    CHECK(cases.size() >= 25);

    for (const SpecCase& spec_case : cases) {
        SourceManager sources;
        DiagnosticSink sink;
        const auto file = sources.load_file(root + "/" + spec_case.file);
        CHECK(file.has_value());
        if (!file) {
            continue;
        }
        parse(sources, *file, sink);

        if (is_syntax_code(spec_case.code)) {
            CHECK(sink.has_errors());
            for (const Diagnostic& diagnostic : sink.diagnostics()) {
                CHECK_EQ(diagnostic.code, spec_case.code);
            }
        } else {
            for (const Diagnostic& diagnostic : sink.diagnostics()) {
                linnet::test::report_failure(
                    spec_case.file.c_str(), 0, "unexpected diagnostic: " + diagnostic.message);
            }
        }
    }
}

TEST("spec: fixtures parse") {
    const std::string root = LINNET_SPEC_TESTS_DIR;
    for (const char* path : {"/fixtures/pkg/src/lib.linnet", "/fixtures/std/nn/identity.linnet"}) {
        SourceManager sources;
        DiagnosticSink sink;
        const auto file = sources.load_file(root + path);
        CHECK(file.has_value());
        if (file) {
            parse(sources, *file, sink);
            CHECK(sink.diagnostics().empty());
        }
    }
}

// The executable specification doubles as the reference for canonical style.
TEST("spec: cases are formatter-clean") {
    const std::string root = LINNET_SPEC_TESTS_DIR;
    const std::vector<SpecCase> cases = read_spec_manifest(root + "/manifest.toml").value();
    for (const SpecCase& spec_case : cases) {
        SourceManager sources;
        DiagnosticSink sink;
        const auto file = sources.load_file(root + "/" + spec_case.file);
        if (!file) {
            continue;
        }
        const ast::Ast tree = parse(sources, *file, sink);
        if (sink.has_errors()) {
            continue; // cases with syntax errors cannot be formatted
        }
        const std::string formatted = format::format(tree, sources);
        if (formatted != sources.contents(*file)) {
            linnet::test::report_failure(spec_case.file.c_str(), 0, "is not formatter-clean");
        }
    }
}

// Every truncation of every case must be handled without crashing, hanging,
// or flooding diagnostics.
TEST("spec: truncated inputs are handled gracefully") {
    const std::string root = LINNET_SPEC_TESTS_DIR;
    const std::vector<SpecCase> cases = read_spec_manifest(root + "/manifest.toml").value();
    for (const SpecCase& spec_case : cases) {
        std::ifstream stream(root + "/" + spec_case.file, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(stream),
                               std::istreambuf_iterator<char>()};
        for (std::size_t length = 0; length < text.size(); ++length) {
            SourceManager sources;
            DiagnosticSink sink;
            const FileId file = sources.add_file(spec_case.file, text.substr(0, length)).value();
            const ast::Ast tree = parse(sources, file, sink);
            CHECK(sink.diagnostics().size() <= 6);
            if (sink.has_errors()) {
                continue;
            }
            // Whatever still parses must format to a fixed point.
            const std::string once = format::format(tree, sources);
            DiagnosticSink second_sink;
            const FileId second = sources.add_file(spec_case.file, once).value();
            const ast::Ast second_tree = parse(sources, second, second_sink);
            CHECK(!second_sink.has_errors());
            CHECK_EQ(format::format(second_tree, sources), once);
        }
    }
}
