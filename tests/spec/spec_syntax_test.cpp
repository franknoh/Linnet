// Runs the syntax-level portion of the executable specification: every case
// must lex and parse, and cases whose expected diagnostic is lexical or
// syntactic (E10xx, E11xx) must report exactly that code. Cases expecting
// semantic errors must still parse cleanly.

#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <fstream>
#include <string>
#include <vector>

using namespace linnet;

namespace {

struct Case {
    std::string file;
    std::string code; // empty when the case expects success
};

// Reads the `file` and `code` keys of each [[case]] table. The manifest is a
// flat list of string keys, so no general TOML support is needed here.
std::vector<Case> read_manifest(const std::string& path) {
    std::vector<Case> cases;
    std::ifstream stream(path);
    std::string line;
    const auto quoted = [](const std::string& text) {
        const std::size_t open = text.find('"');
        const std::size_t close = text.rfind('"');
        return open < close ? text.substr(open + 1, close - open - 1) : std::string();
    };
    while (std::getline(stream, line)) {
        if (line.starts_with("[[case]]")) {
            cases.emplace_back();
        } else if (!cases.empty() && line.starts_with("file")) {
            cases.back().file = quoted(line);
        } else if (!cases.empty() && line.starts_with("code")) {
            cases.back().code = quoted(line);
        }
    }
    return cases;
}

bool is_syntax_code(const std::string& code) {
    return code.starts_with("E10") || code.starts_with("E11");
}

} // namespace

TEST("spec: syntax conformance") {
    const std::string root = LINNET_SPEC_TESTS_DIR;
    const std::vector<Case> cases = read_manifest(root + "/manifest.toml");
    CHECK(cases.size() >= 25);

    for (const Case& spec_case : cases) {
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

// Every truncation of every case must be handled without crashing, hanging,
// or flooding diagnostics.
TEST("spec: truncated inputs are handled gracefully") {
    const std::string root = LINNET_SPEC_TESTS_DIR;
    for (const Case& spec_case : read_manifest(root + "/manifest.toml")) {
        std::ifstream stream(root + "/" + spec_case.file, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(stream),
                               std::istreambuf_iterator<char>()};
        for (std::size_t length = 0; length < text.size(); ++length) {
            SourceManager sources;
            DiagnosticSink sink;
            const FileId file = sources.add_file(spec_case.file, text.substr(0, length)).value();
            parse(sources, file, sink);
            CHECK(sink.diagnostics().size() <= 6);
        }
    }
}
