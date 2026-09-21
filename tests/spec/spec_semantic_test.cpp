// Runs the executable specification through the whole frontend: every case
// marked `ok` must check without diagnostics, and every other case must report
// exactly its expected code. The snapshots under diagnostics/ pin down the
// rendered text of selected errors.

#include "linnet/diagnostic/render.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <filesystem>
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

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

// Checks one file registered under `display_path` and renders its diagnostics
// the way the command line does.
std::string check_and_render(const std::string& display_path,
                             const std::string& text,
                             std::vector<std::string>& codes) {
    SourceManager sources;
    DiagnosticSink sink;
    const FileId file = sources.add_file(display_path, text).value();
    const ast::Ast tree = parse(sources, file, sink);
    if (!sink.has_errors()) {
        const ast::Ast* const modules[] = {&tree};
        sema::analyze(sources, modules, sink);
    }
    sink.sort_by_location();
    std::string rendered;
    for (const Diagnostic& diagnostic : sink.diagnostics()) {
        codes.push_back(diagnostic.code);
        rendered += rendered.empty() ? "" : "\n";
        rendered += render_diagnostic(sources, diagnostic);
    }
    return rendered;
}

} // namespace

TEST("spec: every case is accepted or rejected with its code") {
    const std::string root = LINNET_SPEC_TESTS_DIR;
    const std::vector<Case> cases = read_manifest(root + "/manifest.toml");
    CHECK(cases.size() >= 28);
    for (const Case& spec_case : cases) {
        std::vector<std::string> codes;
        const std::string rendered =
            check_and_render(spec_case.file, read_file(root + "/" + spec_case.file), codes);
        if (spec_case.code.empty()) {
            if (!codes.empty()) {
                linnet::test::report_failure(
                    spec_case.file.c_str(), 0, "expected no diagnostics, got:\n" + rendered);
            }
            continue;
        }
        if (codes.empty()) {
            linnet::test::report_failure(
                spec_case.file.c_str(), 0, "expected " + spec_case.code + ", got no diagnostics");
        }
        for (const std::string& code : codes) {
            if (code != spec_case.code) {
                linnet::test::report_failure(spec_case.file.c_str(),
                                             0,
                                             "expected only " + spec_case.code + ", got:\n" +
                                                 rendered);
                break;
            }
        }
    }
}

TEST("spec: diagnostic snapshots") {
    const std::filesystem::path root = LINNET_SPEC_TESTS_DIR;
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root / "diagnostics")) {
        if (entry.path().extension() != ".stderr") {
            continue;
        }
        ++count;
        const std::string name = "invalid/" + entry.path().stem().string() + ".linnet";
        std::vector<std::string> codes;
        const std::string rendered = check_and_render(name, read_file(root / name), codes);
        CHECK_EQ(rendered, read_file(entry.path()));
    }
    CHECK(count >= 3);
}
