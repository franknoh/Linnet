// Runs the executable specification through the whole frontend: every case
// marked `ok` must check without diagnostics, and every other case must report
// exactly its expected code. The snapshots under diagnostics/ pin down the
// rendered text of selected errors.

#include "linnet/diagnostic/render.hpp"
#include "linnet/package/spec_manifest.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace linnet;

namespace {

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
    const std::vector<SpecCase> cases = read_spec_manifest(root + "/manifest.toml").value();
    CHECK(cases.size() >= 28);
    for (const SpecCase& spec_case : cases) {
        std::vector<std::string> codes;
        const std::string rendered =
            check_and_render(spec_case.file, read_file(root + "/" + spec_case.file), codes);
        if (!spec_case.expects_error) {
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
