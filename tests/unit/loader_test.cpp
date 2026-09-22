#include "linnet/module/loader.hpp"
#include "linnet/sema/analysis.hpp"

#include "test.hpp"

#include <filesystem>
#include <string>
#include <vector>

using namespace linnet;

namespace {

std::filesystem::path fixture(const std::string& relative) {
    return std::filesystem::path(LINNET_SPEC_TESTS_DIR) / "fixtures" / relative;
}

struct Loaded {
    SourceManager sources;
    DiagnosticSink sink;
    Program program;

    Loaded(const std::filesystem::path& file, const std::filesystem::path& std_root) {
        const std::vector<std::filesystem::path> files{file};
        program = load_program(sources, files, LoaderOptions{std_root, {}}, sink);
        if (!sink.has_errors()) {
            const std::vector<const ast::Ast*> modules = program.module_pointers();
            sema::analyze(sources, modules, sink, &program.imports);
        }
    }

    std::string codes() const {
        std::string out;
        for (const Diagnostic& diagnostic : sink.diagnostics()) {
            out += (out.empty() ? "" : " ") + diagnostic.code;
        }
        return out;
    }
};

} // namespace

TEST("loader: finds the package root") {
    const std::filesystem::path root = find_package_root(fixture("pkg/src/layers/scale.linnet"));
    CHECK(std::filesystem::equivalent(root, fixture("pkg")));
    CHECK(find_package_root(fixture("std/nn/identity.linnet")).empty());
}

TEST("loader: follows crate, std, and dependency imports transitively") {
    const Loaded loaded(fixture("pkg/src/main.linnet"), fixture("std"));
    CHECK_EQ(loaded.codes(), "");
    CHECK_EQ(loaded.program.modules.size(), 6U);
    CHECK_EQ(loaded.program.imports.size(), 5U);
}

TEST("loader: a malformed manifest is reported once") {
    const Loaded loaded(fixture("broken_manifest/src/lib.linnet"), {});
    CHECK_EQ(loaded.codes(), "E5001");
}

TEST("loader: a missing standard library is an unknown module, reported once") {
    const Loaded loaded(fixture("pkg/src/main.linnet"), {});
    CHECK_EQ(loaded.codes(), "E1202");
}

TEST("loader: a missing file is reported") {
    const Loaded loaded(fixture("pkg/src/absent.linnet"), {});
    CHECK(loaded.sink.has_errors());
    CHECK(loaded.program.modules.empty());
}
