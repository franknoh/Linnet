#include "linnet/ast/dump.hpp"
#include "linnet/backend/plan.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/diagnostic/json.hpp"
#include "linnet/diagnostic/render.hpp"
#include "linnet/format/formatter.hpp"
#include "linnet/ir/lower.hpp"
#include "linnet/lsp/server.hpp"
#include "linnet/module/loader.hpp"
#include "linnet/package/manifest.hpp"
#include "linnet/package/spec_manifest.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/source/source_manager.hpp"
#include "linnet/syntax/lexer.hpp"
#include "linnet/syntax/parser.hpp"
#include "linnet/version.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

using namespace linnet;

// Exit statuses are part of the CLI contract and must stay stable.
constexpr int exit_success = 0;
constexpr int exit_failure = 1; // the input has errors
constexpr int exit_usage = 2;   // the command line is wrong

bool stderr_is_terminal() {
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}

void print_usage(std::FILE* out) {
    std::fputs("Usage: linnet <command> [options]\n"
               "\n"
               "Commands:\n"
               "  plan [--root <Block>] [--std <dir>] <file>\n"
               "                                       Print the materializer plan (JSON) of a\n"
               "                                       root block and everything it uses\n"
               "  init [<dir>]                         Create linnet.toml and src/lib.linnet\n"
               "  check [options] <path>...            Check syntax, types, and shapes of the\n"
               "                                       given files and everything they import\n"
               "  lint [--std <dir>] <path>...         Like check, but warnings also fail\n"
               "      --strict                         Treat warnings as errors\n"
               "      --json                           Print diagnostics as JSON on stdout\n"
               "      --std <dir>                      Standard library directory\n"
               "  fmt [--check] <path>...              Format files, or directories recursively;\n"
               "                                       `-` formats stdin to stdout\n"
               "  lsp --stdio [--std <dir>]            Run the language server\n"
               "  spec-test [--std <dir>] <dir>        Run the executable specification in <dir>\n"
               "  inspect --tokens <file>              Show the tokens of a file\n"
               "  inspect --ast <file>                 Show the syntax tree of a file\n"
               "  inspect --core-ir <file>             Show the Core IR of a file\n"
               "  inspect --parameters [--json] <file> Show the parameter manifest of each\n"
               "                                       block declared in a file\n"
               "\n"
               "Options:\n"
               "      --no-color   Disable colored diagnostics\n"
               "  -h, --help       Show this help\n"
               "  -V, --version    Show the toolchain version\n",
               out);
}

int usage_error(const std::string& message) {
    std::fprintf(stderr, "linnet: %s\n\n", message.c_str());
    print_usage(stderr);
    return exit_usage;
}

struct Options {
    bool color = stderr_is_terminal();
    bool json = false;   // diagnostics as JSON on stdout
    bool strict = false; // warnings fail the command
};

int report(const SourceManager& sources, DiagnosticSink& sink, const Options& options) {
    sink.sort_by_location();
    bool has_warnings = false;
    for (const Diagnostic& diagnostic : sink.diagnostics()) {
        has_warnings = has_warnings || diagnostic.severity == Severity::Warning;
    }
    const bool has_failed = sink.has_errors() || (options.strict && has_warnings);
    if (options.json) {
        std::fputs(render_json(sources, sink.diagnostics()).c_str(), stdout);
        return has_failed ? exit_failure : exit_success;
    }
    bool is_first = true;
    for (const Diagnostic& diagnostic : sink.diagnostics()) {
        if (!is_first) {
            std::fputc('\n', stderr);
        }
        is_first = false;
        const std::string text = render_diagnostic(sources, diagnostic, {.color = options.color});
        std::fputs(text.c_str(), stderr);
    }
    return has_failed ? exit_failure : exit_success;
}

void collect_sources(const std::filesystem::path& root, std::vector<std::filesystem::path>& out);

// The standard library directory: an explicit option wins, then the
// LINNET_STD environment variable, then locations relative to the executable
// (an installed `share/linnet/stdlib`, or `stdlib` in a source checkout).
std::filesystem::path find_std_root(const std::string& option, const char* program) {
    std::error_code error;
    if (!option.empty()) {
        return option;
    }
    if (const char* from_environment = std::getenv("LINNET_STD")) {
        return from_environment;
    }
    std::filesystem::path directory =
        std::filesystem::weakly_canonical(std::filesystem::path(program), error).parent_path();
    for (int depth = 0; depth < 4 && !directory.empty(); ++depth) {
        for (const char* candidate : {"share/linnet/stdlib", "stdlib"}) {
            if (std::filesystem::is_directory(directory / candidate, error)) {
                return directory / candidate;
            }
        }
        directory = directory.parent_path();
    }
    return {};
}

int run_check(std::span<const std::string_view> args, Options options, const char* program) {
    std::vector<std::filesystem::path> paths;
    std::string std_option;
    bool has_path = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--std") {
            if (i + 1 == args.size()) {
                return usage_error("--std requires a directory");
            }
            std_option = args[++i];
        } else if (arg == "--json") {
            options.json = true;
        } else if (arg == "--strict") {
            options.strict = true;
        } else if (arg.starts_with("-")) {
            return usage_error("unknown check option '" + std::string(arg) + "'");
        } else {
            has_path = true;
            collect_sources(arg, paths);
        }
    }
    if (!has_path) {
        return usage_error("check requires at least one path");
    }
    std::sort(paths.begin(), paths.end());

    SourceManager sources;
    DiagnosticSink sink;
    const LoaderOptions loader_options{find_std_root(std_option, program), {}};
    const Program program_modules = load_program(sources, paths, loader_options, sink);
    // Semantic analysis assumes well-formed trees.
    if (!sink.has_errors()) {
        const std::vector<const ast::Ast*> modules = program_modules.module_pointers();
        sema::analyze(sources, modules, sink, &program_modules.imports);
    }
    return report(sources, sink, options);
}

// Creates a package skeleton without touching files that already exist.
bool write_file(const std::filesystem::path& path, const std::string& contents);

int run_init(std::span<const std::string_view> args) {
    if (args.size() > 1 || (!args.empty() && args.front().starts_with("-"))) {
        return usage_error("init takes at most one directory");
    }
    std::error_code error;
    const std::filesystem::path directory =
        args.empty() ? std::filesystem::current_path(error) : std::filesystem::path(args.front());
    std::string name = std::filesystem::weakly_canonical(directory, error).filename().string();
    for (char& c : name) {
        if (c == '-' || c == ' ' || c == '.') {
            c = '_';
        }
    }
    if (!is_valid_package_name(name)) {
        name = "model";
    }

    const std::filesystem::path manifest = directory / "linnet.toml";
    const std::filesystem::path library = directory / "src" / "lib.linnet";
    if (std::filesystem::exists(manifest, error)) {
        std::fprintf(stderr, "linnet: %s already exists\n", manifest.generic_string().c_str());
        return exit_failure;
    }
    std::filesystem::create_directories(directory / "src", error);
    if (error) {
        std::fprintf(
            stderr, "linnet: cannot create %s\n", (directory / "src").generic_string().c_str());
        return exit_failure;
    }
    if (!write_file(manifest, default_manifest(name))) {
        std::fprintf(stderr, "linnet: cannot write %s\n", manifest.generic_string().c_str());
        return exit_failure;
    }
    if (!std::filesystem::exists(library, error) &&
        !write_file(library,
                    "module " + name + "\n\npub fn identity(x: f32) -> f32 {\n    return x\n}\n")) {
        std::fprintf(stderr, "linnet: cannot write %s\n", library.generic_string().c_str());
        return exit_failure;
    }
    std::printf("created package `%s` in %s\n", name.c_str(), directory.generic_string().c_str());
    return exit_success;
}

int run_plan(std::span<const std::string_view> args, const Options& options, const char* program) {
    std::string std_option;
    std::string root;
    std::string_view path;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--std" || arg == "--root") {
            if (i + 1 == args.size()) {
                return usage_error(std::string(arg) + " requires a value");
            }
            (arg == "--std" ? std_option : root) = args[++i];
        } else if (arg.starts_with("-")) {
            return usage_error("unknown plan option '" + std::string(arg) + "'");
        } else if (!path.empty()) {
            return usage_error("plan takes exactly one file");
        } else {
            path = arg;
        }
    }
    if (path.empty()) {
        return usage_error("plan requires a file");
    }
    SourceManager sources;
    DiagnosticSink sink;
    const std::vector<std::filesystem::path> paths{std::filesystem::path(path)};
    const LoaderOptions loader_options{find_std_root(std_option, program), {}};
    const Program program_modules = load_program(sources, paths, loader_options, sink);
    if (sink.has_errors()) {
        return report(sources, sink, options);
    }
    const std::vector<const ast::Ast*> modules = program_modules.module_pointers();
    const sema::AnalysisResult analysis =
        sema::analyze(sources, modules, sink, &program_modules.imports);
    if (sink.has_errors()) {
        return report(sources, sink, options);
    }
    ir::Module core = ir::lower(sources, modules, analysis.model);
    const std::vector<std::string> problems = ir::verify(core);
    for (const std::string& problem : problems) {
        std::fprintf(stderr, "linnet: IR verifier: %s\n", problem.c_str());
    }
    if (!problems.empty()) {
        return exit_failure;
    }
    const auto plan = backend::export_plan(core, backend::PlanOptions{root, 0, modules});
    if (!plan) {
        std::fprintf(stderr, "linnet: %s\n", plan.error().c_str());
        return exit_failure;
    }
    std::fputs(plan->c_str(), stdout);
    return report(sources, sink, options);
}

// Runs every case of a spec-tests directory through the frontend and compares
// the diagnostic codes with the manifest. Diagnostics snapshots under
// diagnostics/ must match the rendered text exactly.
int run_spec_test(std::span<const std::string_view> args,
                  const Options& options,
                  const char* program) {
    std::string std_option;
    std::filesystem::path root;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--std") {
            if (i + 1 == args.size()) {
                return usage_error("--std requires a directory");
            }
            std_option = args[++i];
        } else if (arg.starts_with("-")) {
            return usage_error("unknown spec-test option '" + std::string(arg) + "'");
        } else if (!root.empty()) {
            return usage_error("spec-test takes exactly one directory");
        } else {
            root = arg;
        }
    }
    if (root.empty()) {
        return usage_error("spec-test requires the spec-tests directory");
    }
    const auto cases = read_spec_manifest(root / "manifest.toml");
    if (!cases) {
        std::fprintf(stderr, "linnet: %s\n", cases.error().c_str());
        return exit_failure;
    }

    const LoaderOptions loader_options{find_std_root(std_option, program), {}};
    std::size_t failures = 0;
    for (const SpecCase& spec_case : *cases) {
        SourceManager sources;
        DiagnosticSink sink;
        const std::vector<std::filesystem::path> files{root / spec_case.file};
        const Program program_modules = load_program(sources, files, loader_options, sink);
        if (!sink.has_errors()) {
            const std::vector<const ast::Ast*> modules = program_modules.module_pointers();
            sema::analyze(sources, modules, sink, &program_modules.imports);
        }
        sink.sort_by_location();

        std::string rendered;
        std::string problem;
        for (const Diagnostic& diagnostic : sink.diagnostics()) {
            if (diagnostic.severity != Severity::Error) {
                continue;
            }
            rendered += rendered.empty() ? "" : "\n";
            rendered += render_diagnostic(sources, diagnostic);
            if (!spec_case.expects_error) {
                problem = "expected no errors";
            } else if (diagnostic.code != spec_case.code) {
                problem = "expected only " + spec_case.code;
            }
        }
        if (spec_case.expects_error && !sink.has_errors()) {
            problem = "expected " + spec_case.code + ", got no errors";
        }

        const std::filesystem::path snapshot =
            root / "diagnostics" / std::filesystem::path(spec_case.file).stem().concat(".stderr");
        std::error_code error;
        if (problem.empty() && std::filesystem::is_regular_file(snapshot, error)) {
            std::ifstream stream(snapshot, std::ios::binary);
            const std::string expected{std::istreambuf_iterator<char>(stream),
                                       std::istreambuf_iterator<char>()};
            // Snapshots name files relative to the spec-tests directory.
            std::string relative = rendered;
            const std::string prefix = (root / "").generic_string();
            for (std::size_t at = relative.find(prefix); at != std::string::npos;
                 at = relative.find(prefix, at)) {
                relative.erase(at, prefix.size());
            }
            if (relative != expected) {
                problem = "diagnostics differ from " + snapshot.generic_string();
                rendered = relative;
            }
        }

        if (!problem.empty()) {
            ++failures;
            std::fprintf(stderr, "FAIL %s: %s\n", spec_case.file.c_str(), problem.c_str());
            if (!rendered.empty()) {
                std::fputs(rendered.c_str(), stderr);
                std::fputc('\n', stderr);
            }
        } else if (options.color) {
            std::printf("ok   %s\n", spec_case.file.c_str());
        }
    }
    std::printf("%zu cases, %zu failed\n", cases->size(), failures);
    return failures == 0 ? exit_success : exit_failure;
}

std::string render_manifests(std::span<const sema::ManifestBlock> blocks, bool as_json) {
    std::string out;
    if (as_json) {
        out = "{\"version\":1,\"blocks\":[";
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            const sema::ManifestBlock& block = blocks[b];
            out += b == 0 ? "" : ",";
            out += "{\"name\":" + json_string(block.name) +
                   ",\"module\":" + json_string(block.module) + ",\"generics\":[";
            for (std::size_t i = 0; i < block.generics.size(); ++i) {
                out += (i == 0 ? "" : ",") + json_string(block.generics[i]);
            }
            out += "],\"entries\":[";
            for (std::size_t e = 0; e < block.entries.size(); ++e) {
                const sema::ManifestEntry& entry = block.entries[e];
                out += e == 0 ? "" : ",";
                out += "{\"path\":" + json_string(entry.path) +
                       ",\"kind\":" + json_string(entry.kind) +
                       ",\"dtype\":" + json_string(entry.dtype) + ",\"shape\":[";
                for (std::size_t i = 0; i < entry.shape.size(); ++i) {
                    out += (i == 0 ? "" : ",") + json_string(entry.shape[i]);
                }
                out += "],\"repeat\":[";
                for (std::size_t i = 0; i < entry.repeat.size(); ++i) {
                    out += (i == 0 ? "" : ",") + json_string(entry.repeat[i]);
                }
                out +=
                    std::string("],\"optional\":") + (entry.is_optional ? "true" : "false") + "}";
            }
            out += "]}";
        }
        return out + "]}\n";
    }
    for (const sema::ManifestBlock& block : blocks) {
        out += block.module + "::" + block.name;
        if (!block.generics.empty()) {
            out += "<";
            for (std::size_t i = 0; i < block.generics.size(); ++i) {
                out += (i == 0 ? "" : ", ") + block.generics[i];
            }
            out += ">";
        }
        out += "\n";
        for (const sema::ManifestEntry& entry : block.entries) {
            out += "  " + entry.kind + " " + entry.path + ": Tensor[";
            for (std::size_t i = 0; i < entry.shape.size(); ++i) {
                out += (i == 0 ? "" : ", ") + entry.shape[i];
            }
            out += "; " + entry.dtype + "]" + (entry.is_optional ? "?" : "");
            for (const std::string& count : entry.repeat) {
                out += " x " + count;
            }
            out += "\n";
        }
    }
    return out;
}

int run_inspect(std::span<const std::string_view> args, Options options, const char* program) {
    std::string_view view;
    std::string_view path;
    std::string std_option;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--json") {
            options.json = true;
        } else if (arg == "--std") {
            if (i + 1 == args.size()) {
                return usage_error("--std requires a directory");
            }
            std_option = args[++i];
        } else if (arg == "--tokens" || arg == "--ast" || arg == "--parameters" ||
                   arg == "--core-ir") {
            if (!view.empty()) {
                return usage_error("inspect takes exactly one view option");
            }
            view = arg;
        } else if (arg.starts_with("-")) {
            return usage_error("unknown inspect option '" + std::string(arg) + "'");
        } else if (!path.empty()) {
            return usage_error("inspect takes exactly one file");
        } else {
            path = arg;
        }
    }
    if (view.empty() || path.empty()) {
        return usage_error("inspect requires a view option and a file");
    }
    if (options.json && view != "--parameters") {
        return usage_error("--json is only available with --parameters");
    }

    SourceManager sources;
    if (view == "--parameters" || view == "--core-ir") {
        DiagnosticSink sink;
        const std::vector<std::filesystem::path> paths{std::filesystem::path(path)};
        const LoaderOptions loader_options{find_std_root(std_option, program), {}};
        const Program program_modules = load_program(sources, paths, loader_options, sink);
        if (sink.has_errors()) {
            return report(sources, sink, options);
        }
        const std::vector<const ast::Ast*> modules = program_modules.module_pointers();
        const sema::AnalysisResult analysis =
            sema::analyze(sources, modules, sink, &program_modules.imports);
        if (sink.has_errors()) {
            return report(sources, sink, options);
        }
        if (view == "--core-ir") {
            const ir::Module core = ir::lower(sources, modules, analysis.model);
            for (const std::string& problem : ir::verify(core)) {
                std::fprintf(stderr, "linnet: IR verifier: %s\n", problem.c_str());
            }
            std::fputs(ir::print(core).c_str(), stdout);
            return ir::verify(core).empty() ? exit_success : exit_failure;
        }
        // Only blocks of the requested file, not of what it imports.
        std::vector<sema::ManifestBlock> blocks;
        std::string wanted;
        for (const ast::Name& segment : modules.front()->module_path) {
            wanted += wanted.empty() ? "" : ".";
            wanted += segment.text;
        }
        for (const sema::ManifestBlock& block : analysis.manifests) {
            if (block.module == wanted) {
                blocks.push_back(block);
            }
        }
        std::fputs(render_manifests(blocks, options.json).c_str(), stdout);
        options.json = false; // diagnostics stay on stderr as text
        return report(sources, sink, options);
    }

    const auto file = sources.load_file(path);
    if (!file) {
        std::fprintf(stderr, "linnet: %s\n", file.error().c_str());
        return exit_failure;
    }

    DiagnosticSink sink;
    std::string output;
    if (view == "--tokens") {
        for (const Token& token : lex(sources, *file, sink).tokens) {
            const LineColumn position = sources.line_column(*file, token.span.begin);
            output += std::to_string(position.line) + ":" + std::to_string(position.column) + " ";
            if (token.kind == TokenKind::Eof || is_keyword(token.kind)) {
                output += token_kind_name(token.kind);
            } else {
                output += std::string(token_kind_name(token.kind)) == sources.text(token.span)
                              ? std::string(sources.text(token.span))
                              : std::string(token_kind_name(token.kind)) + " " +
                                    std::string(sources.text(token.span));
            }
            output += '\n';
        }
    } else {
        output = ast::dump(parse(sources, *file, sink), sources);
    }
    std::fputs(output.c_str(), stdout);
    return report(sources, sink, options);
}

// Formats one registered file. Returns false after reporting when it has
// syntax errors; a file that does not parse is never rewritten.
bool format_file(const SourceManager& sources,
                 FileId file,
                 const Options& options,
                 std::string& formatted) {
    DiagnosticSink sink;
    const ast::Ast tree = parse(sources, file, sink);
    if (sink.has_errors()) {
        report(sources, sink, options);
        return false;
    }
    formatted = format::format(tree, sources);
    return true;
}

bool write_file(const std::filesystem::path& path, const std::string& contents) {
    // Write beside the target and rename over it so that an interrupted run
    // cannot leave a truncated source file.
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        stream << contents;
        if (!stream.flush()) {
            return false;
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

void collect_sources(const std::filesystem::path& root, std::vector<std::filesystem::path>& out) {
    std::error_code error;
    if (!std::filesystem::is_directory(root, error)) {
        out.push_back(root);
        return;
    }
    std::filesystem::recursive_directory_iterator walker(root, error);
    const std::filesystem::recursive_directory_iterator end;
    while (!error && walker != end) {
        const std::filesystem::path& path = walker->path();
        const std::string name = path.filename().string();
        if (walker->is_directory(error) && (name.starts_with(".") || name == "build")) {
            walker.disable_recursion_pending();
        } else if (walker->is_regular_file(error) && path.extension() == ".linnet") {
            out.push_back(path);
        }
        walker.increment(error);
    }
}

int run_fmt(std::span<const std::string_view> args, const Options& options) {
    bool is_check = false;
    bool use_stdin = false;
    bool has_path = false;
    std::vector<std::filesystem::path> paths;
    for (const std::string_view arg : args) {
        if (arg == "--check") {
            is_check = true;
        } else if (arg == "-") {
            use_stdin = true;
        } else if (arg.starts_with("-")) {
            return usage_error("unknown fmt option '" + std::string(arg) + "'");
        } else {
            has_path = true;
            collect_sources(arg, paths);
        }
    }
    if (use_stdin && has_path) {
        return usage_error("fmt reads either stdin or paths, not both");
    }
    if (!use_stdin && !has_path) {
        return usage_error("fmt requires at least one path, or `-` for stdin");
    }

    SourceManager sources;
    if (use_stdin) {
        std::string input{std::istreambuf_iterator<char>(std::cin),
                          std::istreambuf_iterator<char>()};
        const std::string original = input;
        const auto file = sources.add_file("<stdin>", std::move(input));
        std::string formatted;
        if (!file || !format_file(sources, *file, options, formatted)) {
            return exit_failure;
        }
        if (!is_check) {
            std::fputs(formatted.c_str(), stdout);
        }
        return is_check && formatted != original ? exit_failure : exit_success;
    }

    std::sort(paths.begin(), paths.end());
    int status = exit_success;
    for (const std::filesystem::path& path : paths) {
        const auto file = sources.load_file(path);
        if (!file) {
            std::fprintf(stderr, "linnet: %s\n", file.error().c_str());
            status = exit_failure;
            continue;
        }
        std::string formatted;
        if (!format_file(sources, *file, options, formatted)) {
            status = exit_failure;
        } else if (formatted != sources.contents(*file)) {
            if (is_check) {
                std::fprintf(stderr, "would reformat %s\n", path.generic_string().c_str());
                status = exit_failure;
            } else if (!write_file(path, formatted)) {
                std::fprintf(
                    stderr, "linnet: %s: cannot write file\n", path.generic_string().c_str());
                status = exit_failure;
            }
        }
    }
    return status;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string_view> args;
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-color") {
            options.color = false;
        } else {
            args.push_back(arg);
        }
    }
    if (args.empty()) {
        print_usage(stderr);
        return exit_usage;
    }

    const std::string_view command = args.front();
    const std::span<const std::string_view> rest(args.begin() + 1, args.end());
    if (command == "-h" || command == "--help" || command == "help") {
        print_usage(stdout);
        return exit_success;
    }
    if (command == "-V" || command == "--version") {
        std::printf("linnet %s\n", version_string);
        return exit_success;
    }
    if (command == "check") {
        return run_check(rest, options, argv[0]);
    }
    if (command == "init") {
        return run_init(rest);
    }
    if (command == "lint") {
        options.strict = true;
        return run_check(rest, options, argv[0]);
    }
    if (command == "lsp") {
        std::string std_option;
        bool is_stdio = false;
        for (std::size_t i = 0; i < rest.size(); ++i) {
            if (rest[i] == "--stdio") {
                is_stdio = true;
            } else if (rest[i] == "--std" && i + 1 < rest.size()) {
                std_option = rest[++i];
            } else {
                return usage_error("unknown lsp option '" + std::string(rest[i]) + "'");
            }
        }
        if (!is_stdio) {
            return usage_error("lsp requires --stdio");
        }
        lsp::Server server(lsp::ServerOptions{find_std_root(std_option, argv[0])});
        return server.run(std::cin, std::cout);
    }
    if (command == "plan") {
        return run_plan(rest, options, argv[0]);
    }
    if (command == "spec-test") {
        return run_spec_test(rest, options, argv[0]);
    }
    if (command == "fmt") {
        return run_fmt(rest, options);
    }
    if (command == "inspect") {
        return run_inspect(rest, options, argv[0]);
    }
    return usage_error("unknown command '" + std::string(command) + "'");
}
