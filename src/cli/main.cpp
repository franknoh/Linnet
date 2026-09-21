#include "linnet/ast/dump.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/diagnostic/render.hpp"
#include "linnet/format/formatter.hpp"
#include "linnet/source/source_manager.hpp"
#include "linnet/syntax/lexer.hpp"
#include "linnet/syntax/parser.hpp"
#include "linnet/version.hpp"

#include <algorithm>
#include <cstdio>
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
               "  fmt [--check] <path>...              Format files, or directories recursively;\n"
               "                                       `-` formats stdin to stdout\n"
               "  inspect (--tokens | --ast) <file>    Show compiler-internal views of a file\n"
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
};

int report(const SourceManager& sources, DiagnosticSink& sink, const Options& options) {
    sink.sort_by_location();
    bool is_first = true;
    for (const Diagnostic& diagnostic : sink.diagnostics()) {
        if (!is_first) {
            std::fputc('\n', stderr);
        }
        is_first = false;
        const std::string text = render_diagnostic(sources, diagnostic, {.color = options.color});
        std::fputs(text.c_str(), stderr);
    }
    return sink.has_errors() ? exit_failure : exit_success;
}

int run_inspect(std::span<const std::string_view> args, const Options& options) {
    std::string_view view;
    std::string_view path;
    for (const std::string_view arg : args) {
        if (arg == "--tokens" || arg == "--ast") {
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

    SourceManager sources;
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
    if (command == "fmt") {
        return run_fmt(rest, options);
    }
    if (command == "inspect") {
        return run_inspect(rest, options);
    }
    return usage_error("unknown command '" + std::string(command) + "'");
}
