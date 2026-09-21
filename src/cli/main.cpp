#include "linnet/ast/dump.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/diagnostic/render.hpp"
#include "linnet/source/source_manager.hpp"
#include "linnet/syntax/lexer.hpp"
#include "linnet/syntax/parser.hpp"
#include "linnet/version.hpp"

#include <cstdio>
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
    if (command == "inspect") {
        return run_inspect(rest, options);
    }
    return usage_error("unknown command '" + std::string(command) + "'");
}
