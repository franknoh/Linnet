#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/source/source_manager.hpp"

#include <deque>
#include <filesystem>
#include <span>
#include <vector>

namespace linnet {

struct LoaderOptions {
    // Directory of the toolchain standard library: `std.nn.linear` is the file
    // `<std_root>/nn/linear.linnet`. Empty when no standard library is known.
    std::filesystem::path std_root;
};

// A set of parsed modules closed under their imports.
struct Program {
    std::deque<ast::Ast> modules; // deque: addresses stay valid while loading
    sema::ImportTable imports;

    std::vector<const ast::Ast*> module_pointers() const;
};

// Parses `files` and, transitively, every module they import. Imports are
// logical paths, never file names, and map to files by fixed rules:
//
//   std.a.b     <std_root>/a/b.linnet
//   crate       <package>/src/lib.linnet
//   crate.a.b   <package>/src/a/b.linnet
//
// where <package> is the nearest directory above the importing file that
// contains `linnet.toml`. Loading only reads and parses files; nothing from a
// package is ever executed. Imports that cannot be mapped are left out of the
// import table for semantic analysis to report.
Program load_program(SourceManager& sources,
                     std::span<const std::filesystem::path> files,
                     const LoaderOptions& options,
                     DiagnosticSink& sink);

// Nearest ancestor of `file` that contains `linnet.toml`, or an empty path.
std::filesystem::path find_package_root(const std::filesystem::path& file);

} // namespace linnet
