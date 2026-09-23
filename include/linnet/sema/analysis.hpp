#pragma once

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/source/source_manager.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace linnet::sema {

// The checked type of one local binding, rendered for tools and tests.
struct BindingInfo {
    SourceSpan span;   // the bound name
    std::string owner; // enclosing function, qualified by its block if any
    std::string name;
    std::string type;
    bool is_inferred = true; // false when the source spells the type
};

// One externally bound tensor of a block, or a nested block. Paths use `.`
// between levels and `[*]` for every element of a sub-block array; `repeat`
// lists the array lengths along the path, outermost first.
struct ManifestEntry {
    std::string path;
    std::string kind; // "param", "buffer", or "state"
    std::string dtype;
    std::vector<std::string> shape; // one dimension expression per axis
    std::vector<std::string> repeat;
    bool is_optional = false;
};

// The parameter manifest of a block declared at module level. Generic
// parameters of the block itself stay symbolic in shapes and repeats.
struct ManifestBlock {
    std::string name;
    std::string module;
    std::vector<std::string> generics;
    std::vector<ManifestEntry> entries;
};

enum class SymbolKind : std::uint8_t {
    Module,
    Const,
    TypeAlias,
    Struct,
    Enum,
    Function,
    Op,
    Entry,
    Block,
    Param,
    Buffer,
    State,
    Sub,
    GenericDim,
    GenericPack,
    GenericDType,
    Parameter,
    Local,
};

// A declared name and its rendered description, for editors.
struct SymbolInfo {
    SymbolKind kind = SymbolKind::Local;
    std::string name;
    SourceSpan span;    // the declared name
    std::string detail; // one-line description: `op linear<...>(...) -> ...`, `let x: T`
};

// One use of a declared name: the source span of the use and the symbol.
struct Reference {
    SourceSpan span;
    std::uint32_t symbol; // index into AnalysisResult::symbols
};

struct Model;

struct AnalysisResult {
    std::vector<BindingInfo> bindings;
    std::vector<ManifestBlock> manifests;
    std::vector<SymbolInfo> symbols;
    std::vector<Reference> references;
    // The typed model (see sema/model.hpp); complete only without errors.
    std::shared_ptr<Model> model;
};

// Where each `use` declaration leads, as decided by the module loader: the key
// is (importing module, index of the `use`), the value the imported module.
using ImportTable = std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>;

// Resolves names and checks types, shapes, and tensor algebra for a set of
// modules that were parsed without errors. Imports follow `imports` when it is
// given; otherwise they are resolved among the given modules by their declared
// module paths. All findings go to `sink`; analysis itself never fails.
AnalysisResult analyze(const SourceManager& sources,
                       std::span<const ast::Ast* const> modules,
                       DiagnosticSink& sink,
                       const ImportTable* imports = nullptr);

} // namespace linnet::sema
