#include "linnet/lsp/server.hpp"

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/format/formatter.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/source/source_manager.hpp"
#include "linnet/syntax/lexer.hpp"
#include "linnet/syntax/parser.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <optional>
#include <set>
#include <variant>

namespace linnet::lsp {

namespace {

template <typename... Visitors>
struct Overloaded : Visitors... {
    using Visitors::operator()...;
};
template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

constexpr int error_method_not_found = -32601;
constexpr int error_invalid_params = -32602;

constexpr auto keywords = std::to_array<std::string_view>({
    "module", "use",   "pub",    "as",     "const",  "type",  "struct",  "enum",    "fn",
    "op",     "block", "entry",  "param",  "buffer", "sub",   "let",     "var",     "return",
    "if",     "else",  "match",  "static", "for",    "in",    "where",   "true",    "false",
    "none",   "some",  "crate",  "Tensor", "Dim",    "Shape", "DType",   "Numeric", "Integer",
    "Float",  "bool",  "i8",     "i16",    "i32",    "i64",   "u8",      "u16",     "u32",
    "u64",    "f16",   "bf16",   "f32",    "f64",    "cast",  "reshape", "permute", "broadcast_to",
    "concat", "iota",  "fill",   "exp",    "log",    "sqrt",  "rsqrt",   "sin",     "cos",
    "tanh",   "abs",   "select", "min",    "max",    "sum",   "prod",    "any",     "all",
});

// Semantic token legend, in the order announced to the client.
constexpr auto token_types = std::to_array<std::string_view>({
    "namespace",     // 0 module
    "variable",      // 1 const, local
    "type",          // 2 alias, generic dtype
    "struct",        // 3
    "enum",          // 4
    "function",      // 5 fn, entry
    "macro",         // 6 op (semantic operation)
    "class",         // 7 block
    "property",      // 8 param, buffer, sub
    "typeParameter", // 9 generic dim/pack
    "parameter",     // 10
});
constexpr auto token_modifiers = std::to_array<std::string_view>({"declaration", "readonly"});

std::uint32_t token_type_of(sema::SymbolKind kind) {
    switch (kind) {
    case sema::SymbolKind::Module:
        return 0;
    case sema::SymbolKind::Const:
    case sema::SymbolKind::Local:
        return 1;
    case sema::SymbolKind::TypeAlias:
    case sema::SymbolKind::GenericDType:
        return 2;
    case sema::SymbolKind::Struct:
        return 3;
    case sema::SymbolKind::Enum:
        return 4;
    case sema::SymbolKind::Function:
    case sema::SymbolKind::Entry:
        return 5;
    case sema::SymbolKind::Op:
        return 6;
    case sema::SymbolKind::Block:
        return 7;
    case sema::SymbolKind::Param:
    case sema::SymbolKind::Buffer:
    case sema::SymbolKind::Sub:
        return 8;
    case sema::SymbolKind::GenericDim:
    case sema::SymbolKind::GenericPack:
        return 9;
    case sema::SymbolKind::Parameter:
        return 10;
    }
    return 1;
}

// LSP SymbolKind numbers.
int lsp_symbol_kind(sema::SymbolKind kind) {
    switch (kind) {
    case sema::SymbolKind::Module:
        return 2;
    case sema::SymbolKind::Const:
        return 14;
    case sema::SymbolKind::TypeAlias:
        return 26;
    case sema::SymbolKind::Struct:
        return 23;
    case sema::SymbolKind::Enum:
        return 10;
    case sema::SymbolKind::Function:
    case sema::SymbolKind::Entry:
    case sema::SymbolKind::Op:
        return 12;
    case sema::SymbolKind::Block:
        return 5;
    case sema::SymbolKind::Param:
    case sema::SymbolKind::Buffer:
    case sema::SymbolKind::Sub:
        return 8;
    case sema::SymbolKind::GenericDim:
    case sema::SymbolKind::GenericPack:
    case sema::SymbolKind::GenericDType:
        return 26;
    case sema::SymbolKind::Parameter:
    case sema::SymbolKind::Local:
        return 13;
    }
    return 13;
}

bool contains(SourceSpan span, FileId file, std::uint32_t offset) {
    return span.file == file && span.begin <= offset && offset < std::max(span.end, span.begin + 1);
}

} // namespace

// ------------------------------------------------------------------------ URIs

std::string uri_from_path(const std::filesystem::path& path) {
    const std::string generic = path.generic_string();
    std::string uri = "file://";
    if (!generic.starts_with('/')) {
        uri += '/'; // Windows drive paths
    }
    for (const char raw : generic) {
        const auto c = static_cast<unsigned char>(raw);
        const bool is_safe = std::isalnum(c) != 0 || c == '/' || c == '-' || c == '_' || c == '.' ||
                             c == '~' || c == ':';
        if (is_safe) {
            uri += static_cast<char>(c);
        } else {
            std::array<char, 4> escaped{};
            std::snprintf(escaped.data(), escaped.size(), "%%%02X", c);
            uri += escaped.data();
        }
    }
    return uri;
}

std::filesystem::path path_from_uri(std::string_view uri) {
    if (!uri.starts_with("file://")) {
        return {};
    }
    uri.remove_prefix(7);
    if (uri.starts_with("localhost")) {
        uri.remove_prefix(9);
    }
    std::string path;
    for (std::size_t i = 0; i < uri.size(); ++i) {
        if (uri[i] == '%' && i + 2 < uri.size()) {
            path += static_cast<char>(std::stoi(std::string(uri.substr(i + 1, 2)), nullptr, 16));
            i += 2;
        } else {
            path += uri[i];
        }
    }
    // `/c:/dir` is a Windows path.
    if (path.size() > 2 && path[0] == '/' && path[2] == ':') {
        path.erase(0, 1);
    }
    return std::filesystem::path(path);
}

// ----------------------------------------------------------------------- state

struct Server::Document {
    std::string text;
    std::int64_t version = 0;
    // Analysis of this document as the root module; rebuilt on every change.
    std::unique_ptr<SourceManager> sources;
    Program program;
    sema::AnalysisResult analysis;
    std::vector<Diagnostic> diagnostics;
    FileId file = invalid_file_id;
    bool has_syntax_errors = false;
};

struct Server::State {
    ServerOptions options;
    std::map<std::string, Document> documents; // by canonical path
    bool has_shut_down = false;

    static std::string key_of(const std::filesystem::path& path) {
        std::error_code error;
        const std::filesystem::path canonical = std::filesystem::weakly_canonical(path, error);
        return (error ? path : canonical).generic_string();
    }

    void analyze(const std::string& key, Document& document) {
        document.sources = std::make_unique<SourceManager>();
        DiagnosticSink sink;
        LoaderOptions loader{options.std_root, {}};
        for (const auto& [other_key, other] : documents) {
            loader.overlays[other_key] = other.text;
        }
        const std::vector<std::filesystem::path> files{std::filesystem::path(key)};
        document.program = load_program(*document.sources, files, loader, sink);
        document.file = document.program.modules.empty() ? invalid_file_id
                                                         : document.program.modules.front().file;
        document.has_syntax_errors = sink.has_errors();
        document.analysis = {};
        if (!sink.has_errors()) {
            const std::vector<const ast::Ast*> modules = document.program.module_pointers();
            document.analysis =
                sema::analyze(*document.sources, modules, sink, &document.program.imports);
        }
        sink.sort_by_location();
        document.diagnostics = sink.diagnostics();
    }

    // -------------------------------------------------------------- positions

    static Json position(const SourceManager& sources, FileId file, std::uint32_t offset) {
        const Utf16Position at = sources.utf16_position(file, offset);
        return Json::object({{"line", at.line}, {"character", at.character}});
    }

    static Json range(const SourceManager& sources, SourceSpan span) {
        return Json::object({{"start", position(sources, span.file, span.begin)},
                             {"end", position(sources, span.file, span.end)}});
    }

    static Json location(const SourceManager& sources, SourceSpan span) {
        return Json::object({{"uri", uri_from_path(std::string(sources.path(span.file)))},
                             {"range", range(sources, span)}});
    }

    static std::uint32_t
    offset_of(const SourceManager& sources, FileId file, const Json& position) {
        return sources.offset_from_utf16(
            file,
            {static_cast<std::uint32_t>(position["line"].as_int()),
             static_cast<std::uint32_t>(position["character"].as_int())});
    }

    // ------------------------------------------------------------ diagnostics

    Json diagnostics_of(const Document& document) const {
        Json::Array items;
        for (const Diagnostic& diagnostic : document.diagnostics) {
            if (diagnostic.primary.span.file != document.file) {
                continue;
            }
            std::string message = diagnostic.message;
            if (!diagnostic.primary.message.empty()) {
                message += ": " + diagnostic.primary.message;
            }
            for (const std::string& note : diagnostic.notes) {
                message += "\nnote: " + note;
            }
            for (const std::string& help : diagnostic.help) {
                message += "\nhelp: " + help;
            }
            Json::Array related;
            for (const Label& label : diagnostic.secondary) {
                related.push_back(
                    Json::object({{"location", location(*document.sources, label.span)},
                                  {"message", label.message}}));
            }
            items.push_back(Json::object({
                {"range", range(*document.sources, diagnostic.primary.span)},
                {"severity",
                 diagnostic.severity == Severity::Error     ? 1
                 : diagnostic.severity == Severity::Warning ? 2
                                                            : 3},
                {"code", diagnostic.code},
                {"source", "linnet"},
                {"message", message},
                {"relatedInformation", std::move(related)},
            }));
        }
        return items;
    }

    // ----------------------------------------------------------------- lookup

    const sema::SymbolInfo* symbol_at(const Document& document, std::uint32_t offset) const {
        for (const sema::Reference& reference : document.analysis.references) {
            if (contains(reference.span, document.file, offset)) {
                return &document.analysis.symbols[reference.symbol];
            }
        }
        for (const sema::SymbolInfo& symbol : document.analysis.symbols) {
            if (contains(symbol.span, document.file, offset)) {
                return &symbol;
            }
        }
        return nullptr;
    }

    Json hover(const Document& document, std::uint32_t offset) const {
        const sema::SymbolInfo* symbol = symbol_at(document, offset);
        if (symbol == nullptr) {
            return nullptr;
        }
        return Json::object(
            {{"contents",
              Json::object(
                  {{"kind", "markdown"}, {"value", "```linnet\n" + symbol->detail + "\n```"}})}});
    }

    Json definition(const Document& document, std::uint32_t offset) const {
        const sema::SymbolInfo* symbol = symbol_at(document, offset);
        if (symbol == nullptr || symbol->span.file == invalid_file_id) {
            return nullptr;
        }
        return location(*document.sources, symbol->span);
    }

    Json references(const Document& document, std::uint32_t offset, bool with_declaration) const {
        const sema::SymbolInfo* symbol = symbol_at(document, offset);
        if (symbol == nullptr) {
            return nullptr;
        }
        const auto index = static_cast<std::uint32_t>(symbol - document.analysis.symbols.data());
        Json::Array items;
        if (with_declaration) {
            items.push_back(location(*document.sources, symbol->span));
        }
        for (const sema::Reference& reference : document.analysis.references) {
            if (reference.symbol == index) {
                items.push_back(location(*document.sources, reference.span));
            }
        }
        return items;
    }

    Json rename(const Document& document, std::uint32_t offset, const std::string& new_name) const {
        const sema::SymbolInfo* symbol = symbol_at(document, offset);
        if (symbol == nullptr) {
            return nullptr;
        }
        const auto index = static_cast<std::uint32_t>(symbol - document.analysis.symbols.data());
        std::map<std::string, Json::Array> edits;
        const auto add = [&](SourceSpan span) {
            edits[uri_from_path(std::string(document.sources->path(span.file)))].push_back(
                Json::object({{"range", range(*document.sources, span)}, {"newText", new_name}}));
        };
        add(symbol->span);
        for (const sema::Reference& reference : document.analysis.references) {
            if (reference.symbol == index) {
                add(reference.span);
            }
        }
        Json::Object changes;
        for (auto& [uri, list] : edits) {
            changes[uri] = std::move(list);
        }
        return Json::object({{"changes", std::move(changes)}});
    }

    // ---------------------------------------------------------------- symbols

    Json document_symbols(const Document& document) const {
        if (document.program.modules.empty()) {
            return Json::Array{};
        }
        const ast::Ast& tree = document.program.modules.front();
        const SourceManager& sources = *document.sources;
        const std::function<Json(ast::ItemId)> item = [&](ast::ItemId id) -> Json {
            const ast::Item& node = tree.item(id);
            std::string name;
            std::string detail;
            int kind = 13;
            SourceSpan selection = node.span;
            Json::Array children;
            std::visit(Overloaded{
                           [&](const ast::ErrorItem&) {},
                           [&](const ast::ConstDecl& decl) {
                               name = decl.name.text;
                               selection = decl.name.span;
                               kind = 14;
                           },
                           [&](const ast::TypeAliasDecl& decl) {
                               name = decl.name.text;
                               selection = decl.name.span;
                               kind = 26;
                           },
                           [&](const ast::StructDecl& decl) {
                               name = decl.name.text;
                               selection = decl.name.span;
                               kind = 23;
                           },
                           [&](const ast::EnumDecl& decl) {
                               name = decl.name.text;
                               selection = decl.name.span;
                               kind = 10;
                           },
                           [&](const ast::FunctionDecl& decl) {
                               name = decl.name.text;
                               selection = decl.name.span;
                               kind = 12;
                               detail = decl.kind == ast::FunctionKind::Fn   ? "fn"
                                        : decl.kind == ast::FunctionKind::Op ? "op"
                                                                             : "entry";
                           },
                           [&](const ast::BlockDecl& decl) {
                               name = decl.name.text;
                               selection = decl.name.span;
                               kind = 5;
                               for (const ast::ItemId member : decl.members) {
                                   Json child = item(member);
                                   if (!child.is_null()) {
                                       children.push_back(std::move(child));
                                   }
                               }
                           },
                           [&](const ast::MemberDecl& decl) {
                               name = decl.name.text;
                               selection = decl.name.span;
                               kind = 8;
                               detail = decl.kind == ast::MemberKind::Param    ? "param"
                                        : decl.kind == ast::MemberKind::Buffer ? "buffer"
                                                                               : "sub";
                           },
                       },
                       node.data);
            if (name.empty()) {
                return nullptr;
            }
            Json symbol = Json::object({{"name", name},
                                        {"kind", kind},
                                        {"range", range(sources, node.span)},
                                        {"selectionRange", range(sources, selection)}});
            if (!detail.empty()) {
                symbol.set("detail", detail);
            }
            if (!children.empty()) {
                symbol.set("children", std::move(children));
            }
            return symbol;
        };
        Json::Array items;
        for (const ast::ItemId id : tree.items) {
            Json symbol = item(id);
            if (!symbol.is_null()) {
                items.push_back(std::move(symbol));
            }
        }
        return items;
    }

    Json workspace_symbols(const std::string& query) const {
        Json::Array items;
        for (const auto& [key, document] : documents) {
            for (const sema::SymbolInfo& symbol : document.analysis.symbols) {
                const bool is_item = symbol.kind != sema::SymbolKind::Local &&
                                     symbol.kind != sema::SymbolKind::Parameter &&
                                     symbol.kind != sema::SymbolKind::GenericDim &&
                                     symbol.kind != sema::SymbolKind::GenericPack &&
                                     symbol.kind != sema::SymbolKind::GenericDType &&
                                     symbol.kind != sema::SymbolKind::Module;
                if (!is_item || symbol.span.file != document.file ||
                    (!query.empty() && symbol.name.find(query) == std::string::npos)) {
                    continue;
                }
                items.push_back(
                    Json::object({{"name", symbol.name},
                                  {"kind", lsp_symbol_kind(symbol.kind)},
                                  {"location", location(*document.sources, symbol.span)}}));
            }
        }
        return items;
    }

    // ------------------------------------------------------------- completion

    Json completion(const Document& document) const {
        Json::Array items;
        std::set<std::string> seen;
        const auto add = [&](const std::string& label, int kind, const std::string& detail) {
            if (seen.insert(label).second) {
                Json item = Json::object({{"label", label}, {"kind", kind}});
                if (!detail.empty()) {
                    item.set("detail", detail);
                }
                items.push_back(std::move(item));
            }
        };
        for (const sema::SymbolInfo& symbol : document.analysis.symbols) {
            const bool is_visible = symbol.kind != sema::SymbolKind::Local &&
                                    symbol.kind != sema::SymbolKind::Parameter &&
                                    symbol.kind != sema::SymbolKind::GenericDim &&
                                    symbol.kind != sema::SymbolKind::GenericPack &&
                                    symbol.kind != sema::SymbolKind::GenericDType;
            if (is_visible && symbol.span.file == document.file) {
                const int kind = symbol.kind == sema::SymbolKind::Block       ? 7
                                 : symbol.kind == sema::SymbolKind::Struct    ? 22
                                 : symbol.kind == sema::SymbolKind::Enum      ? 13
                                 : symbol.kind == sema::SymbolKind::Const     ? 21
                                 : symbol.kind == sema::SymbolKind::Module    ? 9
                                 : symbol.kind == sema::SymbolKind::TypeAlias ? 7
                                 : symbol.kind == sema::SymbolKind::Param ||
                                         symbol.kind == sema::SymbolKind::Buffer ||
                                         symbol.kind == sema::SymbolKind::Sub
                                     ? 10
                                     : 3;
                add(symbol.name, kind, symbol.detail);
            }
        }
        for (const std::string_view keyword : keywords) {
            add(std::string(keyword), 14, "");
        }
        return items;
    }

    // ------------------------------------------------------------- formatting

    Json formatting(const Document& document) const {
        if (document.has_syntax_errors || document.program.modules.empty()) {
            return nullptr;
        }
        const std::string formatted =
            format::format(document.program.modules.front(), *document.sources);
        if (formatted == document.text) {
            return Json::Array{};
        }
        const SourceManager& sources = *document.sources;
        const SourceSpan whole{document.file, 0, sources.size(document.file)};
        return Json::array(
            {Json::object({{"range", range(sources, whole)}, {"newText", formatted}})});
    }

    // -------------------------------------------------------- semantic tokens

    Json semantic_tokens(const Document& document) const {
        struct Token {
            std::uint32_t begin;
            std::uint32_t length;
            std::uint32_t type;
            std::uint32_t modifiers;
        };
        std::vector<Token> tokens;
        const auto add = [&](SourceSpan span, sema::SymbolKind kind, bool is_declaration) {
            if (span.file != document.file || span.empty()) {
                return;
            }
            const bool is_readonly =
                kind == sema::SymbolKind::Const || kind == sema::SymbolKind::Param ||
                kind == sema::SymbolKind::Buffer || kind == sema::SymbolKind::Parameter;
            tokens.push_back({span.begin,
                              span.size(),
                              token_type_of(kind),
                              (is_declaration ? 1U : 0U) | (is_readonly ? 2U : 0U)});
        };
        for (const sema::SymbolInfo& symbol : document.analysis.symbols) {
            add(symbol.span, symbol.kind, true);
        }
        for (const sema::Reference& reference : document.analysis.references) {
            add(reference.span, document.analysis.symbols[reference.symbol].kind, false);
        }
        std::sort(tokens.begin(), tokens.end(), [](const Token& a, const Token& b) {
            return a.begin < b.begin;
        });

        Json::Array data;
        Utf16Position previous{0, 0};
        std::uint32_t last_begin = 0;
        for (const Token& token : tokens) {
            if (!data.empty() && token.begin == last_begin) {
                continue; // one token per position
            }
            last_begin = token.begin;
            const SourceManager& sources = *document.sources;
            const Utf16Position start = sources.utf16_position(document.file, token.begin);
            const Utf16Position end =
                sources.utf16_position(document.file, token.begin + token.length);
            if (end.line != start.line) {
                continue;
            }
            data.push_back(start.line - previous.line);
            data.push_back(start.line == previous.line ? start.character - previous.character
                                                       : start.character);
            data.push_back(end.character - start.character);
            data.push_back(token.type);
            data.push_back(token.modifiers);
            previous = start;
        }
        return Json::object({{"data", std::move(data)}});
    }

    // ------------------------------------------------------------ inlay hints

    Json inlay_hints(const Document& document, std::uint32_t begin, std::uint32_t end) const {
        Json::Array hints;
        for (const sema::BindingInfo& binding : document.analysis.bindings) {
            if (binding.span.file != document.file || !binding.is_inferred ||
                binding.span.end < begin || binding.span.end > end) {
                continue;
            }
            hints.push_back(Json::object({
                {"position", position(*document.sources, document.file, binding.span.end)},
                {"label", ": " + binding.type},
                {"kind", 1},
                {"paddingLeft", false},
            }));
        }
        return hints;
    }
};

// ---------------------------------------------------------------------- server

Server::Server(ServerOptions options) : state_(std::make_unique<State>()) {
    state_->options = std::move(options);
}

Server::~Server() = default;

bool Server::handle(const Json& message, const std::function<void(const Json&)>& send) {
    State& state = *state_;
    const std::string& method = message["method"].as_string();
    const Json& params = message["params"];
    const Json& id = message["id"];
    const bool is_request = !id.is_null();

    const auto respond = [&](Json result) {
        send(Json::object({{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}}));
    };
    const auto fail = [&](int code, const std::string& text) {
        send(Json::object({{"jsonrpc", "2.0"},
                           {"id", id},
                           {"error", Json::object({{"code", code}, {"message", text}})}}));
    };
    const auto publish = [&](const std::string& key, const Document& document) {
        send(Json::object({{"jsonrpc", "2.0"},
                           {"method", "textDocument/publishDiagnostics"},
                           {"params",
                            Json::object({{"uri", uri_from_path(key)},
                                          {"diagnostics", state.diagnostics_of(document)}})}}));
    };
    // Documents that import a changed document see its new contents.
    const auto reanalyze_all = [&] {
        for (auto& [key, document] : state.documents) {
            state.analyze(key, document);
            publish(key, document);
        }
    };
    const auto document_of = [&](const Json& text_document) -> std::pair<std::string, Document*> {
        const std::string key = State::key_of(path_from_uri(text_document["uri"].as_string()));
        const auto found = state.documents.find(key);
        return {key, found == state.documents.end() ? nullptr : &found->second};
    };

    if (method == "initialize") {
        Json::Array types;
        for (const std::string_view type : token_types) {
            types.emplace_back(type);
        }
        Json::Array modifiers;
        for (const std::string_view modifier : token_modifiers) {
            modifiers.emplace_back(modifier);
        }
        respond(Json::object({
            {"capabilities",
             Json::object({
                 {"textDocumentSync", Json::object({{"openClose", true}, {"change", 1}})},
                 {"hoverProvider", true},
                 {"definitionProvider", true},
                 {"referencesProvider", true},
                 {"renameProvider", true},
                 {"documentSymbolProvider", true},
                 {"workspaceSymbolProvider", true},
                 {"documentFormattingProvider", true},
                 {"inlayHintProvider", true},
                 {"completionProvider", Json::object({{"triggerCharacters", Json::array({"."})}})},
                 {"semanticTokensProvider",
                  Json::object({{"legend",
                                 Json::object({{"tokenTypes", std::move(types)},
                                               {"tokenModifiers", std::move(modifiers)}})},
                                {"full", true}})},
             })},
            {"serverInfo", Json::object({{"name", "linnet"}})},
        }));
        return true;
    }
    if (method == "shutdown") {
        state.has_shut_down = true;
        respond(nullptr);
        return true;
    }
    if (method == "exit") {
        return false;
    }

    if (method == "textDocument/didOpen") {
        const Json& text_document = params["textDocument"];
        const std::string key = State::key_of(path_from_uri(text_document["uri"].as_string()));
        Document& document = state.documents[key];
        document.text = text_document["text"].as_string();
        document.version = text_document["version"].as_int();
        reanalyze_all();
        return true;
    }
    if (method == "textDocument/didChange") {
        auto [key, document] = document_of(params["textDocument"]);
        if (document == nullptr) {
            return true;
        }
        for (const Json& change : params["contentChanges"].as_array()) {
            if (change["range"].is_null()) {
                document->text = change["text"].as_string();
            } else {
                // Incremental edits are not announced; a client that sends
                // them anyway is answered from the last full text.
                const std::uint32_t begin =
                    State::offset_of(*document->sources, document->file, change["range"]["start"]);
                const std::uint32_t end =
                    State::offset_of(*document->sources, document->file, change["range"]["end"]);
                document->text.replace(begin, end - begin, change["text"].as_string());
            }
        }
        document->version = params["textDocument"]["version"].as_int();
        reanalyze_all();
        return true;
    }
    if (method == "textDocument/didClose") {
        auto [key, document] = document_of(params["textDocument"]);
        if (document != nullptr) {
            state.documents.erase(key);
            send(Json::object(
                {{"jsonrpc", "2.0"},
                 {"method", "textDocument/publishDiagnostics"},
                 {"params",
                  Json::object({{"uri", uri_from_path(key)}, {"diagnostics", Json::Array{}}})}}));
        }
        return true;
    }
    if (method == "textDocument/didSave" || method == "initialized" || method.starts_with("$/") ||
        method == "workspace/didChangeConfiguration") {
        return true;
    }

    if (method == "workspace/symbol") {
        respond(state.workspace_symbols(params["query"].as_string()));
        return true;
    }

    if (method.starts_with("textDocument/")) {
        auto [key, document] = document_of(params["textDocument"]);
        if (document == nullptr || document->file == invalid_file_id) {
            if (is_request) {
                respond(nullptr);
            }
            return true;
        }
        const SourceManager& sources = *document->sources;
        const auto offset = [&] {
            return State::offset_of(sources, document->file, params["position"]);
        };
        if (method == "textDocument/hover") {
            respond(state.hover(*document, offset()));
        } else if (method == "textDocument/definition") {
            respond(state.definition(*document, offset()));
        } else if (method == "textDocument/references") {
            respond(state.references(
                *document, offset(), params["context"]["includeDeclaration"].as_bool()));
        } else if (method == "textDocument/rename") {
            respond(state.rename(*document, offset(), params["newName"].as_string()));
        } else if (method == "textDocument/documentSymbol") {
            respond(state.document_symbols(*document));
        } else if (method == "textDocument/completion") {
            respond(state.completion(*document));
        } else if (method == "textDocument/formatting") {
            respond(state.formatting(*document));
        } else if (method == "textDocument/semanticTokens/full") {
            respond(state.semantic_tokens(*document));
        } else if (method == "textDocument/inlayHint") {
            respond(state.inlay_hints(
                *document,
                State::offset_of(sources, document->file, params["range"]["start"]),
                State::offset_of(sources, document->file, params["range"]["end"])));
        } else if (is_request) {
            fail(error_method_not_found, "unsupported method " + method);
        }
        return true;
    }

    if (is_request) {
        fail(params.is_null() && method.empty() ? error_invalid_params : error_method_not_found,
             method.empty() ? "message has no method" : "unsupported method " + method);
    }
    return true;
}

int Server::run(std::istream& in, std::ostream& out) {
    const auto send = [&](const Json& message) {
        const std::string body = message.dump();
        out << "Content-Length: " << body.size() << "\r\n\r\n" << body;
        out.flush();
    };
    while (true) {
        // Headers, then a body of exactly Content-Length bytes.
        std::size_t length = 0;
        std::string line;
        bool has_headers = false;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                has_headers = true;
                break;
            }
            constexpr std::string_view prefix = "Content-Length:";
            if (line.starts_with(prefix)) {
                length = static_cast<std::size_t>(std::stoul(line.substr(prefix.size())));
            }
        }
        if (!has_headers || length == 0) {
            return state_->has_shut_down ? 0 : 1;
        }
        std::string body(length, '\0');
        in.read(body.data(), static_cast<std::streamsize>(length));
        if (in.gcount() != static_cast<std::streamsize>(length)) {
            return 1;
        }
        const auto message = parse_json(body);
        if (!message) {
            send(Json::object(
                {{"jsonrpc", "2.0"},
                 {"id", nullptr},
                 {"error", Json::object({{"code", -32700}, {"message", message.error()}})}}));
            continue;
        }
        if (!handle(*message, send)) {
            return state_->has_shut_down ? 0 : 1;
        }
    }
}

} // namespace linnet::lsp
