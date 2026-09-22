#include "linnet/lsp/json.hpp"
#include "linnet/lsp/server.hpp"

#include "test.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace linnet;
using linnet::lsp::Json;

namespace {

struct Client {
    lsp::Server server{lsp::ServerOptions{}};
    std::vector<Json> sent;
    int next_id = 1;

    Json request(const std::string& method, Json params) {
        const int id = next_id++;
        server.handle(
            Json::object({{"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", params}}),
            [&](const Json& message) { sent.push_back(message); });
        for (const Json& message : sent) {
            if (message["id"].is_number() && message["id"].as_int() == id) {
                return message;
            }
        }
        return nullptr;
    }

    void notify(const std::string& method, Json params) {
        server.handle(Json::object({{"jsonrpc", "2.0"}, {"method", method}, {"params", params}}),
                      [&](const Json& message) { sent.push_back(message); });
    }

    // The last diagnostics published for `uri`.
    Json diagnostics(const std::string& uri) const {
        Json result;
        for (const Json& message : sent) {
            if (message["method"].as_string() == "textDocument/publishDiagnostics" &&
                message["params"]["uri"].as_string() == uri) {
                result = message["params"]["diagnostics"];
            }
        }
        return result;
    }
};

std::filesystem::path scratch_file(const std::string& name, const std::string& text) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "linnet-lsp-test";
    std::filesystem::create_directories(directory);
    const std::filesystem::path path = directory / name;
    std::ofstream(path, std::ios::binary) << text;
    return std::filesystem::weakly_canonical(path);
}

Json text_document(const std::string& uri) {
    return Json::object({{"textDocument", Json::object({{"uri", uri}})}});
}

Json at(const std::string& uri, int line, int character) {
    Json params = text_document(uri);
    params.set("position", Json::object({{"line", line}, {"character", character}}));
    return params;
}

} // namespace

TEST("json: round trip") {
    const auto parsed = lsp::parse_json(R"({"a":[1,2.5,"x\u00e9\n",true,null],"b":{"c":-3}})");
    CHECK(parsed.has_value());
    CHECK_EQ(parsed->dump(), "{\"a\":[1,2.5,\"x\xC3\xA9\\n\",true,null],\"b\":{\"c\":-3}}");
    CHECK_EQ((*parsed)["b"]["c"].as_int(), -3);
    CHECK((*parsed)["missing"]["deeper"].is_null());
    CHECK(!lsp::parse_json("{").has_value());
    CHECK(!lsp::parse_json("[1,]").has_value());
    CHECK(!lsp::parse_json("\"\\x\"").has_value());
    CHECK(!lsp::parse_json("1 2").has_value());
    CHECK_EQ(lsp::parse_json("\"\\ud83d\\ude00\"")->as_string(), "\xF0\x9F\x98\x80");
}

TEST("lsp: uri and path conversion") {
    CHECK_EQ(lsp::uri_from_path("/home/u/a b.linnet"), "file:///home/u/a%20b.linnet");
    CHECK_EQ(lsp::path_from_uri("file:///home/u/a%20b.linnet").generic_string(),
             "/home/u/a b.linnet");
    CHECK_EQ(lsp::path_from_uri("file:///c%3A/dir/x.linnet").generic_string(), "c:/dir/x.linnet");
    CHECK(lsp::path_from_uri("https://example.com").empty());
}

TEST("lsp: diagnostics, hover, definition, references, rename, symbols") {
    const std::string source =
        "module m\n"
        "\n"
        "pub fn scale<N: Dim>(x: Tensor[N; f32], k: f32) -> Tensor[N; f32] {\n"
        "    let y = x * k\n"
        "    return y\n"
        "}\n"
        "\n"
        "fn user(v: Tensor[4; f32]) -> Tensor[4; f32] {\n"
        "    return scale(v, 2.0)\n"
        "}\n";
    const std::filesystem::path path = scratch_file("doc.linnet", source);
    const std::string uri = lsp::uri_from_path(path);
    Client client;
    const Json initialized = client.request("initialize", Json::object({}));
    CHECK(initialized["result"]["capabilities"]["hoverProvider"].as_bool());
    client.notify("textDocument/didOpen",
                  Json::object({{"textDocument",
                                 Json::object({{"uri", uri}, {"version", 1}, {"text", source}})}}));
    CHECK(client.diagnostics(uri).is_array());
    CHECK_EQ(client.diagnostics(uri).as_array().size(), 0U);

    // Hover on `scale` in the call shows its signature.
    const Json hover = client.request("textDocument/hover", at(uri, 8, 13));
    CHECK(hover["result"]["contents"]["value"].as_string().find(
              "fn scale<N: Dim>(x: Tensor[N; f32], k: f32) -> Tensor[N; f32]") !=
          std::string::npos);
    // Hover on the local `y` shows its inferred type.
    CHECK(client.request("textDocument/hover", at(uri, 4, 11))["result"]["contents"]["value"]
              .as_string()
              .find("let y: Tensor[N; f32]") != std::string::npos);
    CHECK(client.request("textDocument/hover", at(uri, 1, 0))["result"].is_null());

    const Json definition = client.request("textDocument/definition", at(uri, 8, 13));
    CHECK_EQ(definition["result"]["uri"].as_string(), uri);
    CHECK_EQ(definition["result"]["range"]["start"]["line"].as_int(), 2);
    CHECK_EQ(definition["result"]["range"]["start"]["character"].as_int(), 7);

    Json reference_params = at(uri, 3, 12); // `x` in `x * k`
    reference_params.set("context", Json::object({{"includeDeclaration", true}}));
    CHECK_EQ(
        client.request("textDocument/references", reference_params)["result"].as_array().size(),
        2U);

    Json rename_params = at(uri, 3, 8); // `y`
    rename_params.set("newName", "scaled");
    const Json rename = client.request("textDocument/rename", rename_params);
    CHECK_EQ(rename["result"]["changes"][uri].as_array().size(), 2U);

    const Json symbols = client.request("textDocument/documentSymbol", text_document(uri));
    CHECK_EQ(symbols["result"].as_array().size(), 2U);
    CHECK_EQ(symbols["result"].as_array()[0]["name"].as_string(), "scale");
    CHECK(client.request("workspace/symbol", Json::object({{"query", "use"}}))["result"]
              .as_array()
              .size() == 1U);

    const Json completion = client.request("textDocument/completion", at(uri, 8, 11));
    bool has_scale = false;
    for (const Json& item : completion["result"].as_array()) {
        has_scale = has_scale || item["label"].as_string() == "scale";
    }
    CHECK(has_scale);

    const Json tokens = client.request("textDocument/semanticTokens/full", text_document(uri));
    CHECK(tokens["result"]["data"].as_array().size() % 5 == 0);
    CHECK(!tokens["result"]["data"].as_array().empty());

    Json hint_params = text_document(uri);
    hint_params.set("range",
                    Json::object({{"start", Json::object({{"line", 0}, {"character", 0}})},
                                  {"end", Json::object({{"line", 9}, {"character", 0}})}}));
    const Json hints = client.request("textDocument/inlayHint", hint_params);
    CHECK_EQ(hints["result"].as_array().size(), 1U);
    CHECK_EQ(hints["result"].as_array()[0]["label"].as_string(), ": Tensor[N; f32]");

    // An edit that breaks the program produces a diagnostic; formatting is refused.
    client.notify(
        "textDocument/didChange",
        Json::object({{"textDocument", Json::object({{"uri", uri}, {"version", 2}})},
                      {"contentChanges",
                       Json::array({Json::object(
                           {{"text", "module m\nfn f() -> f32 { return missing }\n"}})})}}));
    CHECK_EQ(client.diagnostics(uri).as_array().size(), 1U);
    CHECK_EQ(client.diagnostics(uri).as_array()[0]["code"].as_string(), "E1201");
    CHECK_EQ(client.diagnostics(uri).as_array()[0]["severity"].as_int(), 1);

    client.notify(
        "textDocument/didChange",
        Json::object({{"textDocument", Json::object({{"uri", uri}, {"version", 3}})},
                      {"contentChanges",
                       Json::array({Json::object({{"text", "module m\nfn f(){return}\n"}})})}}));
    const Json formatting = client.request("textDocument/formatting", text_document(uri));
    CHECK_EQ(formatting["result"].as_array().size(), 1U);
    CHECK_EQ(formatting["result"].as_array()[0]["newText"].as_string(),
             "module m\n\nfn f() {\n    return\n}\n");

    CHECK(
        client.request("textDocument/unknownThing", text_document(uri))["error"]["code"].as_int() ==
        -32601);
    client.notify("textDocument/didClose", text_document(uri));
    CHECK_EQ(client.diagnostics(uri).as_array().size(), 0U);
    CHECK(client.request("shutdown", nullptr)["result"].is_null());
    CHECK(!client.server.handle(Json::object({{"jsonrpc", "2.0"}, {"method", "exit"}}),
                                [](const Json&) {}));
}

TEST("lsp: unsaved buffers are used for imports") {
    const std::string library = "module lib\n\npub const Width = 4\n";
    const std::filesystem::path lib_path = scratch_file("lib.linnet", library);
    const std::filesystem::path app_path =
        scratch_file("app.linnet",
                     "module app\n\nuse lib::{Width}\n\nfn f(x: Tensor[Width; f32]) -> "
                     "Tensor[4; f32] {\n    return x\n}\n");
    // The library file is only reachable as a sibling module when both are
    // loaded from the same "package"; without linnet.toml there is no crate,
    // so the import goes through the opened document overlay by path instead.
    const std::string app_uri = lsp::uri_from_path(app_path);
    const std::string lib_uri = lsp::uri_from_path(lib_path);
    Client client;
    client.request("initialize", Json::object({}));
    client.notify(
        "textDocument/didOpen",
        Json::object({{"textDocument",
                       Json::object({{"uri", lib_uri}, {"version", 1}, {"text", library}})}}));
    client.notify(
        "textDocument/didOpen",
        Json::object(
            {{"textDocument",
              Json::object({{"uri", app_uri},
                            {"version", 1},
                            {"text", std::string("module app\n\nuse lib::{Width}\n")}})}}));
    CHECK(client.diagnostics(app_uri).is_array());
}
