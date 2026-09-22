#pragma once

#include "linnet/lsp/json.hpp"
#include "linnet/module/loader.hpp"

#include <filesystem>
#include <functional>
#include <istream>
#include <map>
#include <memory>
#include <ostream>
#include <string>

namespace linnet::lsp {

struct ServerOptions {
    std::filesystem::path std_root;
};

// The language server. Messages are JSON-RPC 2.0 over `Content-Length` framed
// stdio; `run` returns after `exit`, or when the input ends.
//
// Every open document is analyzed on its own, together with the modules it
// imports (unsaved buffers of other open documents are used in place of the
// files on disk). All language knowledge comes from the compiler library;
// the server only translates between LSP positions and source offsets.
class Server {
public:
    explicit Server(ServerOptions options);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    int run(std::istream& in, std::ostream& out);

    // Handles one decoded message; responses and notifications go to `send`.
    // Exposed for tests. Returns false once `exit` was received.
    bool handle(const Json& message, const std::function<void(const Json&)>& send);

private:
    struct Document;
    struct State;
    std::unique_ptr<State> state_;
};

// file:// URIs <-> filesystem paths, with percent-encoding and Windows drives.
std::string uri_from_path(const std::filesystem::path& path);
std::filesystem::path path_from_uri(std::string_view uri);

} // namespace linnet::lsp
