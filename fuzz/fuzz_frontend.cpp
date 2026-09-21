// libFuzzer entry point for the frontend. Invariants:
//   - lexing, parsing, and dumping never crash or hang on any input;
//   - input that parses cleanly formats to text that parses cleanly;
//   - formatting is a fixed point.

#include "linnet/ast/dump.hpp"
#include "linnet/format/formatter.hpp"
#include "linnet/syntax/parser.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    using namespace linnet;

    SourceManager sources;
    DiagnosticSink sink;
    const FileId file =
        sources.add_file("fuzz.linnet", std::string(reinterpret_cast<const char*>(data), size))
            .value();
    const ast::Ast tree = parse(sources, file, sink);
    const std::string dumped = ast::dump(tree, sources);
    if (sink.has_errors()) {
        return 0;
    }

    const std::string once = format::format(tree, sources);
    DiagnosticSink second_sink;
    const FileId second = sources.add_file("formatted.linnet", once).value();
    const ast::Ast second_tree = parse(sources, second, second_sink);
    if (second_sink.has_errors() || ast::dump(second_tree, sources) != dumped ||
        second_tree.comments.size() != tree.comments.size() ||
        format::format(second_tree, sources) != once) {
        std::abort();
    }
    return 0;
}
