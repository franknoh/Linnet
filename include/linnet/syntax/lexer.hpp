#pragma once

#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/source/source_manager.hpp"
#include "linnet/syntax/token.hpp"

#include <vector>

namespace linnet {

struct LexResult {
    std::vector<Token> tokens;     // always terminated by one Eof token
    std::vector<Comment> comments; // in source order; never part of `tokens`
};

// Tokenizes a whole file. Never fails: malformed input is reported to `sink`
// and lexing continues. Malformed literals still yield a literal token, and
// characters that cannot start a token are skipped, so the parser sees as
// little damage as possible.
LexResult lex(const SourceManager& sources, FileId file, DiagnosticSink& sink);

} // namespace linnet
