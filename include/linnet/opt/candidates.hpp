#pragma once

#include "linnet/ir/ir.hpp"
#include "linnet/opt/passes.hpp"

#include <string>
#include <vector>

namespace linnet::opt {

// One way a backend could implement a semantic operation. The canonical
// decomposition — the op's own body — is always available and always
// selected until a backend registers something better.
struct Candidate {
    std::string implementation; // "canonical decomposition", "torch.sdpa", ...
    Legality legality = Legality::Exact;
    std::vector<std::string> requirements; // conditions the candidate needs
    bool is_selected = false;
    std::string reason;
};

struct Explanation {
    std::string function;    // where the call is
    std::string semantic_op; // the callee's semantic identity
    SourceSpan span;         // the call site
    std::vector<Candidate> candidates;
};

// Lists every semantic call of the module with its candidates. A backend
// that claims ops adds candidates; with none registered, every call reports
// only its canonical decomposition.
std::vector<Explanation> explain(const ir::Module& module);

// Text rendering for `linnet explain`.
std::string render_explanations(const std::vector<Explanation>& explanations,
                                const SourceManager& sources);

} // namespace linnet::opt
