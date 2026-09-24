#pragma once

#include "linnet/ir/ir.hpp"
#include "linnet/opt/passes.hpp"

#include <optional>
#include <string>
#include <string_view>
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

// A backend's registered implementations of semantic operations, by the
// operation's semantic identity.
struct NativeCandidate {
    std::string semantic_op;
    std::string implementation;
    Legality legality;
    std::vector<std::string> requirements;
};

// The PyTorch backend's registry: library calls whose results agree with the
// canonical decomposition up to floating-point rounding (or exactly, for a
// gather and a mask). Every graph target reads the same registry and spells
// the implementations it has; the rest fall back to the canonical body.
std::vector<NativeCandidate> torch_candidates();

// Chooses an implementation for every semantic call: the strongest registered
// candidate whose legality is within `allowed`, else the canonical
// decomposition. Records the choice in the call's attributes (`name` is
// unchanged; `names` gains the selected implementation as its only entry).
void select_candidates(ir::Module& module,
                       const std::vector<NativeCandidate>& registry,
                       Legality allowed);

// Lists every semantic call of the module with its candidates, as selected by
// `select_candidates` (or the canonical decomposition when it did not run).
std::vector<Explanation>
explain(ir::Module& module, const std::vector<NativeCandidate>& registry, Legality allowed);

// Parses "exact", "equivalent" (numerically equivalent), or "fast"
// (approximate: kernels in the input dtype).
std::optional<Legality> parse_legality(std::string_view text);

// Text rendering for `linnet explain`.
std::string render_explanations(const std::vector<Explanation>& explanations,
                                const SourceManager& sources);

} // namespace linnet::opt
