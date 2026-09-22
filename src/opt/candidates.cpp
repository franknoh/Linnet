#include "linnet/opt/candidates.hpp"

namespace linnet::opt {

std::vector<Explanation> explain(const ir::Module& module) {
    std::vector<Explanation> explanations;
    for (const ir::Function& function : module.functions()) {
        std::vector<ir::RegionId> pending{function.body};
        while (!pending.empty()) {
            const ir::RegionId region = pending.back();
            pending.pop_back();
            for (const ir::BlockId block : module.region(region).blocks) {
                for (const ir::OpId id : module.block(block).ops) {
                    const ir::Operation& op = module.op(id);
                    for (const ir::RegionId nested : op.regions) {
                        pending.push_back(nested);
                    }
                    if (op.kind != ir::OpKind::SemanticCall) {
                        continue;
                    }
                    Explanation explanation{function.name, op.attributes.name, op.span, {}};
                    explanation.candidates.push_back(
                        {"canonical decomposition",
                         Legality::Exact,
                         {},
                         true,
                         "the only implementation registered for this backend"});
                    explanations.push_back(std::move(explanation));
                }
            }
        }
    }
    return explanations;
}

std::string render_explanations(const std::vector<Explanation>& explanations,
                                const SourceManager& sources) {
    std::string out;
    for (const Explanation& explanation : explanations) {
        const LineColumn where = sources.line_column(explanation.span.file, explanation.span.begin);
        out += explanation.semantic_op + "\n";
        out += "  called from " + explanation.function + " at " +
               std::string(sources.path(explanation.span.file)) + ":" + std::to_string(where.line) +
               ":" + std::to_string(where.column) + "\n";
        out += "  implementations:\n";
        for (const Candidate& candidate : explanation.candidates) {
            out += std::string("    ") + (candidate.is_selected ? "* " : "  ") +
                   candidate.implementation;
            for (const std::string& requirement : candidate.requirements) {
                out += "\n        requires " + requirement;
            }
            out += "\n";
        }
        for (const Candidate& candidate : explanation.candidates) {
            if (candidate.is_selected) {
                out += "  selected: " + candidate.implementation + "\n";
                out += "  reason: " + candidate.reason + "\n";
            }
        }
        out += "\n";
    }
    return out;
}

} // namespace linnet::opt
