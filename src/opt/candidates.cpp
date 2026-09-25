#include "linnet/opt/candidates.hpp"

namespace linnet::opt {

std::vector<NativeCandidate> torch_candidates() {
    // Each library call reorders or fuses floating-point arithmetic relative
    // to the `.linnet` body, so results agree only up to rounding.
    const Legality equivalent = Legality::NumericallyEquivalent;
    // `(input dtype)` variants skip the canonical f32 accumulation and run
    // the kernel in the tensor's own dtype: faster in bf16, within rounding
    // of the body only approximately.
    const Legality fast = Legality::Approximate;
    // A gather and a boolean mask are the same numbers however they are
    // computed, so these two are Exact and selected under every policy.
    return {
        {"std.nn.embedding::embedding", "torch.nn.functional.embedding", Legality::Exact, {}},
        {"std.nn.attention::causal_mask", "torch.tril", Legality::Exact, {}},
        {"std.nn.conv::conv2d", "torch.nn.functional.conv2d", equivalent, {"a square kernel"}},
        {"std.nn.pool::global_average_pool2d", "torch.Tensor.mean", equivalent, {}},
        {"std.nn.norm::batch_norm", "torch.nn.functional.batch_norm", equivalent, {}},
        {"std.nn.resize::upsample_nearest2d",
         "torch.nn.functional.interpolate(nearest)",
         Legality::Exact,
         {"a whole scale factor"}},
        {"std.nn.cache::write_at",
         "torch.Tensor.index_copy",
         Legality::Exact,
         {"the position is inside the cache"}},
        {"std.nn.cache::write_span",
         "torch.Tensor.index_copy",
         Legality::Exact,
         {"the span is inside the cache"}},
        {"std.nn.cache::write_rows",
         "torch.Tensor.index_put",
         Legality::Exact,
         {"every position is inside the cache"}},
        {"std.nn.cache::write_slot",
         "torch.Tensor.index_put",
         Legality::Exact,
         {"the slot and the span are inside the cache"}},
        {"std.nn.attention::grouped_attention_rows",
         "torch.nn.functional.scaled_dot_product_attention(enable_gqa)(input dtype)",
         fast,
         {"mask is boolean with true meaning attend", "key/value heads divide query heads"}},
        {"std.nn.attention::grouped_attention_rows",
         "torch.nn.functional.scaled_dot_product_attention(enable_gqa)",
         equivalent,
         {"mask is boolean with true meaning attend", "key/value heads divide query heads"}},
        {"std.nn.attention::attention_rows",
         "torch.nn.functional.scaled_dot_product_attention(input dtype)",
         fast,
         {"mask is boolean with true meaning attend"}},
        {"std.nn.attention::attention_rows",
         "torch.nn.functional.scaled_dot_product_attention",
         equivalent,
         {"mask is boolean with true meaning attend"}},
        {"std.nn.attention::grouped_attention",
         "torch.nn.functional.scaled_dot_product_attention(enable_gqa)(input dtype)",
         fast,
         {"mask is boolean with true meaning attend", "key/value heads divide query heads"}},
        {"std.nn.attention::grouped_attention",
         "torch.nn.functional.scaled_dot_product_attention(enable_gqa)",
         equivalent,
         {"mask is boolean with true meaning attend", "key/value heads divide query heads"}},
        {"std.nn.softmax::softmax", "torch.softmax(input dtype)", fast, {}},
        {"std.nn.norm::rms_norm", "torch.rms_norm(input dtype)", fast, {}},
        {"std.nn.norm::layer_norm", "torch.nn.functional.layer_norm(input dtype)", fast, {}},
        {"std.nn.attention::attention",
         "torch.nn.functional.scaled_dot_product_attention(input dtype)",
         fast,
         {"mask is boolean with true meaning attend"}},
        {"std.linalg::matmul", "torch.matmul", equivalent, {}},
        {"std.linalg::batched_matmul", "torch.matmul", equivalent, {}},
        {"std.nn.linear::linear", "torch.nn.functional.linear", equivalent, {}},
        {"std.nn.softmax::softmax", "torch.softmax", equivalent, {}},
        {"std.nn.activations::relu", "torch.relu", Legality::IEEEEquivalent, {}},
        {"std.nn.activations::sigmoid", "torch.sigmoid", equivalent, {}},
        {"std.nn.activations::silu", "torch.nn.functional.silu", equivalent, {}},
        {"std.nn.activations::gelu", "torch.nn.functional.gelu(tanh)", equivalent, {}},
        {"std.nn.activations::gelu_erf", "torch.nn.functional.gelu", equivalent, {}},
        {"std.nn.norm::rms_norm", "torch.rms_norm", equivalent, {}},
        {"std.nn.norm::layer_norm", "torch.nn.functional.layer_norm", equivalent, {}},
        {"std.nn.attention::attention",
         "torch.nn.functional.scaled_dot_product_attention",
         equivalent,
         {"mask is boolean with true meaning attend"}},
    };
}

namespace {

const char* canonical_name = "canonical decomposition";

// The most permissive candidate the policy allows; null for the decomposition.
const NativeCandidate* choose(const std::string& semantic_op,
                              const std::vector<NativeCandidate>& registry,
                              Legality allowed) {
    const NativeCandidate* best = nullptr;
    for (const NativeCandidate& candidate : registry) {
        if (candidate.semantic_op == semantic_op && candidate.legality <= allowed &&
            (best == nullptr || candidate.legality > best->legality)) {
            best = &candidate;
        }
    }
    return best;
}

template <typename Visit>
void for_each_semantic_call(ir::Module& module, Visit&& visit) {
    for (std::size_t i = 0; i < module.functions().size(); ++i) {
        const ir::Function& function = module.functions()[i];
        std::vector<ir::RegionId> pending{function.body};
        while (!pending.empty()) {
            const ir::RegionId region = pending.back();
            pending.pop_back();
            for (const ir::BlockId block : module.region(region).blocks) {
                for (const ir::OpId id : module.block(block).ops) {
                    for (const ir::RegionId nested : module.op(id).regions) {
                        pending.push_back(nested);
                    }
                    if (module.op(id).kind == ir::OpKind::SemanticCall) {
                        visit(function, module.op(id));
                    }
                }
            }
        }
    }
}

} // namespace

void select_candidates(ir::Module& module,
                       const std::vector<NativeCandidate>& registry,
                       Legality allowed) {
    for_each_semantic_call(module, [&](const ir::Function&, ir::Operation& op) {
        const NativeCandidate* chosen = choose(op.attributes.name, registry, allowed);
        op.attributes.names = {chosen == nullptr ? canonical_name : chosen->implementation};
    });
}

std::optional<Legality> parse_legality(std::string_view text) {
    if (text == "exact") {
        return Legality::Exact;
    }
    if (text == "equivalent") {
        return Legality::NumericallyEquivalent;
    }
    if (text == "fast") {
        return Legality::Approximate;
    }
    return std::nullopt;
}

std::vector<Explanation>
explain(ir::Module& module, const std::vector<NativeCandidate>& registry, Legality allowed) {
    std::vector<Explanation> explanations;
    for_each_semantic_call(module, [&](const ir::Function& function, ir::Operation& op) {
        Explanation explanation{function.name, op.attributes.name, op.span, {}};
        const NativeCandidate* chosen = choose(op.attributes.name, registry, allowed);
        explanation.candidates.push_back(
            {canonical_name,
             Legality::Exact,
             {},
             chosen == nullptr,
             chosen == nullptr ? "no registered implementation is allowed by the numerics policy"
                               : ""});
        for (const NativeCandidate& candidate : registry) {
            if (candidate.semantic_op != op.attributes.name) {
                continue;
            }
            const bool is_selected = chosen == &candidate;
            explanation.candidates.push_back(
                {candidate.implementation,
                 candidate.legality,
                 candidate.requirements,
                 is_selected,
                 is_selected ? "strongest implementation allowed by the numerics policy" : ""});
        }
        explanations.push_back(std::move(explanation));
    });
    return explanations;
}

std::string render_explanations(const std::vector<Explanation>& explanations,
                                const SourceManager& sources) {
    const auto legality_name = [](Legality legality) {
        switch (legality) {
        case Legality::Exact:
            return "exact";
        case Legality::IEEEEquivalent:
            return "IEEE-equivalent";
        case Legality::NumericallyEquivalent:
            return "numerically equivalent";
        case Legality::Approximate:
            return "approximate";
        }
        return "?";
    };
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
                   candidate.implementation + " (" + legality_name(candidate.legality) + ")";
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
