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
        // Inputs taken in an order (GPTQ's activation order): a gather.
        {"std.quant::take_inputs", "torch.index_select", Legality::Exact, {}},
        {"std.nn.attention::causal_mask", "torch.tril", Legality::Exact, {}},
        {"std.nn.conv::conv2d", "torch.nn.functional.conv2d", equivalent, {"a square kernel"}},
        {"std.nn.conv::conv2d_rect", "torch.nn.functional.conv2d(rect)", equivalent, {}},
        {"std.nn.conv::conv1d", "torch.nn.functional.conv1d", equivalent, {}},
        // A maximum is one of its inputs, however it is found.
        {"std.nn.pool::max_pool2d",
         "torch.nn.functional.max_pool2d",
         Legality::Exact,
         {"the padding is at most half the window"}},
        // tinygemm's 4-bit matrix product (and ONNX Runtime's MatMulNBits):
        // the scales and zero points meet the weights in the input dtype.
        {"std.quant::linear_int4_groups",
         "torch.ops.aten._weight_int4pack_mm",
         fast,
         {"CUDA, bf16, and a group size the kernel takes; otherwise the body runs"}},
        // FP8 bytes read as `float8_e4m3fn` and widened: the same values the
        // body computes from the bits.
        {"std.quant::decode_fp8",
         "linnet.decode_fp8",
         Legality::Exact,
         {"no byte is the NaN pattern, which the body reads as 480"}},
        // The input rounded to FP8 a row at a time and multiplied in FP8
        // (`torch._scaled_mm`): the body multiplies it unrounded.
        {"std.quant::linear_fp8",
         "torch._scaled_mm",
         fast,
         {"CUDA (compute capability 9 or later), bf16, both widths multiples of 16; otherwise the "
          "weight is decoded and multiplied"}},
        // A grouped matrix product over rows sorted by expert, reading each
        // chosen expert's weight where it lies; it accumulates in f32 and
        // rounds once to bf16, as the body's cast does.
        {"std.nn.moe::linear_experts",
         "torch._grouped_mm",
         fast,
         {"CUDA (compute capability 9 or later), bf16, and widths the kernel takes; otherwise "
          "the chosen experts are gathered"}},
        // The same product for an input every chosen expert shares, and for
        // inputs of their own weighed and summed: in place of the bodies'
        // product with every expert.
        {"std.nn.moe::linear_experts_shared",
         "torch._grouped_mm(shared)",
         fast,
         {"CUDA (compute capability 9 or later), bf16, and widths the kernel takes; otherwise "
          "every expert multiplies every row"}},
        {"std.nn.moe::combine_experts",
         "torch._grouped_mm(combined)",
         fast,
         {"CUDA (compute capability 9 or later), bf16, and widths the kernel takes; otherwise "
          "every expert multiplies every row"}},
        {"std.nn.pool::global_average_pool2d", "torch.Tensor.mean", equivalent, {}},
        {"std.nn.norm::batch_norm", "torch.nn.functional.batch_norm", equivalent, {}},
        {"std.nn.norm::group_norm", "torch.nn.functional.group_norm", equivalent, {}},
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
        {"std.nn.cache::write_slots",
         "torch.Tensor.index_put",
         Legality::Exact,
         {"the slots are distinct and the span is inside the cache"}},
        {"std.quant::mxfp4_experts",
         "linnet.mxfp4_experts",
         equivalent,
         {"Triton on CUDA for a few rows; the chosen experts unpacked otherwise"}},
        {"std.quant::mxfp4_linear_experts_shared",
         "linnet.mxfp4_grouped(shared)",
         fast,
         {"Triton on CUDA, each expert times the rows that chose it; dequantized otherwise"}},
        {"std.quant::mxfp4_combine_experts",
         "linnet.mxfp4_grouped(combine)",
         fast,
         {"Triton on CUDA, each expert times the rows that chose it; dequantized otherwise"}},
        {"std.quant::mxfp4_experts_shared",
         "linnet.mxfp4_experts(shared)",
         equivalent,
         {"Triton on CUDA for a few rows; the chosen experts unpacked otherwise"}},
        // An output head and its loss a block of tokens at a time, never
        // holding the [N, V] logits; the sums run in another order.
        {"std.nn.loss::linear_cross_entropy",
         "linnet.linear_cross_entropy",
         equivalent,
         {"PyTorch: blocks of rows, the gradient computed with the loss"}},
        {"std.nn.loss::linear_token_log_probs",
         "linnet.linear_token_log_probs",
         equivalent,
         {"PyTorch: blocks of rows, multiplied again in backward"}},
        // The same over a vocabulary split across the shards' processes: each
        // block's maxima, sums and target logits combined, never the logits.
        {"std.nn.loss::split_cross_entropy",
         "linnet.split_cross_entropy",
         equivalent,
         {"PyTorch: blocks of rows, the gradient computed with the loss"}},
        {"std.nn.loss::split_token_log_probs",
         "linnet.split_token_log_probs",
         equivalent,
         {"PyTorch: blocks of rows, multiplied again in backward"}},
        {"std.nn.parallel::shared",
         "torch.distributed.shared",
         Legality::Exact,
         {"one shard, or one process per shard reading the input whole"}},
        {"std.nn.parallel::all_reduce",
         "torch.distributed.all_reduce",
         Legality::Exact,
         {"one shard, or one process per shard holding its part of the sum"}},
        {"std.nn.parallel::all_gather",
         "torch.distributed.all_gather",
         Legality::Exact,
         {"one shard, or one process per shard holding its slice"}},
        {"std.nn.cache::write_tokens",
         "torch.Tensor.index_put(tokens)",
         Legality::Exact,
         {"the (row, position) pairs are distinct and inside the cache"}},
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
        {"std.nn.attention::sink_attention",
         "linnet.sink_attention",
         fast,
         {"mask is boolean with true meaning attend", "key/value heads divide query heads"}},
        // FlexAttention over each row's pages where they lie in the pool.
        {"std.nn.attention::paged_attention",
         "linnet.paged_attention",
         fast,
         {"key/value heads divide query heads", "every page a row reads lies in the pool"}},
        // The same over the pool for prompt tokens, each page read once for
        // the tokens that see it.
        {"std.nn.attention::paged_prefill_attention",
         "linnet.paged_prefill_attention",
         fast,
         {"key/value heads divide query heads",
          "every page a row reads lies in the pool",
          "a page holds the same positions in every row that lists it"}},
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
        if (function.gradient != ir::no_id) {
            pending.push_back(function.gradient);
        }
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
