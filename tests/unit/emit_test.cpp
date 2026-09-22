#include "linnet/emit/source.hpp"
#include "linnet/ir/lower.hpp"
#include "linnet/opt/passes.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

// Emits `source` back from its Core IR; `optimize` runs the canonical
// pipeline first. Fails the test when the source does not check.
std::string emitted(const std::string& source, bool optimize = false) {
    SourceManager sources;
    DiagnosticSink sink;
    const FileId file = sources.add_file("m.linnet", source).value();
    const ast::Ast tree = parse(sources, file, sink);
    const ast::Ast* const modules[] = {&tree};
    const sema::AnalysisResult analysis = sema::analyze(sources, modules, sink);
    CHECK(!sink.has_errors());
    ir::Module module = ir::lower(sources, modules, analysis.model);
    if (optimize) {
        opt::run_pipeline(module, opt::canonical_passes());
    }
    const auto text = emit::emit_source(module, emit::EmitOptions{});
    CHECK(text.has_value());
    return text.value_or("");
}

bool checks(const std::string& source) {
    SourceManager sources;
    DiagnosticSink sink;
    const FileId file = sources.add_file("m.linnet", source).value();
    const ast::Ast tree = parse(sources, file, sink);
    const ast::Ast* const modules[] = {&tree};
    sema::analyze(sources, modules, sink);
    return !sink.has_errors();
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// Emitting, checking, and emitting again must be a fixed point.
void check_round_trip(const std::string& source) {
    const std::string once = emitted(source);
    CHECK(checks(once));
    CHECK(emitted(once) == once);
    const std::string optimized = emitted(source, true);
    CHECK(checks(optimized));
}

} // namespace

TEST("emit: index notation, reductions, and let bindings") {
    const std::string source =
        "module m\n"
        "pub fn f<B: Dim, N: Dim>(x: Tensor[B, N; f32]) -> Tensor[B; f32] {\n"
        "    let scaled[b, n] = x[b, n] * 2.0\n"
        "    let total[b] = sum[n] scaled[b, n]\n"
        "    return total\n"
        "}\n";
    const std::string text = emitted(source);
    CHECK(contains(text, "module m\n"));
    CHECK(contains(text, "pub fn f<B: Dim, N: Dim>(x: Tensor[B, N; f32]) -> Tensor[B; f32] {"));
    CHECK(contains(text, "let scaled[b, n] = (x[b, n] * 2.0)"));
    CHECK(contains(text, "let total[b] = sum[n] scaled[b, n]"));
    CHECK(contains(text, "return total"));
    check_round_trip(source);
}

TEST("emit: shape packs, generic arguments, and prelude calls") {
    const std::string source =
        "module m\n"
        "fn scale<*S: Shape, N: Dim, T: Float>(x: Tensor[*S, N; T], k: T) -> Tensor[*S, N; T] {\n"
        "    let y[*s, n] = x[*s, n] * k\n"
        "    return y\n"
        "}\n"
        "pub fn g<B: Dim, N: Dim>(x: Tensor[B, N; bf16]) -> Tensor[N, B; f32] {\n"
        "    let two = cast<bf16>(2.0)\n"
        "    return permute(cast<f32>(scale(x, two)), [1, 0])\n"
        "}\n";
    const std::string text = emitted(source);
    CHECK(contains(text, "fn scale<*S: Shape, N: Dim, T: Float>("));
    CHECK(contains(text, "let y[*s, n] = (x[*s, n] * k)"));
    CHECK(contains(text, "scale<[B], N, bf16>(x, cast<bf16>(2.0))"));
    CHECK(contains(text, "permute("));
    check_round_trip(source);
}

TEST("emit: blocks with members, defaults, and methods") {
    const std::string source = "module m\n"
                               "pub block Affine<N: Dim, T: Float = f32> {\n"
                               "    param weight: Tensor[N; T]\n"
                               "    param bias: Tensor[N; T]? = none\n"
                               "    buffer count: Tensor[N; T]\n"
                               "    pub fn forward(x: Tensor[N; T]) -> Tensor[N; T] {\n"
                               "        let y = x * weight + count\n"
                               "        return match bias {\n"
                               "            some(b) => y + b\n"
                               "            none => y\n"
                               "        }\n"
                               "    }\n"
                               "}\n";
    const std::string text = emitted(source);
    CHECK(contains(text, "pub block Affine<N: Dim, T: Float = f32> {"));
    CHECK(contains(text, "    param weight: Tensor[N; T]\n"));
    CHECK(contains(text, "    param bias: Tensor[N; T]? = none\n"));
    CHECK(contains(text, "    buffer count: Tensor[N; T]\n"));
    CHECK(contains(text, "    pub fn forward(x: Tensor[N; T]) -> Tensor[N; T] {"));
    CHECK(contains(text, "match bias { some(b) => (y + b) none => y }"));
    check_round_trip(source);
}

TEST("emit: static for over sub arrays and tuple destructuring") {
    const std::string source = "module m\n"
                               "block Layer<N: Dim> {\n"
                               "    param w: Tensor[N; f32]\n"
                               "    pub fn forward(x: Tensor[N; f32]) -> Tensor[N; f32] {\n"
                               "        return x * w\n"
                               "    }\n"
                               "}\n"
                               "pub block Stack<N: Dim, L: Dim> {\n"
                               "    sub layers: [Layer<N>; L]\n"
                               "    pub fn forward(x: Tensor[N; f32]) -> Tensor[N; f32] {\n"
                               "        var h = x\n"
                               "        static for layer in layers {\n"
                               "            h = layer.forward(h)\n"
                               "        }\n"
                               "        let (a, b) = (h, x)\n"
                               "        return a + b\n"
                               "    }\n"
                               "}\n";
    const std::string text = emitted(source);
    CHECK(contains(text, "sub layers: [Layer<N>; L]"));
    CHECK(contains(text, "var h = x\n"));
    CHECK(contains(text, "static for layer in layers {\n"));
    CHECK(contains(text, "h = layer.forward(h)\n"));
    CHECK(contains(text, "let (a, b) = (h, x)"));
    check_round_trip(source);
}

TEST("emit: values used inside a region keep their own binding") {
    const std::string source = "module m\n"
                               "pub fn f<N: Dim>() -> Tensor[N; i64] {\n"
                               "    let positions = iota<i64>(N)\n"
                               "    let y[n] = positions[n] + 1\n"
                               "    return y\n"
                               "}\n";
    const std::string text = emitted(source);
    CHECK(contains(text, "let positions = iota<i64>(N)\n"));
    CHECK(contains(text, "let y[n] = (positions[n] + 1)"));
    check_round_trip(source);
}

TEST("emit: optimized modules emit too") {
    const std::string source =
        "module m\n"
        "pub fn f<B: Dim, H: Dim>(x: Tensor[B, H; f32]) -> Tensor[B, H; f32] {\n"
        "    let same = reshape(x, [B, H])\n"
        "    let twice = same + same\n"
        "    return cast<f32>(twice)\n"
        "}\n";
    const std::string text = emitted(source, true);
    CHECK(contains(text, "return (x + x)"));
    CHECK(checks(text));
}
