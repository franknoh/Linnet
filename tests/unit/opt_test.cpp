#include "linnet/ir/lower.hpp"
#include "linnet/opt/passes.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

std::string optimized(const std::string& source) {
    SourceManager sources;
    DiagnosticSink sink;
    const FileId file = sources.add_file("m.linnet", source).value();
    const ast::Ast tree = parse(sources, file, sink);
    const ast::Ast* const modules[] = {&tree};
    const sema::AnalysisResult analysis = sema::analyze(sources, modules, sink);
    CHECK(!sink.has_errors());
    ir::Module module = ir::lower(sources, modules, analysis.model);
    opt::run_pipeline(module, opt::canonical_passes());
    return ir::print(module);
}

std::size_t count(const std::string& text, const std::string& needle) {
    std::size_t total = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + 1)) {
        ++total;
    }
    return total;
}

} // namespace

TEST("opt: identity views and casts disappear") {
    const std::string text = optimized(
        "module m\nfn f<B: Dim, H: Dim>(x: Tensor[B, H; f32]) -> Tensor[B, H; f32] {\n"
        "    let a = reshape(x, [B, H])\n    let b = cast<f32>(a)\n"
        "    let c = permute(b, [0, 1])\n    let d = broadcast_to(c, [B, H])\n    return d\n}\n");
    CHECK_EQ(count(text, "reshape"), 0U);
    CHECK_EQ(count(text, "cast"), 0U);
    CHECK_EQ(count(text, "permute"), 0U);
    CHECK_EQ(count(text, "broadcast"), 0U);
    CHECK(text.find("return %x0") != std::string::npos);
}

TEST("opt: permutations compose and negations cancel") {
    const std::string text =
        optimized("module m\nfn f<A: Dim, B: Dim, C: Dim>(x: Tensor[A, B, C; f32]) -> Tensor[A, B, "
                  "C; f32] {\n"
                  "    let p = permute(permute(x, [1, 2, 0]), [2, 0, 1])\n    return -(-p)\n}\n");
    CHECK_EQ(count(text, "permute"), 0U);
    CHECK_EQ(count(text, "neg"), 0U);
    const std::string swapped =
        optimized("module m\nfn f<A: Dim, B: Dim, C: Dim>(x: Tensor[A, B, C; f32]) -> Tensor[C, A, "
                  "B; f32] {\n"
                  "    return permute(permute(x, [1, 2, 0]), [1, 2, 0])\n}\n");
    CHECK_EQ(count(swapped, "permute"), 1U);
    CHECK(swapped.find("permute [2, 0, 1]") != std::string::npos);
}

TEST("opt: common subexpressions merge and dead code goes") {
    const std::string text = optimized(
        "module m\nfn f<N: Dim>(x: Tensor[N; f32], y: Tensor[N; f32]) -> Tensor[N; f32] {\n"
        "    let a = exp(x) * y\n    let b = exp(x) * y\n    let unused = x + y\n    return a + "
        "b\n}\n");
    CHECK_EQ(count(text, "exp"), 1U);
    CHECK_EQ(count(text, " mul "), 1U);
    CHECK_EQ(count(text, "add"), 1U);
}

TEST("opt: integer constants fold without overflow") {
    const std::string text =
        optimized("module m\nfn f() -> i64 {\n    let a = 6 * 7 + 1\n    return a\n}\n");
    CHECK(text.find("const.int 43") != std::string::npos);
    const std::string huge = optimized(
        "module m\nfn f() -> i64 {\n    let a: i64 = 9223372036854775807\n    return a + 1\n}\n");
    CHECK(huge.find("add") != std::string::npos);
}

TEST("opt: reductions and calls are untouched by CSE across regions") {
    const std::string text = optimized(
        "module m\nfn f<N: Dim>(a: Tensor[N; f32], b: Tensor[N; f32]) -> f32 {\n"
        "    let s = sum[i] a[i] * b[i]\n    let t = sum[i] a[i] * a[i]\n    return s + t\n}\n");
    CHECK_EQ(count(text, "reduce sum"), 2U);
}

TEST("opt: state reads and writes survive DCE and CSE in order") {
    const std::string text =
        optimized("module m\nblock B<N: Dim> {\n    state s: Tensor[N; f32]\n"
                  "    pub entry step(x: Tensor[N; f32]) -> Tensor[N; f32] {\n"
                  "        let before = s\n        s = x\n        let after = s\n"
                  "        s = after + before\n        return after - before\n    }\n}\n");
    CHECK_EQ(count(text, "state.write"), 2U);
    CHECK_EQ(count(text, "state.read"), 2U); // the two reads are not merged
    CHECK(text.find("state.read") < text.find("state.write"));
}
