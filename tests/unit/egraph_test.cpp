#include "linnet/opt/egraph.hpp"

#include "lowered.hpp"
#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

using test::count;

std::string saturated(const std::string& source, opt::Legality allowed) {
    return test::with_lowered(source, [&](ir::Module& module) {
        opt::run_pipeline(module, opt::optimizing_passes(allowed));
        return ir::print(module);
    });
}

} // namespace

TEST("egraph: composed views collapse to one") {
    const std::string text =
        saturated("module m\n"
                  "pub fn f<B: Dim, H: Dim>(x: Tensor[B, H; f32]) -> Tensor[H, B; f32] {\n"
                  "    let a = reshape(x, [B * H])\n"
                  "    let b = reshape(a, [H, B])\n"
                  "    let c = permute(permute(b, [1, 0]), [1, 0])\n"
                  "    return c\n"
                  "}\n",
                  opt::Legality::Exact);
    CHECK_EQ(count(text, "reshape"), 1U);
    CHECK_EQ(count(text, "permute"), 0U);
}

TEST("egraph: double negation and identities disappear") {
    const std::string text =
        saturated("module m\n"
                  "pub fn f<N: Dim>(x: Tensor[N; f32], y: Tensor[N; f32]) -> Tensor[N; f32] {\n"
                  "    let z = -(-x) * 1.0\n"
                  "    return select(y > 0.0, z, z) + y\n"
                  "}\n",
                  opt::Legality::Exact);
    CHECK_EQ(count(text, "neg"), 0U);
    CHECK_EQ(count(text, "select"), 0U);
    CHECK_EQ(count(text, "= mul"), 0U);
    CHECK_EQ(count(text, "= add"), 1U);
}

TEST("egraph: commutative duplicates merge") {
    const std::string text =
        saturated("module m\n"
                  "pub fn f<N: Dim>(x: Tensor[N; f32], y: Tensor[N; f32]) -> Tensor[N; f32] {\n"
                  "    let a = x * y\n"
                  "    let b = y * x\n"
                  "    return a + b\n"
                  "}\n",
                  opt::Legality::Exact);
    CHECK_EQ(count(text, "= mul"), 1U);
}

TEST("egraph: factoring needs the numerically-equivalent policy") {
    const std::string source = "module m\n"
                               "pub fn f<N: Dim>(a: Tensor[N; f32], b: Tensor[N; f32], c: "
                               "Tensor[N; f32]) -> Tensor[N; f32] {\n"
                               "    return a * c + b * c\n"
                               "}\n";
    CHECK_EQ(count(saturated(source, opt::Legality::Exact), "= mul"), 2U);
    const std::string factored = saturated(source, opt::Legality::NumericallyEquivalent);
    CHECK_EQ(count(factored, "= mul"), 1U);
    CHECK_EQ(count(factored, "= add"), 1U);
}

TEST("egraph: additive identity is not exact") {
    const std::string source = "module m\n"
                               "pub fn f<N: Dim>(x: Tensor[N; f32]) -> Tensor[N; f32] {\n"
                               "    return x + 0.0\n"
                               "}\n";
    CHECK_EQ(count(saturated(source, opt::Legality::Exact), "= add"), 1U);
    CHECK_EQ(count(saturated(source, opt::Legality::IEEEEquivalent), "= add"), 0U);
}

TEST("egraph: rules declare their legality") {
    bool has_exact = false;
    bool has_numeric = false;
    for (const opt::Rewrite& rewrite : opt::rewrites()) {
        has_exact = has_exact || rewrite.legality == opt::Legality::Exact;
        has_numeric = has_numeric || rewrite.legality == opt::Legality::NumericallyEquivalent;
    }
    CHECK(has_exact);
    CHECK(has_numeric);
}
