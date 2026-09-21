#include "linnet/sema/analysis.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <string>
#include <vector>

using namespace linnet;

namespace {

struct Checked {
    SourceManager sources;
    DiagnosticSink sink;
    sema::AnalysisResult result;

    explicit Checked(const std::vector<std::string>& files) {
        std::vector<ast::Ast> trees;
        for (std::size_t i = 0; i < files.size(); ++i) {
            const FileId file =
                sources.add_file("m" + std::to_string(i) + ".linnet", files[i]).value();
            trees.push_back(parse(sources, file, sink));
        }
        CHECK(!sink.has_errors());
        std::vector<const ast::Ast*> modules;
        modules.reserve(trees.size());
        for (const ast::Ast& tree : trees) {
            modules.push_back(&tree);
        }
        result = sema::analyze(sources, modules, sink);
    }

    // Error codes only; lints are covered by their own test.
    std::string codes() const { return codes_of(Severity::Error); }
    std::string warnings() const { return codes_of(Severity::Warning); }

    std::string codes_of(Severity severity) const {
        std::string out;
        for (const Diagnostic& diagnostic : sink.diagnostics()) {
            if (diagnostic.severity == severity) {
                out += (out.empty() ? "" : " ") + diagnostic.code;
            }
        }
        return out;
    }

    std::string messages() const {
        std::string out;
        for (const Diagnostic& diagnostic : sink.diagnostics()) {
            if (diagnostic.severity == Severity::Error) {
                out += diagnostic.code + ": " + diagnostic.message + "\n";
            }
        }
        return out;
    }

    // Type of the last binding with this name.
    std::string type_of(std::string_view name) const {
        std::string type = "<unbound>";
        for (const sema::BindingInfo& binding : result.bindings) {
            if (binding.name == name) {
                type = binding.type;
            }
        }
        return type;
    }
};

Checked check(const std::string& body) {
    return Checked({"module m\n" + body});
}

// Codes reported for a function body with some common parameters in scope.
std::string body_codes(const std::string& statements) {
    const Checked checked =
        check("fn f<B: Dim, S: Dim, H: Dim, *P: Shape, T: Float>(x: Tensor[B, S, H; f32], "
              "y: Tensor[B, S, H; f32], h: Tensor[H; f32], p: Tensor[*P, H; T], s: f32, n: i32, "
              "flag: bool, o: Tensor[H; f32]?, m: Tensor[B, S; bool]) {\n" +
              statements + "\n}\n");
    return checked.codes();
}

std::string binding_type(const std::string& statements, std::string_view name) {
    const Checked checked =
        check("fn f<B: Dim, S: Dim, H: Dim, *P: Shape, T: Float>(x: Tensor[B, S, H; f32], "
              "y: Tensor[B, S, H; f32], h: Tensor[H; f32], p: Tensor[*P, H; T], s: f32, n: i32, "
              "flag: bool, o: Tensor[H; f32]?, m: Tensor[B, S; bool]) {\n" +
              statements + "\n}\n");
    CHECK_EQ(checked.messages(), "");
    return checked.type_of(name);
}

} // namespace

TEST("sema: literals adopt the dtype of their context") {
    CHECK_EQ(binding_type("let a = 1", "a"), "i64");
    CHECK_EQ(binding_type("let a = 1.5", "a"), "f64");
    CHECK_EQ(binding_type("let a: f32 = 1", "a"), "f32");
    CHECK_EQ(binding_type("let a = x * 0.5 + 1", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = s * 2", "a"), "f32");
    CHECK_EQ(binding_type("let a = (1, 2.0, true)", "a"), "(i64, f64, bool)");
    CHECK_EQ(binding_type("let a = -(1 + 2) * 3", "a"), "i64");
    CHECK_EQ(binding_type("let a = H / 2 + 1", "a"), "dimension H / 2 + 1");
    CHECK_EQ(body_codes("let a: i8 = 200"), "E2105");
    CHECK_EQ(body_codes("let a: u8 = -1"), "E2105");
    CHECK_EQ(body_codes("let a: i32 = 1.5"), "E2101");
    CHECK_EQ(body_codes("let a: f16 = 1e10"), "E2105");
    CHECK_EQ(body_codes("let a = n * 1.5"), "E2104");
    CHECK_EQ(body_codes("let a: bool = 1"), "E2101");
    CHECK_EQ(body_codes("let a = none"), "E2113");
    CHECK_EQ(body_codes("let a = \"text\""), "E2104");
    CHECK_EQ(body_codes("let a = 99999999999999999999"), "E2105");
}

TEST("sema: arithmetic requires identical dtypes and provable broadcasting") {
    CHECK_EQ(binding_type("let a = x + h", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = x * s", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = x < y", "a"), "Tensor[B, S, H; bool]");
    CHECK_EQ(binding_type("let a = p + p", "a"), "Tensor[*P, H; T]");
    CHECK_EQ(binding_type("let a = -x", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = n == 1 && !flag", "a"), "bool");
    CHECK_EQ(body_codes("let a = x + n"), "E2103");
    CHECK_EQ(body_codes("let a = s + n"), "E2103");
    CHECK_EQ(body_codes("let a = x + m"), "E2103");
    CHECK_EQ(body_codes("let a = p + x"), "E2103");
    CHECK_EQ(body_codes("let a = m + m"), "E2104");
    CHECK_EQ(body_codes("let a = m && m"), "E2116");
    CHECK_EQ(body_codes("let a = !x"), "E2116");
    CHECK_EQ(body_codes("let a = -flag"), "E2104");
    CHECK_EQ(body_codes("let a = o + h"), "E2104");
}

TEST("sema: broadcasting with one-sized and unproven dimensions") {
    const char* source = "fn f<A: Dim, B: Dim>(a: Tensor[A, 1; f32], b: Tensor[1, B; f32], "
                         "c: Tensor[B; f32]) {\n    let r = a + b\n    let q = a + c\n}\n";
    const Checked checked = check(source);
    CHECK_EQ(checked.messages(), "");
    CHECK_EQ(checked.type_of("r"), "Tensor[A, B; f32]");
    CHECK_EQ(checked.type_of("q"), "Tensor[A, B; f32]");

    CHECK_EQ(check("fn f<A: Dim, B: Dim>(a: Tensor[A; f32], b: Tensor[B; f32]) where A == B {\n"
                   "    let r = a + b\n}\n")
                 .codes(),
             "");
    CHECK_EQ(check("fn f<A: Dim, *P: Shape>(a: Tensor[A; f32], b: Tensor[*P; f32]) {\n"
                   "    let r = a + b\n}\n")
                 .codes(),
             "E2207");
}

TEST("sema: names, mutability, and scopes") {
    CHECK_EQ(body_codes("let a = missing"), "E1201");
    CHECK_EQ(body_codes("let a = 1\na = 2"), "E2108");
    CHECK_EQ(body_codes("s = 2.0"), "E2108");
    CHECK_EQ(body_codes("var a = 1\na = 2\na = a + 1"), "");
    CHECK_EQ(body_codes("var a = 1\na = 2.5"), "E2109");
    CHECK_EQ(body_codes("var a = x\na = h"), "E2109");
    CHECK_EQ(body_codes("let a = 1\nlet a = x"), "");
    CHECK_EQ(body_codes("let exp = 1"), "E1207");
    CHECK_EQ(body_codes("let f32 = 1"), "E1207");
    CHECK_EQ(body_codes("let a = f"), "E1208");
    CHECK_EQ(body_codes("let (a, b) = (1, x)\nlet c = b + 1.0"), "");
    CHECK_EQ(body_codes("let (a, b) = x"), "E2117");
    CHECK_EQ(body_codes("let some(a) = o"), "E2117");
    CHECK_EQ(check("fn f() { return }\nfn f() { return }\n").codes(), "E1204");
    CHECK_EQ(check("fn f(a: f32, a: f32) { return }\n").codes(), "E1205");
    CHECK_EQ(check("fn f<N: Dim, N: Dim>() { return }\n").codes(), "E1205");
    CHECK_EQ(check("fn cast() { return }\n").codes(), "E1207");
}

TEST("sema: returns") {
    CHECK_EQ(check("fn f() -> f32 { return 1.0 }\n").codes(), "");
    CHECK_EQ(check("fn f() -> f32 { let a = 1.0 }\n").codes(), "E2120");
    CHECK_EQ(check("fn f() -> f32 { return }\n").codes(), "E2101");
    CHECK_EQ(check("fn f() { return 1 }\n").codes(), "E2101");
    CHECK_EQ(check("fn f() -> i32 { return 1.5 }\n").codes(), "E2101");
    CHECK_EQ(
        check("fn f<A: Dim, B: Dim>(x: Tensor[A; f32]) -> Tensor[B; f32] { return x }\n").codes(),
        "E2202");
    CHECK_EQ(check("fn f<A: Dim>(x: Tensor[A; f32]) -> Tensor[A; f16] { return x }\n").codes(),
             "E2103");
    CHECK_EQ(check("fn f(o: f32?) -> f32? { return none }\n").codes(), "");
    CHECK_EQ(check("fn f() -> f32? { return some(1.0) }\n").codes(), "");
    CHECK_EQ(check("fn f() -> (f32, i8) { return (1, 2) }\n").codes(), "");
}

TEST("sema: if and match") {
    CHECK_EQ(binding_type("let a = if flag { x } else { y }", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = if flag { 1.0 } else { s }", "a"), "f32");
    CHECK_EQ(binding_type("let a = if flag { 1 } else { 2 }", "a"), "i64");
    CHECK_EQ(binding_type("let a = match o { some(v) => v + h none => h }", "a"), "Tensor[H; f32]");
    CHECK_EQ(binding_type("let a = match o { some(v) => v other => h }", "a"), "Tensor[H; f32]");
    CHECK_EQ(body_codes("let a = if n { x } else { y }"), "E2114");
    CHECK_EQ(body_codes("let a = if flag { x } else { h }"), "E2115");
    CHECK_EQ(body_codes("let a = match o { none => h }"), "E2118");
    CHECK_EQ(body_codes("let a = match x { none => h }"), "E2117 E2117");
    CHECK_EQ(body_codes("let a = match o { v => h none => h }"), "E2117");

    const char* enums = "enum Mask { None, Causal, Window }\n"
                        "fn f(k: Mask, s: f32) -> f32 {\n"
                        "    let same = k == Mask.Causal\n"
                        "    return match k { None => s Causal => s * 2.0 Window => 0.0 }\n}\n";
    CHECK_EQ(check(enums).messages(), "");
    CHECK_EQ(check("enum Mask { None, Causal }\nfn f(k: Mask) -> f32 { return match k { None => "
                   "1.0 } }\n")
                 .codes(),
             "E2118");
    CHECK_EQ(check("enum Mask { None }\nfn f() -> Mask { return Mask.Other }\n").codes(), "E2107");
}

TEST("sema: tensor comprehensions and reductions") {
    CHECK_EQ(binding_type("let c[b, s] = sum[k] x[b, s, k] * y[b, s, k]", "c"),
             "Tensor[B, S; f32]");
    CHECK_EQ(binding_type("let c[k, b] = x[b, 0, k]", "c"), "Tensor[H, B; f32]");
    CHECK_EQ(binding_type("let c[*q, i, j] = p[*q, i] * p[*q, j]", "c"), "Tensor[*P, H, H; T]");
    CHECK_EQ(binding_type("let t = sum[b, s, k] x[b, s, k]", "t"), "f32");
    CHECK_EQ(binding_type("let t = max[k] h[k] * s", "t"), "f32");
    CHECK_EQ(binding_type("let t = any[b, s] m[b, s]", "t"), "bool");
    CHECK_EQ(binding_type("let c[b] = sum<f32>[s, k] x[b, s, k] + 1.0", "c"), "Tensor[B; f32]");
    CHECK_EQ(binding_type("let c[b, s] = sum[k] x[b, s, k] * (sum[j] h[j])", "c"),
             "Tensor[B, S; f32]");

    CHECK_EQ(body_codes("let c[b] = x[b, s0, k]"), "E3101 E3101");
    CHECK_EQ(body_codes("let c[b] = sum[k] x[b, k]"), "E2204");
    CHECK_EQ(body_codes("let c[b] = sum[k] x[b, b, k]"), "E2201");
    CHECK_EQ(body_codes("let c[b, z] = sum[s, k] x[b, s, k]"), "E3103");
    CHECK_EQ(body_codes("let c[b] = sum[s, k, j] x[b, s, k]"), "E3102");
    CHECK_EQ(body_codes("let c[b, b] = sum[s, k] x[b, s, k]"), "E3105");
    CHECK_EQ(body_codes("let c[k] = h[k] * x"), "E3104");
    CHECK_EQ(body_codes("let c[k] = 1.0"), "E3104");
    CHECK_EQ(body_codes("let c[k] = h[k] * k"), "E3104");
    CHECK_EQ(body_codes("let c[q] = p[q, q]"), "E3106");
    CHECK_EQ(body_codes("let c[k] = p[*r, k]"), "E3101");
    CHECK_EQ(body_codes("let t = sum<f16>[k] h[k]"), "E2103");
    CHECK_EQ(body_codes("let t = sum[b, s] m[b, s]"), "E2104");
    CHECK_EQ(body_codes("let t = all[k] h[k]"), "E2104");
}

TEST("sema: pack indices") {
    const Checked checked = check(
        "fn f<*P: Shape, I: Dim, O: Dim, T: Float>(x: Tensor[*P, I; T], w: Tensor[O, I; T]) "
        "-> Tensor[*P, O; T] {\n    let y[*s, o] = sum[i] x[*s, i] * w[o, i]\n    return y\n}\n");
    CHECK_EQ(checked.messages(), "");
    CHECK_EQ(checked.type_of("y"), "Tensor[*P, O; T]");
}

TEST("sema: slicing") {
    CHECK_EQ(binding_type("let a = x[0]", "a"), "Tensor[S, H; f32]");
    CHECK_EQ(binding_type("let a = x[0, 1, 2]", "a"), "f32");
    CHECK_EQ(binding_type("let a = x[:, 1:3]", "a"), "Tensor[B, max(0, min(3, S) - 1), H; f32]");
    CHECK_EQ(binding_type("let a = x[..., 0]", "a"), "Tensor[B, S; f32]");
    CHECK_EQ(binding_type("let a = p[..., 0:1]", "a"), "Tensor[*P, min(1, H); T]");
    CHECK_EQ(binding_type("let a = x[..., ::2]", "a"), "Tensor[B, S, (H + 1) / 2; f32]");
    CHECK_EQ(binding_type("let a = x[n]", "a"), "Tensor[S, H; f32]");
    CHECK_EQ(body_codes("let a = x[0, 0, 0, 0]"), "E2204");
    CHECK_EQ(body_codes("let a = x[..., 0, ...]"), "E2204");
    CHECK_EQ(body_codes("let a = p[0]"), "E3106");
    CHECK_EQ(body_codes("let a = x[-1]"), "E2209");
    CHECK_EQ(body_codes("let a = x[::0]"), "E2209");
    CHECK_EQ(body_codes("let a = x[n:]"), "E2209");
    CHECK_EQ(body_codes("let a = x[s]"), "E3104");
    CHECK_EQ(body_codes("let a = s[0]"), "E2104");

    CHECK_EQ(check("fn f(x: Tensor[4; f32]) -> f32 { return x[4] }\n").codes(), "E2209");
    CHECK_EQ(
        check("fn f(x: Tensor[4, 6; f32]) -> Tensor[2, 3; f32] { return x[1:3, ::2] }\n").codes(),
        "");
}

TEST("sema: calls infer generics and check constraints") {
    const char* prelude =
        "op linear<*S: Shape, In: Dim, Out: Dim, T: Float>(x: Tensor[*S, In; T], "
        "w: Tensor[Out, In; T], bias: Tensor[Out; T]? = none) -> Tensor[*S, Out; T] {\n"
        "    let y[*s, o] = sum[i] x[*s, i] * w[o, i]\n"
        "    return match bias { some(b) => y + b none => y }\n}\n"
        "fn split<H: Dim, N: Dim>(x: Tensor[H; f32]) -> Tensor[N, H / N; f32]\n"
        "where H % N == 0, N > 0 {\n    return reshape(x, [N, H / N])\n}\n";
    const auto with = [&](const std::string& rest) { return check(prelude + rest); };

    const Checked good =
        with("fn g<B: Dim>(x: Tensor[B, 7, 4096; bf16], w: Tensor[8192, 4096; bf16], "
             "b: Tensor[8192; bf16]) {\n"
             "    let a = linear(x, w)\n    let c = linear(x, w, some(b))\n"
             "    let d = linear(x, bias = none, w = w)\n"
             "    let e = split<4096, 32>(fill<f32>([4096], 0.0))\n}\n");
    CHECK_EQ(good.messages(), "");
    CHECK_EQ(good.type_of("a"), "Tensor[B, 7, 8192; bf16]");
    CHECK_EQ(good.type_of("e"), "Tensor[32, 128; f32]");

    CHECK_EQ(
        with("fn g(x: Tensor[3, 5; f32], w: Tensor[8, 6; f32]) { let a = linear(x, w) }\n").codes(),
        "E2202");
    CHECK_EQ(
        with("fn g(x: Tensor[3, 5; f32], w: Tensor[8, 5; f16]) { let a = linear(x, w) }\n").codes(),
        "E2103");
    CHECK_EQ(
        with("fn g(x: Tensor[3, 5; i32], w: Tensor[8, 5; i32]) { let a = linear(x, w) }\n").codes(),
        "E2112");
    CHECK_EQ(with("fn g(x: Tensor[3, 5; f32], w: Tensor[8, 5; f32], b: Tensor[8; f32]) { let a = "
                  "linear(x, w, b) }\n")
                 .codes(),
             "E2101");
    CHECK_EQ(with("fn g(x: Tensor[3, 5; f32]) { let a = linear(x) }\n").codes(), "E2102");
    CHECK_EQ(with("fn g(x: Tensor[3, 5; f32]) { let a = linear(x, x, none, x) }\n").codes(),
             "E2102");
    CHECK_EQ(with("fn g(x: Tensor[3, 5; f32]) { let a = linear(x, x, extra = x) }\n").codes(),
             "E2102");
    CHECK_EQ(with("fn g(x: Tensor[10; f32]) { let a = split(x) }\n").codes(), "E2110");
    CHECK_EQ(with("fn g(x: Tensor[10; f32]) { let a = split<10, 3>(x) }\n").codes(), "E2206");
    CHECK_EQ(with("fn g<H: Dim, N: Dim>(x: Tensor[H; f32]) { let a = split<H, N>(x) }\n").codes(),
             "E2206");
    CHECK_EQ(with("fn g<H: Dim, N: Dim>(x: Tensor[H; f32]) where H % N == 0, N > 0 { let a = "
                  "split<H, N>(x) }\n")
                 .codes(),
             "");
    CHECK_EQ(with("fn g(x: f32) { let a = x(1) }\n").codes(), "E2106");
    CHECK_EQ(with("fn g() { let a = nothing(1) }\n").codes(), "E1201");
    CHECK_EQ(with("fn g() { let a = split<1, 2, 3>(1) }\n").codes(), "E2111");
}

TEST("sema: scalar generics and literal arguments") {
    const char* source = "fn scale<T: Float>(x: T, k: T) -> T { return x * k }\n"
                         "fn g(a: f32, b: f16) {\n"
                         "    let r = scale(a, 2.0)\n    let q = scale(b, 1)\n}\n";
    const Checked checked = check(source);
    CHECK_EQ(checked.messages(), "");
    CHECK_EQ(checked.type_of("r"), "f32");
    CHECK_EQ(checked.type_of("q"), "f16");
    CHECK_EQ(
        check("fn id<T: Integer>(x: T) -> T { return x }\nfn g() { let a = id(1.5) }\n").codes(),
        "E2110");
    CHECK_EQ(check("fn half(x: i8) -> i8 { return x }\nfn g() { let a = half(300) }\n").codes(),
             "E2105");
}

TEST("sema: recursion is rejected, directly and indirectly") {
    CHECK_EQ(check("fn a(x: i32) -> i32 { return a(x) }\n").codes(), "E2119");
    CHECK_EQ(check("fn a(x: i32) -> i32 { return b(x) }\nfn b(x: i32) -> i32 { return c(x) }\n"
                   "fn c(x: i32) -> i32 { return a(x) }\nfn d(x: i32) -> i32 { return a(x) }\n")
                 .codes(),
             "E2119");
    CHECK_EQ(
        check("fn a(x: i32) -> i32 { return b(b(x)) }\nfn b(x: i32) -> i32 { return x }\n").codes(),
        "");
}

TEST("sema: prelude functions") {
    CHECK_EQ(binding_type("let a = cast<f16>(x)", "a"), "Tensor[B, S, H; f16]");
    CHECK_EQ(binding_type("let a = cast<T>(h)", "a"), "Tensor[H; T]");
    CHECK_EQ(binding_type("let a = cast<f32>(H) * s", "a"), "f32");
    CHECK_EQ(binding_type("let a = exp(x) + sqrt(y)", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = abs(n)", "a"), "i32");
    CHECK_EQ(binding_type("let a = max(x, 0.0)", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = min(H, 8)", "a"), "dimension min(8, H)");
    CHECK_EQ(binding_type("let a = select(m, x[..., 0], 0.0)", "a"), "Tensor[B, S; f32]");
    CHECK_EQ(binding_type("let a = select(flag, s, 1.0)", "a"), "f32");
    CHECK_EQ(binding_type("let a = reshape(x, [B * S, H])", "a"), "Tensor[B * S, H; f32]");
    CHECK_EQ(binding_type("let a = permute(x, [2, 0, 1])", "a"), "Tensor[H, B, S; f32]");
    CHECK_EQ(binding_type("let a = broadcast_to(h, [B, S, H])", "a"), "Tensor[B, S, H; f32]");
    CHECK_EQ(binding_type("let a = concat(x, y, x, axis = 1)", "a"), "Tensor[B, 3 * S, H; f32]");
    CHECK_EQ(binding_type("let a = concat(p, p, axis = -1)", "a"), "Tensor[*P, 2 * H; T]");
    CHECK_EQ(binding_type("let a = iota(H)", "a"), "Tensor[H; i64]");
    CHECK_EQ(binding_type("let a = iota<f32>(4)", "a"), "Tensor[4; f32]");
    CHECK_EQ(binding_type("let a = fill<f32>([B, 2], 0)", "a"), "Tensor[B, 2; f32]");
    CHECK_EQ(binding_type("let a = fill([H], s)", "a"), "Tensor[H; f32]");

    CHECK_EQ(body_codes("let a = cast(x)"), "E2110");
    CHECK_EQ(body_codes("let a = cast<H>(x)"), "E1208");
    CHECK_EQ(body_codes("let a = exp(n)"), "E2104");
    CHECK_EQ(body_codes("let a = exp(x, y)"), "E2102");
    CHECK_EQ(body_codes("let a = reshape(x, [B, H])"), "E2203");
    CHECK_EQ(body_codes("let a = reshape(x, y)"), "E2101");
    CHECK_EQ(body_codes("let a = permute(x, [0, 0, 1])"), "E2210");
    CHECK_EQ(body_codes("let a = permute(p, [1, 0])"), "E2210");
    CHECK_EQ(body_codes("let a = broadcast_to(x, [H])"), "E2207");
    CHECK_EQ(body_codes("let a = concat(x, h, axis = 0)"), "E2202");
    CHECK_EQ(body_codes("let a = concat(p, p, axis = 0)"), "E2210");
    CHECK_EQ(body_codes("let a = concat(x, y)"), "E2102");
    CHECK_EQ(body_codes("let a = select(x, x, y)"), "E2114");
    CHECK_EQ(body_codes("let a = fill([H], 1.0)"), "E2110");
    CHECK_EQ(body_codes("let a = gather(x, n)"), "E2190");
}

TEST("sema: dimension expressions") {
    CHECK_EQ(check("const Hidden = 4096\nconst Heads = 32\nconst HeadDim = Hidden / Heads\n"
                   "fn f(x: Tensor[Hidden; f32]) -> Tensor[Heads, HeadDim; f32] {\n"
                   "    return reshape(x, [Heads, HeadDim])\n}\n")
                 .messages(),
             "");
    CHECK_EQ(check("fn f<H: Dim, N: Dim>(x: Tensor[H / N; f32]) { return }\n").codes(), "E2205");
    CHECK_EQ(check("fn f<H: Dim>(x: Tensor[H - H - 1; f32]) { return }\n").codes(), "E2208");
    CHECK_EQ(check("fn f(x: Tensor[1.5; f32]) { return }\n").codes(), "E2123");
    CHECK_EQ(check("fn f(n: i32, x: Tensor[n; f32]) { return }\n").codes(), "E1201");
    CHECK_EQ(check("const A = B\nconst B = A\n").codes(), "E2122");
    CHECK_EQ(check("const Eps = 1e-5\nfn f(x: f32) -> f32 { return x + Eps }\n").codes(), "");
    CHECK_EQ(check("const Eps: f32 = 1e-5\nfn f(x: f16) -> f16 { return x + Eps }\n").codes(),
             "E2103");
}

TEST("sema: blocks, members, and methods") {
    const char* model =
        "block Linear<In: Dim, Out: Dim, T: Float = bf16> {\n"
        "    param weight: Tensor[Out, In; T]\n"
        "    param bias: Tensor[Out; T]? = none\n"
        "    pub fn forward<*S: Shape>(x: Tensor[*S, In; T]) -> Tensor[*S, Out; T] {\n"
        "        let y[*s, o] = sum[i] x[*s, i] * weight[o, i]\n"
        "        return match bias { some(b) => y + b none => y }\n    }\n}\n"
        "block Mlp<H: Dim, Layers: Dim> {\n"
        "    sub up: Linear<H, 4 * H>\n"
        "    sub down: Linear<4 * H, H>\n"
        "    sub stack: [Linear<H, H>; Layers]\n"
        "    buffer scale: Tensor[H; bf16]\n"
        "    pub entry run<B: Dim>(x0: Tensor[B, H; bf16]) -> Tensor[B, H; bf16] {\n"
        "        var x = down.forward(max(up.forward(x0), 0.0)) * scale\n"
        "        static for layer in stack {\n            x = layer.forward(x)\n        }\n"
        "        let first = stack[0].forward(x)\n"
        "        let w = up.weight\n"
        "        return x\n    }\n}\n";
    const Checked checked = check(model);
    CHECK_EQ(checked.messages(), "");
    CHECK_EQ(checked.type_of("x"), "Tensor[B, H; bf16]");
    CHECK_EQ(checked.type_of("w"), "Tensor[4 * H, H; bf16]");
    CHECK_EQ(checked.type_of("layer"), "Linear<H, H, bf16>");

    CHECK_EQ(check("block B { param w: f32 }\n").codes(), "E4101");
    CHECK_EQ(check("block B { param w: Tensor[4; f32] = none }\n").codes(), "E4101");
    CHECK_EQ(check("block B { param w: Tensor[4; f32]? = fill<f32>([4], 0.0) }\n").codes(),
             "E4102");
    CHECK_EQ(check("block B { sub s: Tensor[4; f32] }\n").codes(), "E4101");
    CHECK_EQ(check("block B { param w: Tensor[4; f32]\n param w: Tensor[4; f32] }\n").codes(),
             "E1205");
    CHECK_EQ(check("block A { sub b: B }\nblock B { }\nblock C { sub a: A\n fn f() { let z = "
                   "a.missing } }\n")
                 .codes(),
             "E2107");
    CHECK_EQ(check("block A<N: Dim> { }\nblock C { sub a: A }\n").codes(), "E2111");
    CHECK_EQ(check("block A<N: Dim> { }\nblock C { sub a: A<f32> }\n").codes(), "E2111");
    CHECK_EQ(check("block C { fn f(xs: [f32; 3]) { static for x in 3 { } } }\n").codes(), "E2121");
}

TEST("sema: structs and aliases") {
    const char* source = "struct Pair<T: DType> {\n    left: T\n    right: (T, T)\n}\n"
                         "type Hidden<B: Dim, T: Float = bf16> = Tensor[B, 64; T]\n"
                         "fn f<B: Dim>(p: Pair<f32>, h: Hidden<B>, g: Hidden<B, f32>) {\n"
                         "    let a = p.left\n    let (b, c) = p.right\n    let d = h\n"
                         "    let e = g\n}\n";
    const Checked checked = check(source);
    CHECK_EQ(checked.messages(), "");
    CHECK_EQ(checked.type_of("a"), "f32");
    CHECK_EQ(checked.type_of("c"), "f32");
    CHECK_EQ(checked.type_of("d"), "Tensor[B, 64; bf16]");
    CHECK_EQ(checked.type_of("e"), "Tensor[B, 64; f32]");
    CHECK_EQ(check("struct S { a: f32 }\nfn f(s: S) { let b = s.z }\n").codes(), "E2107");
    CHECK_EQ(check("struct S { a: S }\n").codes(), "E2122");
    CHECK_EQ(check("type A = B\ntype B = A\n").codes(), "E2122");
    CHECK_EQ(check("fn f(x: Missing) { return }\n").codes(), "E1201");
    CHECK_EQ(check("const N = 4\nfn f(x: N) { return }\n").codes(), "E1208");
    CHECK_EQ(check("fn f(x: f32<i32>) { return }\n").codes(), "E2111");
    CHECK_EQ(check("fn f<N: Shape>() { return }\n").codes(), "E2111");
    CHECK_EQ(check("fn f<T: Float = i32>(x: T) { return }\n").codes(), "E2112");
}

TEST("sema: imports across modules") {
    const std::string library = "module std.nn.ops\n"
                                "pub op double<*S: Shape, T: Numeric>(x: Tensor[*S; T]) -> "
                                "Tensor[*S; T] {\n    return x + x\n}\n"
                                "fn hidden(x: f32) -> f32 { return x }\n"
                                "pub const Width = 8\n";
    const auto with = [&](const std::string& user) { return Checked({library, user}); };

    CHECK_EQ(with("module app\nuse std.nn.ops::{double, Width as W}\n"
                  "fn f(x: Tensor[W; f32]) -> Tensor[8; f32] { return double(x) }\n")
                 .messages(),
             "");
    CHECK_EQ(with("module app\nuse std.nn.ops\n"
                  "fn f(x: Tensor[ops.Width; f32]) -> Tensor[8; f32] { return ops.double(x) }\n")
                 .codes(),
             "E2123");
    CHECK_EQ(with("module app\nuse std.nn.ops\n"
                  "fn f(x: Tensor[8; f32]) -> Tensor[8; f32] { return ops.double(x) }\n")
                 .messages(),
             "");
    CHECK_EQ(with("module app\nuse std.nn.ops::hidden\n").codes(), "E1203");
    CHECK_EQ(with("module app\nuse std.nn.ops::missing\n").codes(), "E1201");
    CHECK_EQ(with("module app\nuse std.nn.other::double\n"
                  "fn f(x: f32) -> f32 { return double(x) }\n")
                 .codes(),
             "E1202");
    CHECK_EQ(with("module app\nuse std.nn.ops::double\nfn double() { return }\n").codes(), "E1204");
    CHECK_EQ(
        Checked({"module pkg.layers.a\npub const N = 2\n",
                 "module pkg.main\nuse crate.layers.a::N\nfn f(x: Tensor[N; f32]) { return }\n"})
            .messages(),
        "");
}

TEST("sema: import cycles are rejected") {
    CHECK_EQ(Checked({"module a\nuse b::{Y}\npub const X = 1\n",
                      "module b\nuse a::{X}\npub const Y = 2\n"})
                 .codes(),
             "E1206");
}

TEST("sema: lints") {
    CHECK_EQ(check("fn f(x: f32, unused_parameter: f32) -> f32 {\n    let a = x\n    let b = x\n"
                   "    let _ignored = x\n    return a\n}\n")
                 .warnings(),
             "W1002");
    CHECK_EQ(check("block B {\n    param used: Tensor[4; f32]\n    param spare: Tensor[4; f32]\n"
                   "    fn f() -> Tensor[4; f32] { return used }\n}\n")
                 .warnings(),
             "W1003");
    const std::string library = "module lib\npub const A = 1\npub const B = 2\n";
    CHECK_EQ(Checked({library,
                      "module app\nuse lib::{A, B}\nuse lib\n"
                      "fn f(x: Tensor[A; f32]) { return }\n"})
                 .warnings(),
             "W1001 W1001");
    CHECK_EQ(Checked({library, "module app\nuse lib\nfn f(x: f32) -> f32 { return x * lib.B }\n"})
                 .warnings(),
             "");
    // Lints stay quiet while there are errors to fix first.
    CHECK_EQ(check("fn f(x: f32) -> f32 {\n    let a = x\n    return missing\n}\n").warnings(), "");
}

TEST("sema: one mistake yields one diagnostic") {
    CHECK_EQ(body_codes("let a = missing + 1\nlet b = a * x\nlet c = b[0]"), "E1201");
    CHECK_EQ(body_codes("let a = x + n\nlet b = a + a\nlet c[i] = a[i]"), "E2103");
    CHECK_EQ(check("fn f(x: Missing) -> Missing2 { return x }\nfn g() { let a = f(1) }\n").codes(),
             "E1201 E1201");
}
