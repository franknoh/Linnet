#include "linnet/ast/dump.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

struct Parsed {
    SourceManager sources;
    DiagnosticSink sink;
    std::string tree;

    explicit Parsed(std::string text) {
        const FileId file = sources.add_file("test.linnet", std::move(text)).value();
        const ast::Ast ast = parse(sources, file, sink);
        tree = ast::dump(ast, sources);
    }

    std::string codes() const {
        std::string out;
        for (const Diagnostic& diagnostic : sink.diagnostics()) {
            out += (out.empty() ? "" : " ") + diagnostic.code;
        }
        return out;
    }

    std::string messages() const {
        std::string out;
        for (const Diagnostic& diagnostic : sink.diagnostics()) {
            out += diagnostic.message + "\n";
        }
        return out;
    }
};

// Dump of `expression` parsed as the value of a `return`, without the wrapper
// lines and their indentation.
std::string expr_tree(const std::string& expression) {
    const Parsed parsed("module m\nfn f() {\n    return " + expression + "\n}\n");
    CHECK_EQ(parsed.codes(), "");
    const std::string marker = "    return\n";
    const std::size_t begin = parsed.tree.find(marker);
    if (begin == std::string::npos) {
        return parsed.tree;
    }
    std::string out;
    std::size_t line_begin = begin + marker.size();
    while (line_begin < parsed.tree.size()) {
        const std::size_t line_end = parsed.tree.find('\n', line_begin);
        out += parsed.tree.substr(line_begin + 6, line_end - line_begin - 6 + 1);
        line_begin = line_end + 1;
    }
    return out;
}

} // namespace

TEST("parser: operator precedence and left associativity") {
    CHECK_EQ(expr_tree("a - b - c * d"),
             "binary -\n"
             "  binary -\n"
             "    name a\n"
             "    name b\n"
             "  binary *\n"
             "    name c\n"
             "    name d\n");
    CHECK_EQ(expr_tree("a || b && !c == d"),
             "binary ||\n"
             "  name a\n"
             "  binary &&\n"
             "    name b\n"
             "    binary ==\n"
             "      unary !\n"
             "        name c\n"
             "      name d\n");
    CHECK_EQ(expr_tree("-x.y(1)[0]"),
             "unary -\n"
             "  index\n"
             "    call\n"
             "      member y\n"
             "        name x\n"
             "      arg\n"
             "        literal 1\n"
             "    literal 0\n");
}

TEST("parser: generic calls versus comparisons") {
    CHECK_EQ(expr_tree("cast<f32>(x)"),
             "call <f32>\n"
             "  name cast\n"
             "  arg\n"
             "    name x\n");
    CHECK_EQ(expr_tree("f<H / N, Tensor[B; T], [L<H>; 2], (A + 1)>(x, axis = 1)"),
             "call <(H / N), Tensor[B; T], [L<H>; 2], (A + 1)>\n"
             "  name f\n"
             "  arg\n"
             "    name x\n"
             "  arg axis\n"
             "    literal 1\n");
    CHECK_EQ(expr_tree("a < b"),
             "binary <\n"
             "  name a\n"
             "  name b\n");
    CHECK_EQ(expr_tree("a < b && c > (d)"),
             "binary &&\n"
             "  binary <\n"
             "    name a\n"
             "    name b\n"
             "  binary >\n"
             "    name c\n"
             "    paren\n"
             "      name d\n");
}

TEST("parser: reductions") {
    CHECK_EQ(expr_tree("x + sum<f32>[d, e] cast<f32>(q[d]) * k[e]"),
             "binary +\n"
             "  name x\n"
             "  sum<f32> [d, e]\n"
             "    binary *\n"
             "      call <f32>\n"
             "        name cast\n"
             "        arg\n"
             "          index\n"
             "            name q\n"
             "            name d\n"
             "      index\n"
             "        name k\n"
             "        name e\n");
    // Reduction names are ordinary identifiers when no index list follows.
    CHECK_EQ(expr_tree("max(a, min) < sum"),
             "binary <\n"
             "  call\n"
             "    name max\n"
             "    arg\n"
             "      name a\n"
             "    arg\n"
             "      name min\n"
             "  name sum\n");
}

TEST("parser: index components") {
    CHECK_EQ(expr_tree("x[:, 1:2, ::2, a:b:c, 3:, ..., *s, i]"),
             "index\n"
             "  name x\n"
             "  slice\n"
             "  slice\n"
             "    start\n"
             "      literal 1\n"
             "    stop\n"
             "      literal 2\n"
             "  slice\n"
             "    step\n"
             "      literal 2\n"
             "  slice\n"
             "    start\n"
             "      name a\n"
             "    stop\n"
             "      name b\n"
             "    step\n"
             "      name c\n"
             "  slice\n"
             "    start\n"
             "      literal 3\n"
             "  ...\n"
             "  pack *s\n"
             "  name i\n");
}

TEST("parser: if, match, tuples, options, shapes") {
    CHECK_EQ(
        expr_tree("if c { (a, b) } else { match o { some((p, q)) => [p, 2 * q] none => none } }"),
        "if\n"
        "  name c\n"
        "  then\n"
        "    tuple\n"
        "      name a\n"
        "      name b\n"
        "  else\n"
        "    match\n"
        "      name o\n"
        "      arm some((p, q))\n"
        "        shape\n"
        "          name p\n"
        "          binary *\n"
        "            literal 2\n"
        "            name q\n"
        "      arm none\n"
        "        none\n");
}

TEST("parser: statements") {
    const Parsed parsed("module m\n"
                        "fn f(x: f32) {\n"
                        "    let a: f32 = 1.0\n"
                        "    let (b, (c, d)) = g(x)\n"
                        "    var v = a\n"
                        "    v = v + b\n"
                        "    static for layer in layers {\n"
                        "        v = layer.forward(v)\n"
                        "    }\n"
                        "    return\n"
                        "}\n");
    CHECK_EQ(parsed.codes(), "");
    CHECK_EQ(parsed.tree,
             "module m\n"
             "fn f\n"
             "  param x: f32\n"
             "  body\n"
             "    let a: f32\n"
             "      literal 1.0\n"
             "    let (b, (c, d))\n"
             "      call\n"
             "        name g\n"
             "        arg\n"
             "          name x\n"
             "    var v\n"
             "      name a\n"
             "    assign v\n"
             "      binary +\n"
             "        name v\n"
             "        name b\n"
             "    static for layer\n"
             "      in\n"
             "        name layers\n"
             "      body\n"
             "        assign v\n"
             "          call\n"
             "            member forward\n"
             "              name layer\n"
             "            arg\n"
             "              name v\n"
             "    return\n");
}

TEST("parser: declarations") {
    const Parsed parsed("module a.b\n"
                        "use std.nn::{linear, rms_norm as norm,}\n"
                        "use crate.layers::Decoder\n"
                        "use dep.ops\n"
                        "pub const Hidden: i64 = 4096\n"
                        "type H<B: Dim, T: Float = bf16, N: Dim = 4> = Tensor[B, Hidden; T]?\n"
                        "struct Pair<T: DType> {\n"
                        "    left: T\n"
                        "    right: (T, T)\n"
                        "}\n"
                        "enum Mask { None, Causal, }\n"
                        "pub block M<L: Dim> {\n"
                        "    param w: Tensor[L; f32]\n"
                        "    param b: Tensor[L; f32]? = none\n"
                        "    buffer mean: Tensor[L; f32]\n"
                        "    sub layers: [Layer<L, f32>; L]\n"
                        "    pub entry run() -> f32 { return 1.0 }\n"
                        "}\n");
    CHECK_EQ(parsed.codes(), "");
    CHECK_EQ(parsed.tree,
             "module a.b\n"
             "use std.nn linear rms_norm as norm\n"
             "use crate.layers Decoder\n"
             "use dep.ops\n"
             "pub const Hidden: i64\n"
             "  literal 4096\n"
             "type H\n"
             "  generic B: Dim\n"
             "  generic T: Float = bf16\n"
             "  generic N: Dim = 4\n"
             "  = Tensor[B, Hidden; T]?\n"
             "struct Pair\n"
             "  generic T: DType\n"
             "  field left: T\n"
             "  field right: (T, T)\n"
             "enum Mask\n"
             "  variant None\n"
             "  variant Causal\n"
             "pub block M\n"
             "  generic L: Dim\n"
             "  param w: Tensor[L; f32]\n"
             "  param b: Tensor[L; f32]? =\n"
             "    none\n"
             "  buffer mean: Tensor[L; f32]\n"
             "  sub layers: [Layer<L, f32>; L]\n"
             "  pub entry run\n"
             "    returns f32\n"
             "    body\n"
             "      return\n"
             "        literal 1.0\n");
}

TEST("parser: missing module declaration") {
    const Parsed parsed("fn f() { return }\n");
    CHECK_EQ(parsed.codes(), "E1102");
    CHECK(parsed.tree.find("fn f") != std::string::npos);
}

TEST("parser: a damaged statement does not disturb its neighbours") {
    const Parsed parsed("module m\n"
                        "fn f(x: f32) -> f32 {\n"
                        "    let a = (x + )\n"
                        "    let b = x\n"
                        "    return b\n"
                        "}\n"
                        "fn g() { return }\n");
    CHECK_EQ(parsed.codes(), "E1101");
    CHECK(parsed.tree.find("let b\n") != std::string::npos);
    CHECK(parsed.tree.find("fn g\n") != std::string::npos);
}

TEST("parser: missing closing brace resumes at the next item") {
    const Parsed parsed("module m\n"
                        "fn f() {\n"
                        "    let a = 1\n"
                        "pub fn g() { return }\n");
    CHECK_EQ(parsed.codes(), "E1101");
    CHECK(parsed.tree.find("pub fn g\n") != std::string::npos);
}

TEST("parser: damaged header keeps later items") {
    const Parsed parsed("module m\n"
                        "fn f(x: , y: f32) -> { return }\n"
                        "fn g() { return }\n");
    CHECK(parsed.sink.error_count() <= 2);
    CHECK(parsed.tree.find("fn g\n") != std::string::npos);
}

TEST("parser: helpful messages") {
    CHECK(Parsed("module m\nfn f() { g(1) }\n").messages().find("expected a statement") !=
          std::string::npos);
    CHECK(Parsed("module m\nfn f() { g(1) }\n")
              .sink.diagnostics()
              .at(0)
              .notes.at(0)
              .find("no expression statements") != std::string::npos);
    CHECK(Parsed("module m\nstruct S { a: f32, b: f32 }\n").messages().find("not commas") !=
          std::string::npos);
    CHECK(Parsed("module m\nfn f<N: Dimension>() { return }\n")
              .messages()
              .find("unknown generic constraint `Dimension`") != std::string::npos);
    CHECK(Parsed("module m\nfn f() { return if a { 1 } else if b { 2 } else { 3 } }\n")
              .messages()
              .find("`else if` is not supported") != std::string::npos);
    CHECK(Parsed("module m\nop f() { return }\n").messages().find("must declare its result") !=
          std::string::npos);
    CHECK(Parsed("module m\nfn f() where N { return }\n").messages().find("must be a comparison") !=
          std::string::npos);
    CHECK(Parsed("module m\nparam w: f32\n").messages().find("only allowed inside a `block`") !=
          std::string::npos);
    CHECK(Parsed("module m\nfn f() { return }\nuse std.nn\n").messages().find("must come before") !=
          std::string::npos);
}

TEST("parser: reserved syntax is rejected, not reinterpreted") {
    CHECK_EQ(Parsed("module m\nfn f() { static for i in 0..N { } return }\n").codes(), "E1103");
    CHECK_EQ(Parsed("module m\nblock B { state cache: f32 }\n").codes(), "E1103");
    CHECK_EQ(Parsed("module m\nextern op f() -> f32\n").codes(), "E1103");
    CHECK_EQ(Parsed("module m\nuse super.x\n").codes(), "E1103");
    CHECK_EQ(Parsed("module m\nfn f() { while true { } return }\n").codes(), "E1103");
    CHECK_EQ(Parsed("module m\nfn f(x: Tensor[; f32]) { return }\n").codes(), "E1103");
    CHECK_EQ(Parsed("module m\nenum E { A(f32) }\n").codes().substr(0, 5), "E1103");
    CHECK_EQ(Parsed("module m\nfn f(mut: f32) { return }\n").codes(), "E1004");
}

TEST("parser: a match arm cannot start with a tuple pattern") {
    const Parsed first("module m\nfn f() { return match o { (c, d) => c none => x } }\n");
    CHECK_EQ(first.codes(), "E1101");
    CHECK(first.messages().find("tuple pattern") != std::string::npos);
    // After another arm the tuple is consumed as a call on that arm's value,
    // so the report is about the `=>`; it is one error either way.
    const Parsed later("module m\nfn f() { return match o { some(a) => a (c, d) => c } }\n");
    CHECK_EQ(later.codes(), "E1101");
    // Tuples inside `some(...)` are still patterns.
    CHECK_EQ(
        Parsed("module m\nfn f() { return match o { some((a, b)) => a none => x } }\n").codes(),
        "");
}

TEST("parser: hostile nesting does not overflow the stack") {
    const std::string cases[] = {
        "module m\nfn f() { return " + std::string(100000, '(') + " }\n",
        "module m\nfn f() { return " + std::string(100000, '[') + " }\n",
        "module m\nfn f() { return a" + std::string(100000, '.') + " }\n",
        "module m\nfn f(x: " + std::string(100000, '(') + ") { return }\n",
        "module m\nfn f() { let " + std::string(100000, '(') + " = 1 }\n",
        "module m\nfn f() { return 1" +
            [] {
                std::string chain;
                for (int i = 0; i < 50000; ++i) {
                    chain += " + 1";
                }
                return chain;
            }() +
            " }\n",
        "module m\n" +
            [] {
                std::string blocks;
                for (int i = 0; i < 50000; ++i) {
                    blocks += "block B { ";
                }
                return blocks;
            }(),
        "module m\nfn f() { return a" +
            [] {
                std::string chain;
                for (int i = 0; i < 50000; ++i) {
                    chain += "<b";
                }
                return chain;
            }() +
            " }\n",
    };
    for (const std::string& text : cases) {
        const Parsed parsed(text);
        CHECK(parsed.sink.has_errors());
        CHECK(parsed.sink.diagnostics().size() < 10);
    }
}
