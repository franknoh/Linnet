#include "linnet/ast/dump.hpp"
#include "linnet/format/formatter.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

struct Formatted {
    std::string text;
    std::string tree;
    std::size_t comments = 0;
};

Formatted format_once(const std::string& source, const format::LayoutOptions& options) {
    SourceManager sources;
    DiagnosticSink sink;
    const FileId file = sources.add_file("test.linnet", source).value();
    const ast::Ast tree = parse(sources, file, sink);
    CHECK(!sink.has_errors());
    return {format::format(tree, sources, options), ast::dump(tree, sources), tree.comments.size()};
}

// Formats `source` and verifies the formatter's invariants: the result parses
// to the same tree, keeps every comment, and is a fixed point.
std::string fmt(const std::string& source, const format::LayoutOptions& options = {}) {
    const Formatted first = format_once(source, options);
    const Formatted second = format_once(first.text, options);
    CHECK_EQ(second.text, first.text);
    CHECK_EQ(second.tree, first.tree);
    CHECK_EQ(second.comments, first.comments);
    return first.text;
}

} // namespace

TEST("format: normalizes spacing, blank lines, and single-line bodies") {
    CHECK_EQ(fmt("module   a.b\nuse std.nn::{linear,rms_norm as norm}\n\n\n\n"
                 "use crate.x::y\nconst A=1\nconst B : i64=2\n\n\nconst C = 3\n"
                 "fn f(x:f32)->f32{return x+1.0}\nfn g(){}\n"),
             "module a.b\n"
             "\n"
             "use std.nn::{linear, rms_norm as norm}\n"
             "\n"
             "use crate.x::y\n"
             "\n"
             "const A = 1\n"
             "const B: i64 = 2\n"
             "\n"
             "const C = 3\n"
             "\n"
             "fn f(x: f32) -> f32 {\n"
             "    return x + 1.0\n"
             "}\n"
             "\n"
             "fn g() {}\n");
}

TEST("format: long signatures break one parameter per line") {
    CHECK_EQ(
        fmt("module m\npub op linear<*S: Shape, In: Dim, Out: Dim, T: Float>(x: Tensor[*S, In; "
            "T], weight: Tensor[Out, In; T], bias: Tensor[Out; T]? = none) -> Tensor[*S, Out; "
            "T] { return x }\n"),
        "module m\n"
        "\n"
        "pub op linear<*S: Shape, In: Dim, Out: Dim, T: Float>(\n"
        "    x: Tensor[*S, In; T],\n"
        "    weight: Tensor[Out, In; T],\n"
        "    bias: Tensor[Out; T]? = none,\n"
        ") -> Tensor[*S, Out; T] {\n"
        "    return x\n"
        "}\n");
}

TEST("format: a trailing comma keeps a list expanded") {
    const std::string expanded = "module m\n"
                                 "\n"
                                 "fn f(\n"
                                 "    x: f32,\n"
                                 ") -> f32 {\n"
                                 "    return g(\n"
                                 "        x,\n"
                                 "    )\n"
                                 "}\n";
    CHECK_EQ(fmt(expanded), expanded);
    CHECK_EQ(fmt("module m\nfn f(\n    x: f32\n) -> f32 { return g(\nx\n) }\n"),
             "module m\n"
             "\n"
             "fn f(x: f32) -> f32 {\n"
             "    return g(x)\n"
             "}\n");
}

TEST("format: where clauses") {
    CHECK_EQ(fmt("module m\nfn a<D: Dim>() where D % 2 == 0 { return }\n"
                 "fn b<H: Dim, N: Dim>() where H % N == 0, N > 0, { return }\n"),
             "module m\n"
             "\n"
             "fn a<D: Dim>()\n"
             "where D % 2 == 0 {\n"
             "    return\n"
             "}\n"
             "\n"
             "fn b<H: Dim, N: Dim>()\n"
             "where\n"
             "    H % N == 0,\n"
             "    N > 0\n"
             "{\n"
             "    return\n"
             "}\n");
}

TEST("format: expressions break at operators and reduction bodies") {
    const format::LayoutOptions narrow{.width = 44};
    CHECK_EQ(fmt("module m\nfn f() {\n"
                 "let score[b, h, q, k] = sum<f32>[d] cast<f32>(query[b, h, q, d]) * "
                 "cast<f32>(key[b, h, k, d])\n"
                 "let total = first_operand + second_operand + third_operand * fourth_operand\n"
                 "return total\n}\n",
                 narrow),
             "module m\n"
             "\n"
             "fn f() {\n"
             "    let score[b, h, q, k] =\n"
             "        sum<f32>[d]\n"
             "            cast<f32>(query[b, h, q, d]) *\n"
             "            cast<f32>(key[b, h, k, d])\n"
             "    let total =\n"
             "        first_operand +\n"
             "        second_operand +\n"
             "        third_operand * fourth_operand\n"
             "    return total\n"
             "}\n");
}

TEST("format: calls, if, match, tuples, slices") {
    const format::LayoutOptions narrow{.width = 30};
    CHECK_EQ(fmt("module m\nfn f() {\n"
                 "let a = some_function(first_argument, second = 2)\n"
                 "let b = if c { x } else { y }\n"
                 "let d = if condition_is_long { value_one } else { value_two }\n"
                 "let e = match o { some(v) => v none => (1,) }\n"
                 "let g = x[ ... , 0 :: 2, a : b ]\n"
                 "return (a, b)\n}\n",
                 narrow),
             "module m\n"
             "\n"
             "fn f() {\n"
             "    let a = some_function(\n"
             "        first_argument,\n"
             "        second = 2,\n"
             "    )\n"
             "    let b =\n"
             "        if c { x } else { y }\n"
             "    let d =\n"
             "        if condition_is_long {\n"
             "            value_one\n"
             "        } else {\n"
             "            value_two\n"
             "        }\n"
             "    let e = match o {\n"
             "        some(v) => v\n"
             "        none => (1,)\n"
             "    }\n"
             "    let g = x[..., 0::2, a:b]\n"
             "    return (a, b)\n"
             "}\n");
}

TEST("format: blocks keep member grouping and separate methods") {
    CHECK_EQ(fmt("module m\nblock B<H: Dim> {\nparam w: Tensor[H; f32]\nparam b: Tensor[H; f32]? "
                 "= none\n\nbuffer mean: Tensor[H; f32]\nsub inner: [Layer<H>; 4]\n"
                 "pub entry run() -> f32 { return 1.0 }\nfn helper() {}\n}\n"
                 "struct S<T: DType> { a: T\n b: (T, T) }\nenum E { A, B }\n"
                 "type Hidden<B: Dim> = Tensor[B, 4096; bf16]?\n"),
             "module m\n"
             "\n"
             "block B<H: Dim> {\n"
             "    param w: Tensor[H; f32]\n"
             "    param b: Tensor[H; f32]? = none\n"
             "\n"
             "    buffer mean: Tensor[H; f32]\n"
             "    sub inner: [Layer<H>; 4]\n"
             "\n"
             "    pub entry run() -> f32 {\n"
             "        return 1.0\n"
             "    }\n"
             "\n"
             "    fn helper() {}\n"
             "}\n"
             "\n"
             "struct S<T: DType> {\n"
             "    a: T\n"
             "    b: (T, T)\n"
             "}\n"
             "\n"
             "enum E {\n"
             "    A,\n"
             "    B,\n"
             "}\n"
             "\n"
             "type Hidden<B: Dim> = Tensor[B, 4096; bf16]?\n");
}

TEST("format: comments survive in tracked positions") {
    const std::string source = "// file header\n"
                               "\n"
                               "/// module docs\n"
                               "module m\n"
                               "\n"
                               "use std.nn::{\n"
                               "    // the projection\n"
                               "    linear, // trailing\n"
                               "}\n"
                               "\n"
                               "/* block\n"
                               "     comment with\n"
                               "   odd indentation */\n"
                               "fn f(\n"
                               "    x: f32, // the input\n"
                               "    // about y\n"
                               "    y: f32,\n"
                               ") -> f32 {\n"
                               "    // leading\n"
                               "    let a = x // trailing a\n"
                               "\n"
                               "    // before return\n"
                               "    return match o {\n"
                               "        // first arm\n"
                               "        some(v) => v // unwrap\n"
                               "        none => a\n"
                               "        // dangling in match\n"
                               "    }\n"
                               "    // dangling in body\n"
                               "}\n"
                               "\n"
                               "enum E {\n"
                               "    A, // first\n"
                               "    // second\n"
                               "    B,\n"
                               "}\n"
                               "\n"
                               "fn empty() {\n"
                               "    // nothing yet\n"
                               "}\n"
                               "// end of file\n";
    CHECK_EQ(fmt(source), source);
}

TEST("format: comments in untracked positions move to a line boundary") {
    CHECK_EQ(
        fmt("module m\nfn f() -> f32 {\n    let a = 1 + // why\n        2\n    return a /* x */ "
            "+ 1\n}\n"),
        "module m\n"
        "\n"
        "fn f() -> f32 {\n"
        "    let a = 1 + 2\n"
        "    // why\n"
        "    return a + 1\n"
        "    /* x */\n"
        "}\n");
}

TEST("format: a comment inside the module line keeps the blank line after it") {
    // Printed after `module t`, the comment reads as the first item's leading
    // comment on the next pass, which puts a blank line before it; the first
    // pass must too, or formatting is not a fixed point.
    CHECK_EQ(fmt("module//\nt fn m(){}"), "module t\n\n//\n\nfn m() {}\n");
    CHECK_EQ(fmt("module//\nu block C{}"), "module u\n\n//\n\nblock C {}\n");
}

TEST("format: one-element tuple types keep their comma") {
    CHECK_EQ(fmt("module m\nfn f(x: (f32,)) -> (f32,) { return x }\n"),
             "module m\n\nfn f(x: (f32,)) -> (f32,) {\n    return x\n}\n");
}

TEST("format: CRLF input and trailing whitespace") {
    CHECK_EQ(fmt("module m\r\n\r\n// note   \r\nfn f() {   \r\n    return\r\n}\r\n"),
             "module m\n"
             "\n"
             "// note\n"
             "fn f() {\n"
             "    return\n"
             "}\n");
}
