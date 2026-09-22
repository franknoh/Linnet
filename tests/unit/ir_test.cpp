#include "linnet/ir/ir.hpp"
#include "linnet/ir/lower.hpp"
#include "linnet/sema/analysis.hpp"
#include "linnet/syntax/parser.hpp"

#include "test.hpp"

#include <string>

using namespace linnet;

namespace {

// Lowers one module and returns its printed IR; verifier problems are
// appended so that a test sees them.
std::string lower_text(const std::string& source) {
    SourceManager sources;
    DiagnosticSink sink;
    const FileId file = sources.add_file("m.linnet", source).value();
    const ast::Ast tree = parse(sources, file, sink);
    const ast::Ast* const modules[] = {&tree};
    const sema::AnalysisResult analysis = sema::analyze(sources, modules, sink);
    CHECK(!sink.has_errors());
    if (sink.has_errors()) {
        return {};
    }
    const ir::Module module = ir::lower(sources, modules, analysis.model);
    std::string text = ir::print(module);
    for (const std::string& problem : ir::verify(module)) {
        text += "VERIFY: " + problem + "\n";
    }
    return text;
}

} // namespace

TEST("ir: comprehension, reduction, option match, and semantic calls") {
    CHECK_EQ(
        lower_text("module m\n"
                   "pub op linear<*S: Shape, In: Dim, Out: Dim, T: Float>(\n"
                   "    x: Tensor[*S, In; T],\n"
                   "    w: Tensor[Out, In; T],\n"
                   "    bias: Tensor[Out; T]? = none,\n"
                   ") -> Tensor[*S, Out; T] {\n"
                   "    let y[*s, o] = sum[i] x[*s, i] * w[o, i]\n"
                   "    return match bias {\n"
                   "        some(b) => y + b\n"
                   "        none => y\n"
                   "    }\n"
                   "}\n"
                   "fn user(x: Tensor[2, 3; f32], w: Tensor[4, 3; f32]) -> Tensor[2, 4; f32] {\n"
                   "    return linear(x, w)\n"
                   "}\n"),
        "op @m::linear<*S, In, Out, T>(%x0: Tensor[*S, In; T], %w1: Tensor[Out, In; T], "
        "%bias2: Tensor[Out; T]?) -> Tensor[*S, Out; T] {\n"
        "    %y3 = comprehension [s: *S, o: Out] : Tensor[*S, Out; T] {\n"
        "    ^(%s4: shape [*S], %o5: i64):\n"
        "        %6 = reduce sum [i: In] : T {\n"
        "        ^(%i7: i64):\n"
        "            %8 = tensor.element %x0, %s4, %i7 : T\n"
        "            %9 = tensor.element %w1, %o5, %i7 : T\n"
        "            %10 = mul %8, %9 : T\n"
        "            yield %10\n"
        "        }\n"
        "        yield %6\n"
        "    }\n"
        "    %11 = option.match %bias2 : Tensor[*S, Out; T] {\n"
        "    ^(%value12: Tensor[Out; T]):\n"
        "        %13 = add %y3, %value12 : Tensor[*S, Out; T]\n"
        "        yield %13\n"
        "    } {\n"
        "        yield %y3\n"
        "    }\n"
        "    return %11\n"
        "}\n"
        "\n"
        "fn @m::user(%x14: Tensor[2, 3; f32], %w15: Tensor[4, 3; f32]) -> Tensor[2, 4; f32] {\n"
        "    %16 = option.none : Tensor[4; f32]?\n"
        "    %17 = semantic.call @m::linear<In = 3, Out = 4, *S = [2], T = f32> %x14, %w15, %16 : "
        "Tensor[2, 4; f32]\n"
        "    return %17\n"
        "}\n\n");
}

TEST("ir: literals take the dtype of their context") {
    const std::string text =
        lower_text("module m\nfn f(x: Tensor[4; f16], n: i32) -> Tensor[4; f16] {\n"
                   "    let a = x * 2 + 0.5\n    let b = n + 1\n    let c = 1\n    return a\n}\n");
    CHECK(text.find("const.float 2.000000 : f16") != std::string::npos);
    CHECK(text.find("const.float 0.500000 : f16") != std::string::npos);
    CHECK(text.find("const.int 1 : i32") != std::string::npos);
    CHECK(text.find("%c") != std::string::npos &&
          text.find("const.int 1 : i64") != std::string::npos);
    CHECK(text.find("VERIFY") == std::string::npos);
}

TEST("ir: blocks, static for, and slicing") {
    const std::string text =
        lower_text("module m\n"
                   "block Layer<H: Dim> {\n    param w: Tensor[H; f32]\n"
                   "    pub fn forward<B: Dim>(x: Tensor[B, H; f32]) -> Tensor[B, H; f32] { return "
                   "x * w }\n}\n"
                   "block Model<H: Dim, N: Dim> {\n    sub layers: [Layer<H>; N]\n"
                   "    pub entry run<B: Dim>(x0: Tensor[B, H; f32]) -> Tensor[B, H / 2; f32] {\n"
                   "        var x = x0\n        static for layer in layers {\n            x = "
                   "layer.forward(x)\n"
                   "        }\n        let first = layers[0].forward(x)\n        return first[:, "
                   "0:H / 2]\n    }\n}\n");
    CHECK(text.find("fn @m::Layer.forward<B>(%self0: Layer<H>, %x1: Tensor[B, H; f32])") !=
          std::string::npos);
    CHECK(text.find("block.param \"w\" %self0") != std::string::npos);
    CHECK(text.find("static_for %") != std::string::npos);
    CHECK(text.find("yield %") != std::string::npos);
    CHECK(text.find("array.get") != std::string::npos);
    CHECK(text.find("slice [0:B:1, 0:H / 2:1]") != std::string::npos);
    CHECK(text.find("VERIFY") == std::string::npos);
}

TEST("ir: if, enums, tuples, defaults, and builtins") {
    const std::string text = lower_text(
        "module m\n"
        "enum Mask { None, Causal }\n"
        "fn scale<T: Float>(x: T, k: T = 2.0) -> T { return x * k }\n"
        "fn f(a: f32, flag: bool, m: Mask, t: Tensor[2, 3; f32]) -> (f32, Tensor[3, 2; f32]) {\n"
        "    let b = if flag { scale(a) } else { scale(a, 3.0) }\n"
        "    let c = match m { None => b Causal => b * 2.0 }\n"
        "    let same = m == Mask.Causal\n"
        "    let p = permute(cast<f32>(t), [1, 0])\n"
        "    let q = reshape(p, [6])\n"
        "    let r = concat(q, q, axis = 0)\n"
        "    let s = fill<f32>([3, 2], 1.0) + iota<f32>(2)\n"
        "    let (u, v) = (c, s)\n"
        "    return (u, select(flag, p, v))\n}\n");
    for (const char* needle : {"if %",
                               "enum.match None Causal",
                               "enum.const \"Causal\"",
                               "const.float 2.000000 : f32",
                               "call @m::scale<T = f32>",
                               "permute [1, 0]",
                               "reshape [6]",
                               "concat axis 0",
                               "fill [3, 2]",
                               "iota [2]",
                               "tuple.make",
                               "tuple.get 0",
                               "tuple.get 1",
                               "select %",
                               "cast %"}) {
        CHECK(text.find(needle) != std::string::npos);
        if (text.find(needle) == std::string::npos) {
            linnet::test::report_failure(needle, 0, text);
        }
    }
    CHECK(text.find("VERIFY") == std::string::npos);
}

TEST("ir: verifier rejects malformed modules") {
    auto model = std::make_shared<sema::Model>();
    ir::Module module(model);
    ir::Function function;
    function.name = "broken";
    function.results.push_back(model->types.scalar(sema::ScalarKind::F32));
    function.body = module.add_region(ir::no_id);
    module.add_function(function);
    const ir::BlockId block = module.add_block(function.body);
    CHECK_EQ(ir::verify(module).size(), 1U); // no terminator
    const ir::ValueId later = module.add_op(
        block, ir::OpKind::ConstInt, {}, {model->types.scalar(sema::ScalarKind::I64)});
    module.add_op(block,
                  ir::OpKind::Add,
                  {module.op(later).results.front(), 999},
                  {model->types.scalar(sema::ScalarKind::I64)});
    module.add_op(block, ir::OpKind::Return, {}, {});
    const std::vector<std::string> problems = ir::verify(module);
    CHECK_EQ(problems.size(), 2U); // undefined operand, wrong return arity
}
