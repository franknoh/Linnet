#include "linnet/backend/graph_export.hpp"
#include "linnet/support/text.hpp"

#include "test.hpp"

#include <string>
#include <utility>
#include <vector>

using namespace linnet;
using namespace linnet::backend;

TEST("python text: lines, with or without a last newline") {
    CHECK(split_lines("a\nb\n") == std::vector<std::string>({"a", "b"}));
    CHECK(split_lines("a\n\nb") == std::vector<std::string>({"a", "", "b"}));
    CHECK(split_lines("").empty());
}

TEST("python text: a generated assignment at any indent") {
    const Assignment nested = assignment_of("        v12 = torch.exp(v3)").value_or(Assignment{});
    CHECK_EQ(nested.indent, std::string(8, ' '));
    CHECK_EQ(nested.name, "v12");
    CHECK_EQ(nested.expression, "torch.exp(v3)");
    CHECK(!assignment_of("    out = v3").has_value());
    CHECK(!assignment_of("    v3.add_(v4)").has_value());
    CHECK(!top_level_assignment("        v12 = v3").has_value());
    CHECK_EQ(
        top_level_assignment("    v12 = v3").value_or(std::pair<std::string, std::string>{}).first,
        "v12");
}

TEST("python text: a call's arguments split outside brackets and strings") {
    const Call call = call_of("F.linear(F.linear(x, a), b[0, 1], \"c, d\")").value_or(Call{});
    CHECK_EQ(call.callee, "F.linear");
    CHECK(call.arguments == std::vector<std::string>({"F.linear(x, a)", "b[0, 1]", "\"c, d\""}));
    CHECK(call_of("f()").value_or(Call{"none", {"x"}}).arguments.empty());
    CHECK(!call_of("f(x)[0]").has_value());
    CHECK(!call_of("f(x) + g(y)").has_value());
    CHECK(!call_of("(x)").has_value());
}

TEST("python text: indented lines keep blank lines blank") {
    CHECK_EQ(indent_lines("a\n\nb", "    "), "    a\n\n    b");
}

TEST("python text: floats and literals") {
    CHECK_EQ(decimal_float(1e-06), "1.0e-06");
    CHECK_EQ(decimal_float(0.5), "0.5");
    CHECK_EQ(decimal_float(2.0), "2.0");
    CHECK_EQ(python_literal(Literal::of_integer(3), sema::ScalarKind::F32).value_or(""), "3.0");
    CHECK_EQ(python_literal(Literal::of_integer(3), sema::ScalarKind::I64).value_or(""), "3");
    CHECK(!python_literal(Literal::of(Literal::Kind::Highest), sema::ScalarKind::F32));
}
