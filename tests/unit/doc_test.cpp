#include "linnet/format/doc.hpp"

#include "test.hpp"

#include <string>
#include <vector>

using namespace linnet::format;

namespace {

// name(arg, arg, ...) with a trailing comma only when broken.
DocId call(DocBuilder& b, std::string_view name, const std::vector<DocId>& args) {
    const DocId separator = b.concat({b.text(","), b.line()});
    return b.group(b.concat({
        b.text(name),
        b.text("("),
        b.indent(b.concat({b.soft_line(), b.join(separator, args)})),
        b.if_break(b.text(","), b.nil()),
        b.soft_line(),
        b.text(")"),
    }));
}

} // namespace

TEST("doc: group that fits prints flat") {
    DocBuilder b;
    const DocId doc = call(b, "f", {b.text("a"), b.text("b")});
    CHECK_EQ(print(b, doc, {.width = 20}), "f(a, b)");
}

TEST("doc: group that does not fit breaks with trailing comma") {
    DocBuilder b;
    const DocId doc = call(b, "function", {b.text("argument_one"), b.text("argument_two")});
    CHECK_EQ(print(b, doc, {.width = 20}),
             "function(\n"
             "    argument_one,\n"
             "    argument_two,\n"
             ")");
}

TEST("doc: exact width fits, one more column does not") {
    DocBuilder b;
    const DocId doc = call(b, "f", {b.text("aaaa"), b.text("bbbb")}); // f(aaaa, bbbb) = 13
    CHECK_EQ(print(b, doc, {.width = 13}), "f(aaaa, bbbb)");
    CHECK_EQ(print(b, doc, {.width = 12}), "f(\n    aaaa,\n    bbbb,\n)");
}

TEST("doc: nested groups break outermost first") {
    DocBuilder b;
    const DocId inner = call(b, "inner", {b.text("x"), b.text("y")});
    const DocId doc = call(b, "outer", {inner, b.text("z")});
    CHECK_EQ(print(b, doc, {.width = 40}), "outer(inner(x, y), z)");
    CHECK_EQ(print(b, doc, {.width = 18}),
             "outer(\n"
             "    inner(x, y),\n"
             "    z,\n"
             ")");
    CHECK_EQ(print(b, doc, {.width = 10}),
             "outer(\n"
             "    inner(\n"
             "        x,\n"
             "        y,\n"
             "    ),\n"
             "    z,\n"
             ")");
}

TEST("doc: text after a group counts toward its fit") {
    DocBuilder b;
    const DocId doc = b.concat({call(b, "f", {b.text("aaaa")}), b.text(" -> Result")});
    CHECK_EQ(print(b, doc, {.width = 17}), "f(aaaa) -> Result");
    CHECK_EQ(print(b, doc, {.width = 16}), "f(\n    aaaa,\n) -> Result");
}

TEST("doc: hard line forces enclosing groups to break") {
    DocBuilder b;
    const DocId body = b.concat({b.text("a"), b.hard_line(), b.text("b")});
    const DocId doc = b.group(b.concat({
        b.text("{"),
        b.indent(b.concat({b.line(), body})),
        b.line(),
        b.text("}"),
    }));
    CHECK_EQ(print(b, doc), "{\n    a\n    b\n}");
}

TEST("doc: blank lines carry no indentation or trailing spaces") {
    DocBuilder b;
    const DocId doc = b.concat({
        b.text("{"),
        b.indent(
            b.concat({b.hard_line(), b.text("a "), b.hard_line(), b.hard_line(), b.text("b")})),
        b.hard_line(),
        b.text("}  "),
    });
    CHECK_EQ(print(b, doc), "{\n    a\n\n    b\n}");
}

TEST("doc: width counts code points, not bytes") {
    DocBuilder b;
    const DocId doc = call(b, "f", {b.text("\xC3\xA9\xC3\xA9"), b.text("\xF0\x9F\x98\x80")});
    CHECK_EQ(print(b, doc, {.width = 8}), "f(\xC3\xA9\xC3\xA9, \xF0\x9F\x98\x80)");
}

TEST("doc: empty and single-part concat collapse") {
    DocBuilder b;
    CHECK_EQ(b.concat({}), b.nil());
    const DocId text = b.text("x");
    CHECK_EQ(b.concat({text}), text);
    CHECK_EQ(b.text(""), b.nil());
    CHECK_EQ(print(b, b.nil()), "");
}

TEST("doc: deeply nested documents do not overflow the stack") {
    DocBuilder b;
    DocId doc = b.text("x");
    for (int i = 0; i < 5000; ++i) {
        doc = b.group(b.indent(doc));
    }
    CHECK_EQ(print(b, doc), "x");
}
