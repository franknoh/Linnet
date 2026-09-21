#pragma once

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace linnet::format {

using DocId = std::uint32_t;

// Arena of layout documents. The formatter describes *what* may break and the
// printer decides *where*, so layout policy lives in exactly one place.
//
// A group is printed flat (on one line) when it fits in the remaining width;
// otherwise its direct line breaks become newlines. Nested groups are decided
// independently.
class DocBuilder {
public:
    DocBuilder();

    DocId nil() const { return 0; }

    // Literal text; must not contain newlines.
    DocId text(std::string_view text);

    DocId line();      // a space when flat, a newline when broken
    DocId soft_line(); // nothing when flat, a newline when broken
    DocId hard_line(); // always a newline; forces every enclosing group to break

    // A newline followed by no indentation at all; forces enclosing groups to
    // break. Used for the continuation lines of multi-line comments, which are
    // reproduced verbatim.
    DocId literal_line();

    // Prints nothing but forces every enclosing group to break, for example
    // after a trailing line comment.
    DocId break_parent();

    DocId concat(std::span<const DocId> parts);
    DocId concat(std::initializer_list<DocId> parts);

    // Parts separated by `separator`.
    DocId join(DocId separator, std::span<const DocId> parts);

    // Increases indentation of newlines inside `body` by one level.
    DocId indent(DocId body);

    DocId group(DocId body);

    // `broken` when the enclosing group is broken, `flat` otherwise. Used for
    // trailing commas that appear only in multi-line lists.
    DocId if_break(DocId broken, DocId flat);

private:
    friend class Printer;

    enum class Kind : std::uint8_t {
        Nil,
        Text,
        Line,
        SoftLine,
        HardLine,
        LiteralLine,
        BreakParent,
        Concat,
        Indent,
        Group,
        IfBreak
    };

    struct Node {
        Kind kind;
        bool forces_break;   // contains a hard line
        std::uint32_t a;     // Text: offset; Concat: first child index; others: child doc
        std::uint32_t b;     // Text: byte length; Concat: child count; IfBreak: flat doc
        std::uint32_t width; // Text: display width in code points
    };

    DocId add(Node node);

    std::vector<Node> nodes_;
    std::vector<DocId> children_;
    std::string text_;
};

struct LayoutOptions {
    std::uint32_t width = 100;
    std::uint32_t indent_width = 4;
};

// Prints a document. Lines never carry trailing whitespace. No newline is
// appended after the final line.
std::string print(const DocBuilder& builder, DocId root, const LayoutOptions& options = {});

} // namespace linnet::format
