#include "linnet/format/doc.hpp"

#include "linnet/source/utf8.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace linnet::format {

DocBuilder::DocBuilder() {
    // Shared leaf nodes occupy fixed slots.
    nodes_.push_back({Kind::Nil, false, 0, 0, 0});
    nodes_.push_back({Kind::Line, false, 0, 0, 0});
    nodes_.push_back({Kind::SoftLine, false, 0, 0, 0});
    nodes_.push_back({Kind::HardLine, true, 0, 0, 0});
    nodes_.push_back({Kind::LiteralLine, true, 0, 0, 0});
    nodes_.push_back({Kind::BreakParent, true, 0, 0, 0});
}

DocId DocBuilder::add(Node node) {
    nodes_.push_back(node);
    return static_cast<DocId>(nodes_.size() - 1);
}

DocId DocBuilder::text(std::string_view text) {
    assert(text.find('\n') == std::string_view::npos);
    if (text.empty()) {
        return nil();
    }
    std::uint32_t width = 0;
    for (std::size_t offset = 0; offset < text.size(); offset += decode_utf8(text, offset).length) {
        ++width;
    }
    const auto offset = static_cast<std::uint32_t>(text_.size());
    text_ += text;
    return add({Kind::Text, false, offset, static_cast<std::uint32_t>(text.size()), width});
}

DocId DocBuilder::line() {
    return 1;
}

DocId DocBuilder::soft_line() {
    return 2;
}

DocId DocBuilder::hard_line() {
    return 3;
}

DocId DocBuilder::literal_line() {
    return 4;
}

DocId DocBuilder::break_parent() {
    return 5;
}

DocId DocBuilder::concat(std::span<const DocId> parts) {
    if (parts.empty()) {
        return nil();
    }
    if (parts.size() == 1) {
        return parts.front();
    }
    const auto first = static_cast<std::uint32_t>(children_.size());
    bool forces_break = false;
    for (const DocId part : parts) {
        assert(part < nodes_.size());
        forces_break = forces_break || nodes_[part].forces_break;
    }
    // `parts` may alias children_ storage only if a caller passed internal
    // state, which the API never exposes; appending is safe.
    children_.insert(children_.end(), parts.begin(), parts.end());
    return add({Kind::Concat, forces_break, first, static_cast<std::uint32_t>(parts.size()), 0});
}

DocId DocBuilder::concat(std::initializer_list<DocId> parts) {
    return concat(std::span<const DocId>(parts.begin(), parts.size()));
}

DocId DocBuilder::join(DocId separator, std::span<const DocId> parts) {
    std::vector<DocId> joined;
    joined.reserve(parts.size() * 2);
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            joined.push_back(separator);
        }
        joined.push_back(parts[i]);
    }
    return concat(joined);
}

DocId DocBuilder::indent(DocId body) {
    return add({Kind::Indent, nodes_[body].forces_break, body, 0, 0});
}

DocId DocBuilder::group(DocId body) {
    return add({Kind::Group, nodes_[body].forces_break, body, 0, 0});
}

DocId DocBuilder::if_break(DocId broken, DocId flat) {
    return add({Kind::IfBreak, nodes_[flat].forces_break, broken, flat, 0});
}

class Printer {
public:
    Printer(const DocBuilder& builder, const LayoutOptions& options)
        : builder_(builder), options_(options) {}

    std::string print(DocId root) {
        stack_.push_back({root, 0, Mode::Break});
        while (!stack_.empty()) {
            const Command command = stack_.back();
            stack_.pop_back();
            step(command);
        }
        trim_trailing_spaces();
        return std::move(out_);
    }

private:
    using Kind = DocBuilder::Kind;
    using Node = DocBuilder::Node;

    enum class Mode : std::uint8_t { Flat, Break };

    struct Command {
        DocId doc;
        std::uint32_t indent;
        Mode mode;
    };

    void
    push_children(const Node& node, std::uint32_t indent, Mode mode, std::vector<Command>& to) {
        for (std::uint32_t i = node.b; i > 0; --i) {
            to.push_back({builder_.children_[node.a + i - 1], indent, mode});
        }
    }

    void step(const Command& command) {
        const Node& node = builder_.nodes_[command.doc];
        switch (node.kind) {
        case Kind::Nil:
            break;
        case Kind::Text:
            out_.append(pending_indent_, ' ');
            pending_indent_ = 0;
            out_.append(builder_.text_, node.a, node.b);
            column_ += node.width;
            break;
        case Kind::Line:
        case Kind::SoftLine:
            if (command.mode == Mode::Break) {
                newline(command.indent);
            } else if (node.kind == Kind::Line) {
                out_ += ' ';
                ++column_;
            }
            break;
        case Kind::HardLine:
            newline(command.indent);
            break;
        case Kind::LiteralLine:
            newline(0);
            break;
        case Kind::BreakParent:
            break;
        case Kind::Concat:
            push_children(node, command.indent, command.mode, stack_);
            break;
        case Kind::Indent:
            stack_.push_back({node.a, command.indent + options_.indent_width, command.mode});
            break;
        case Kind::Group: {
            const Command flat{node.a, command.indent, Mode::Flat};
            const bool is_flat = !node.forces_break && fits(flat);
            stack_.push_back({node.a, command.indent, is_flat ? Mode::Flat : Mode::Break});
            break;
        }
        case Kind::IfBreak:
            stack_.push_back(
                {command.mode == Mode::Break ? node.a : node.b, command.indent, command.mode});
            break;
        }
    }

    void trim_trailing_spaces() {
        while (!out_.empty() && out_.back() == ' ') {
            out_.pop_back();
        }
    }

    void newline(std::uint32_t indent) {
        trim_trailing_spaces();
        out_ += '\n';
        // Indentation is emitted lazily so blank lines stay empty.
        pending_indent_ = indent;
        column_ = indent;
    }

    // Whether `first`, followed by the rest of the current line, fits in the
    // remaining width. The rest of the line ends at the first newline that the
    // already-decided enclosing layout will produce.
    bool fits(const Command& first) {
        if (column_ > options_.width) {
            return false;
        }
        std::uint32_t remaining = options_.width - column_;
        std::size_t rest = stack_.size();
        scratch_.clear();
        scratch_.push_back(first);

        while (true) {
            if (scratch_.empty()) {
                if (rest == 0) {
                    return true;
                }
                scratch_.push_back(stack_[--rest]);
            }
            const Command command = scratch_.back();
            scratch_.pop_back();
            const Node& node = builder_.nodes_[command.doc];
            switch (node.kind) {
            case Kind::Nil:
                break;
            case Kind::Text:
                if (node.width > remaining) {
                    return false;
                }
                remaining -= node.width;
                break;
            case Kind::Line:
            case Kind::SoftLine:
                if (command.mode == Mode::Break) {
                    return true;
                }
                if (node.kind == Kind::Line) {
                    if (remaining == 0) {
                        return false;
                    }
                    --remaining;
                }
                break;
            case Kind::HardLine:
            case Kind::LiteralLine:
                return true;
            case Kind::BreakParent:
                break;
            case Kind::Concat:
                push_children(node, command.indent, command.mode, scratch_);
                break;
            case Kind::Indent:
                scratch_.push_back({node.a, command.indent, command.mode});
                break;
            case Kind::Group:
                // A nested group that must break ends the line at its first break.
                scratch_.push_back(
                    {node.a, command.indent, node.forces_break ? Mode::Break : command.mode});
                break;
            case Kind::IfBreak:
                scratch_.push_back(
                    {command.mode == Mode::Break ? node.a : node.b, command.indent, command.mode});
                break;
            }
        }
    }

    const DocBuilder& builder_;
    const LayoutOptions& options_;
    std::vector<Command> stack_;
    std::vector<Command> scratch_;
    std::string out_;
    std::uint32_t column_ = 0;
    std::uint32_t pending_indent_ = 0;
};

std::string print(const DocBuilder& builder, DocId root, const LayoutOptions& options) {
    return Printer(builder, options).print(root);
}

} // namespace linnet::format
