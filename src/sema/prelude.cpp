// The dtypes the prelude declares (spec §3.1), read from the compiled-in
// `stdlib/prelude.linnet` once: each `dtype` declaration's name and the
// scalar kind of the format it describes.

#include "linnet/prelude.hpp"

#include "linnet/ast/ast.hpp"
#include "linnet/diagnostic/diagnostic.hpp"
#include "linnet/sema/types.hpp"
#include "linnet/source/source_manager.hpp"
#include "linnet/syntax/parser.hpp"

#include <charconv>
#include <stdexcept>
#include <string>

namespace linnet::sema {

namespace {

[[noreturn]] void malformed(const std::string& why) {
    throw std::logic_error("the compiled-in prelude is malformed: " + why);
}

std::map<std::string, ScalarKind, std::less<>> read_prelude() {
    SourceManager sources;
    const FileId file = sources.add_file("std/prelude.linnet", std::string(prelude_source)).value();
    DiagnosticSink sink;
    const ast::Ast tree = parse(sources, file, sink);
    if (sink.has_errors()) {
        malformed("it does not parse");
    }
    std::map<std::string, ScalarKind, std::less<>> declared;
    for (const ast::ItemId id : tree.items) {
        const auto* decl = std::get_if<ast::DTypeDecl>(&tree.item(id).data);
        if (decl == nullptr) {
            continue;
        }
        const std::string name(decl->name.text);
        const bool is_float = decl->family.text == "float";
        std::map<std::string_view, std::int64_t> values;
        for (const ast::DTypeArg& arg : decl->args) {
            const ast::Expr& value = tree.expr(arg.value);
            const auto* literal = std::get_if<ast::LiteralExpr>(&value.data);
            if (literal == nullptr) {
                malformed("`" + name + "` has a parameter that is not a literal");
            }
            std::int64_t number = 0;
            if (literal->kind == ast::LiteralKind::True ||
                literal->kind == ast::LiteralKind::False) {
                number = literal->kind == ast::LiteralKind::True ? 1 : 0;
            } else {
                const std::string_view text = sources.text(value.span);
                const auto [end, error] =
                    std::from_chars(text.data(), text.data() + text.size(), number);
                if (error != std::errc() || end != text.data() + text.size()) {
                    malformed("`" + name + "` has a width that is not an integer");
                }
            }
            values.emplace(arg.name.text, number);
        }
        const auto kind = scalar_of(dtype_format(is_float, values));
        // Its name is the one plans and generated code spell the kind with.
        if (!kind || scalar_name(*kind) != name) {
            malformed("`" + name + "` describes no format of that name");
        }
        declared.emplace(name, *kind);
    }
    return declared;
}

} // namespace

const std::map<std::string, ScalarKind, std::less<>>& prelude_dtypes() {
    static const std::map<std::string, ScalarKind, std::less<>> declared = read_prelude();
    return declared;
}

} // namespace linnet::sema
