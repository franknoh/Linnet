#include "linnet/ast/dump.hpp"

#include <string_view>

namespace linnet::ast {

namespace {

template <typename... Visitors>
struct Overloaded : Visitors... {
    using Visitors::operator()...;
};
template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

class Dumper {
public:
    Dumper(const Ast& ast, const SourceManager& sources) : ast_(ast), sources_(sources) {}

    std::string run() {
        line("module " + path(ast_.module_path));
        for (const UseDecl& use : ast_.uses) {
            std::string text = "use " + path(use.path);
            for (const ImportName& import : use.names) {
                text += " " + std::string(import.name.text);
                if (!import.alias.text.empty()) {
                    text += " as " + std::string(import.alias.text);
                }
            }
            line(text);
        }
        for (const ItemId id : ast_.items) {
            item(id);
        }
        return std::move(out_);
    }

private:
    void line(std::string_view text) {
        out_.append(indent_ * 2, ' ');
        out_ += text;
        out_ += '\n';
    }

    template <typename Body>
    void nested(std::string_view header, Body&& body) {
        line(header);
        ++indent_;
        body();
        --indent_;
    }

    static std::string path(const std::vector<Name>& names) {
        std::string text;
        for (const Name& name : names) {
            if (!text.empty()) {
                text += '.';
            }
            text += name.text.empty() ? "<missing>" : name.text;
        }
        return text.empty() ? "<missing>" : text;
    }

    static std::string name(const Name& name) {
        return name.text.empty() ? "<missing>" : std::string(name.text);
    }

    // ------------------------------------------------------------------- types

    std::string generic_args(const std::vector<GenericArg>& args) {
        std::string text = "<";
        for (std::size_t i = 0; i < args.size(); ++i) {
            text += i == 0 ? "" : ", ";
            text += args[i].type != no_id ? type(args[i].type) : inline_expr(args[i].expr);
        }
        return text + ">";
    }

    std::string type(TypeId id) {
        if (id == no_id) {
            return "<none>";
        }
        return std::visit(
            Overloaded{
                [](const ErrorType&) { return std::string("<error>"); },
                [&](const NamedType& named) {
                    return path(named.path) + (named.args.empty() ? "" : generic_args(named.args));
                },
                [&](const TensorType& tensor) {
                    std::string text = "Tensor[";
                    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
                        const ShapeElement& element = tensor.shape[i];
                        text += i == 0 ? "" : ", ";
                        text += element.dim == no_id ? "*" + name(element.pack)
                                                     : inline_expr(element.dim);
                    }
                    return text + "; " + type(tensor.dtype) + "]";
                },
                [&](const TupleType& tuple) {
                    std::string text = "(";
                    for (std::size_t i = 0; i < tuple.elements.size(); ++i) {
                        text += i == 0 ? "" : ", ";
                        text += type(tuple.elements[i]);
                    }
                    return text + ")";
                },
                [&](const ArrayType& array) {
                    return "[" + type(array.element) + "; " + inline_expr(array.length) + "]";
                },
                [&](const OptionalType& optional) { return type(optional.inner) + "?"; },
            },
            ast_.type(id).data);
    }

    // Fully parenthesized single-line rendering, used inside types.
    std::string inline_expr(ExprId id) {
        if (id == no_id) {
            return "<none>";
        }
        const Expr& node = ast_.expr(id);
        if (const auto* binary = std::get_if<BinaryExpr>(&node.data)) {
            return "(" + inline_expr(binary->lhs) + " " +
                   std::string(binary_op_spelling(binary->op)) + " " + inline_expr(binary->rhs) +
                   ")";
        }
        if (const auto* unary = std::get_if<UnaryExpr>(&node.data)) {
            return std::string(unary_op_spelling(unary->op)) + inline_expr(unary->operand);
        }
        if (const auto* paren = std::get_if<ParenExpr>(&node.data)) {
            return inline_expr(paren->inner);
        }
        if (const auto* named = std::get_if<NameExpr>(&node.data)) {
            return name(named->name);
        }
        if (std::holds_alternative<LiteralExpr>(node.data)) {
            return std::string(source_text(node.span));
        }
        if (std::holds_alternative<ErrorExpr>(node.data)) {
            return "<error>";
        }
        if (const auto* member = std::get_if<MemberExpr>(&node.data)) {
            return inline_expr(member->base) + "." + name(member->member);
        }
        // Other forms are rare inside types. Their source text is not shown:
        // the dump must not depend on layout or comments.
        return "{expr}";
    }

    std::string_view source_text(SourceSpan span) const { return sources_.text(span); }

    // ------------------------------------------------------------- expressions

    void optional_expr(std::string_view label, ExprId id) {
        if (id != no_id) {
            nested(label, [&] { expr(id); });
        }
    }

    void expr(ExprId id) {
        if (id == no_id) {
            line("<none>");
            return;
        }
        const Expr& node = ast_.expr(id);
        std::visit(
            Overloaded{
                [&](const ErrorExpr&) { line("error"); },
                [&](const LiteralExpr&) { line("literal " + std::string(source_text(node.span))); },
                [&](const NameExpr& named) { line("name " + name(named.name)); },
                [&](const NoneExpr&) { line("none"); },
                [&](const SomeExpr& some) { nested("some", [&] { expr(some.value); }); },
                [&](const ParenExpr& paren) { nested("paren", [&] { expr(paren.inner); }); },
                [&](const TupleExpr& tuple) {
                    nested("tuple", [&] {
                        for (const ExprId element : tuple.elements) {
                            expr(element);
                        }
                    });
                },
                [&](const ShapeExpr& shape) {
                    nested("shape", [&] {
                        for (const ExprId dim : shape.dims) {
                            expr(dim);
                        }
                    });
                },
                [&](const UnaryExpr& unary) {
                    nested("unary " + std::string(unary_op_spelling(unary.op)),
                           [&] { expr(unary.operand); });
                },
                [&](const BinaryExpr& binary) {
                    nested("binary " + std::string(binary_op_spelling(binary.op)), [&] {
                        expr(binary.lhs);
                        expr(binary.rhs);
                    });
                },
                [&](const CallExpr& call) {
                    nested("call" + (call.generic_args.empty()
                                         ? std::string()
                                         : " " + generic_args(call.generic_args)),
                           [&] {
                               expr(call.callee);
                               for (const Argument& arg : call.args) {
                                   if (arg.keyword.text.empty()) {
                                       nested("arg", [&] { expr(arg.value); });
                                   } else {
                                       nested("arg " + name(arg.keyword), [&] { expr(arg.value); });
                                   }
                               }
                           });
                },
                [&](const IndexExpr& index) {
                    nested("index", [&] {
                        expr(index.base);
                        for (const IndexComponent& component : index.components) {
                            index_component(component);
                        }
                    });
                },
                [&](const MemberExpr& member) {
                    nested("member " + name(member.member), [&] { expr(member.base); });
                },
                [&](const ReductionExpr& reduction) {
                    std::string header(reduction_kind_spelling(reduction.kind));
                    if (reduction.accumulator != no_id) {
                        header += "<" + type(reduction.accumulator) + ">";
                    }
                    header += " [";
                    for (std::size_t i = 0; i < reduction.indices.size(); ++i) {
                        header += i == 0 ? "" : ", ";
                        header += name(reduction.indices[i]);
                    }
                    nested(header + "]", [&] { expr(reduction.body); });
                },
                [&](const IfExpr& conditional) {
                    nested("if", [&] {
                        expr(conditional.condition);
                        nested("then", [&] { expr(conditional.then_value); });
                        nested("else", [&] { expr(conditional.else_value); });
                    });
                },
                [&](const MatchExpr& match) {
                    nested("match", [&] {
                        expr(match.scrutinee);
                        for (const MatchArm& arm : match.arms) {
                            nested("arm " + pattern(arm.pattern), [&] { expr(arm.value); });
                        }
                    });
                },
            },
            node.data);
    }

    void index_component(const IndexComponent& component) {
        switch (component.kind) {
        case IndexKind::Expr:
            expr(component.value);
            break;
        case IndexKind::Ellipsis:
            line("...");
            break;
        case IndexKind::Pack:
            line("pack *" + name(component.pack));
            break;
        case IndexKind::Slice:
            nested("slice", [&] {
                optional_expr("start", component.start);
                optional_expr("stop", component.stop);
                optional_expr("step", component.step);
            });
            break;
        }
    }

    // ---------------------------------------------------- patterns, statements

    std::string pattern(PatternId id) {
        return std::visit(
            Overloaded{
                [](const ErrorPattern&) { return std::string("<error>"); },
                [](const BindingPattern& binding) { return name(binding.name); },
                [&](const TuplePattern& tuple) {
                    std::string text = "(";
                    for (std::size_t i = 0; i < tuple.elements.size(); ++i) {
                        text += i == 0 ? "" : ", ";
                        text += pattern(tuple.elements[i]);
                    }
                    return text + ")";
                },
                [&](const SomePattern& some) { return "some(" + pattern(some.inner) + ")"; },
                [](const NonePattern&) { return std::string("none"); },
            },
            ast_.pattern(id).data);
    }

    static std::string
    annotation(const std::string& text, const std::string& type_text, TypeId id) {
        return id == no_id ? text : text + ": " + type_text;
    }

    void body(const std::vector<StmtId>& statements) {
        nested("body", [&] {
            for (const StmtId id : statements) {
                stmt(id);
            }
        });
    }

    void stmt(StmtId id) {
        std::visit(Overloaded{
                       [&](const ErrorStmt&) { line("error"); },
                       [&](const LetStmt& let) {
                           nested(
                               annotation("let " + pattern(let.pattern), type(let.type), let.type),
                               [&] { expr(let.value); });
                       },
                       [&](const ComprehensionStmt& comprehension) {
                           std::string header = "let " + name(comprehension.target) + "[";
                           for (std::size_t i = 0; i < comprehension.outputs.size(); ++i) {
                               const IndexOutput& output = comprehension.outputs[i];
                               header += i == 0 ? "" : ", ";
                               header += (output.is_pack ? "*" : "") + name(output.name);
                           }
                           nested(header + "]", [&] { expr(comprehension.value); });
                       },
                       [&](const VarStmt& var) {
                           nested(annotation("var " + name(var.name), type(var.type), var.type),
                                  [&] { expr(var.value); });
                       },
                       [&](const AssignStmt& assign) {
                           nested("assign " + name(assign.target), [&] { expr(assign.value); });
                       },
                       [&](const ReturnStmt& ret) {
                           if (ret.value == no_id) {
                               line("return");
                           } else {
                               nested("return", [&] { expr(ret.value); });
                           }
                       },
                       [&](const StaticForStmt& loop) {
                           nested("static for " + pattern(loop.pattern), [&] {
                               nested("in", [&] { expr(loop.iterable); });
                               body(loop.body);
                           });
                       },
                   },
                   ast_.stmt(id).data);
    }

    // ------------------------------------------------------------------- items

    void generics(const std::vector<GenericParam>& params) {
        for (const GenericParam& param : params) {
            std::string text = "generic ";
            text += param.is_pack ? "*" : "";
            text += name(param.name) + ": " + name(param.constraint_name);
            if (param.default_value.type != no_id) {
                text += " = " + type(param.default_value.type);
            } else if (param.default_value.expr != no_id) {
                text += " = " + inline_expr(param.default_value.expr);
            }
            line(text);
        }
    }

    void item(ItemId id) {
        const Item& node = ast_.item(id);
        const std::string visibility = node.is_pub ? "pub " : "";
        std::visit(Overloaded{
                       [&](const ErrorItem&) { line(visibility + "error"); },
                       [&](const ConstDecl& decl) {
                           nested(annotation(visibility + "const " + name(decl.name),
                                             type(decl.type),
                                             decl.type),
                                  [&] { expr(decl.value); });
                       },
                       [&](const TypeAliasDecl& decl) {
                           nested(visibility + "type " + name(decl.name), [&] {
                               generics(decl.generics);
                               line("= " + type(decl.type));
                           });
                       },
                       [&](const StructDecl& decl) {
                           nested(visibility + "struct " + name(decl.name), [&] {
                               generics(decl.generics);
                               for (const FieldDecl& field : decl.fields) {
                                   line("field " + name(field.name) + ": " + type(field.type));
                               }
                           });
                       },
                       [&](const EnumDecl& decl) {
                           nested(visibility + "enum " + name(decl.name), [&] {
                               generics(decl.generics);
                               for (const Name& variant : decl.variants) {
                                   line("variant " + name(variant));
                               }
                           });
                       },
                       [&](const FunctionDecl& decl) {
                           const std::string_view keyword = decl.kind == FunctionKind::Fn ? "fn "
                                                            : decl.kind == FunctionKind::Op
                                                                ? "op "
                                                                : "entry ";
                           nested(visibility + std::string(keyword) + name(decl.name), [&] {
                               generics(decl.generics);
                               for (const Parameter& param : decl.parameters) {
                                   const std::string header =
                                       "param " + name(param.name) + ": " + type(param.type);
                                   if (param.default_value == no_id) {
                                       line(header);
                                   } else {
                                       nested(header + " =", [&] { expr(param.default_value); });
                                   }
                               }
                               if (decl.return_type != no_id) {
                                   line("returns " + type(decl.return_type));
                               }
                               for (const ExprId constraint : decl.constraints) {
                                   nested("where", [&] { expr(constraint); });
                               }
                               body(decl.body);
                           });
                       },
                       [&](const BlockDecl& decl) {
                           nested(visibility + "block " + name(decl.name), [&] {
                               generics(decl.generics);
                               for (const ExprId constraint : decl.constraints) {
                                   nested("where", [&] { expr(constraint); });
                               }
                               for (const ItemId member : decl.members) {
                                   item(member);
                               }
                           });
                       },
                       [&](const MemberDecl& decl) {
                           const std::string keyword = std::string(member_keyword(decl.kind)) + " ";
                           const std::string header = visibility + std::string(keyword) +
                                                      name(decl.name) + ": " + type(decl.type);
                           if (decl.default_value == no_id) {
                               line(header);
                           } else {
                               nested(header + " =", [&] { expr(decl.default_value); });
                           }
                       },
                   },
                   node.data);
    }

    const Ast& ast_;
    const SourceManager& sources_;
    std::string out_;
    std::size_t indent_ = 0;
};

} // namespace

std::string dump(const Ast& ast, const SourceManager& sources) {
    return Dumper(ast, sources).run();
}

} // namespace linnet::ast
