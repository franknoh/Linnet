#include "checker.hpp"

#include <map>

namespace linnet::sema {

namespace {

std::string_view keyword_of(const ast::FunctionDecl& decl) {
    switch (decl.kind) {
    case ast::FunctionKind::Fn:
        return "fn";
    case ast::FunctionKind::Op:
        return "op";
    case ast::FunctionKind::Entry:
        return "entry";
    }
    return "fn";
}

} // namespace

std::string Checker::describe(EntityId id) {
    const Entity& entity = entities_[id];
    std::string name(entity.name);
    const auto generics_text = [&](const DeclInfo& info) {
        std::string text;
        for (const GenericInfo& generic : info.generics) {
            text += text.empty() ? "<" : ", ";
            text += generic.kind == GenericKind::Pack ? "*" : "";
            text += generic.name;
            text += generic.kind == GenericKind::Dim            ? ": Dim"
                    : generic.kind == GenericKind::Pack         ? ": Shape"
                    : generic.constraint == DTypeClass::Float   ? ": Float"
                    : generic.constraint == DTypeClass::Integer ? ": Integer"
                    : generic.constraint == DTypeClass::Numeric ? ": Numeric"
                                                                : ": DType";
        }
        return text.empty() ? text : text + ">";
    };
    const auto type_text = [&](TypeId type) {
        return type == no_type ? std::string("?") : types_.to_string(type);
    };

    switch (entity.kind) {
    case EntityKind::Module: {
        std::string path;
        for (const ast::Name& segment : modules_[entity.module_ref]->module_path) {
            path += path.empty() ? "" : ".";
            path += segment.text;
        }
        return "module " + path;
    }
    case EntityKind::Const:
        return "const " + name + ": " + type_text(entity.type);
    case EntityKind::TypeAlias:
        return "type " + name + generics_text(decls_[id]) + " = " + type_text(decls_[id].aliased);
    case EntityKind::Struct:
        return "struct " + name + generics_text(decls_[id]);
    case EntityKind::Enum:
        return "enum " + name + generics_text(decls_[id]);
    case EntityKind::Block:
        return "block " + name + generics_text(decls_[id]);
    case EntityKind::Function: {
        const auto& decl =
            std::get<ast::FunctionDecl>(modules_[entity.module]->item(entity.item).data);
        const DeclInfo& info = decls_[id];
        std::string text = std::string(keyword_of(decl)) + " " + name + generics_text(info) + "(";
        for (std::size_t i = 0; i < info.params.size(); ++i) {
            text += i == 0 ? "" : ", ";
            text += std::string(info.params[i].name) + ": " + type_text(info.params[i].type);
            text += info.params[i].has_default ? " = ..." : "";
        }
        text += ")";
        if (info.result != no_type && types_.kind(info.result) != TypeKind::Unit) {
            text += " -> " + type_text(info.result);
        }
        return text;
    }
    case EntityKind::Member: {
        const auto& decl =
            std::get<ast::MemberDecl>(modules_[entity.module]->item(entity.item).data);
        const std::string_view keyword = decl.kind == ast::MemberKind::Param    ? "param"
                                         : decl.kind == ast::MemberKind::Buffer ? "buffer"
                                                                                : "sub";
        return std::string(keyword) + " " + name + ": " + type_text(entity.type);
    }
    case EntityKind::GenericDim:
        return name + ": Dim";
    case EntityKind::GenericPack:
        return "*" + name + ": Shape";
    case EntityKind::GenericDType: {
        const DTypeClass constraint = types_.dtype_var(entity.dtype_var).constraint;
        return name + (constraint == DTypeClass::Float     ? ": Float"
                       : constraint == DTypeClass::Integer ? ": Integer"
                       : constraint == DTypeClass::Numeric ? ": Numeric"
                                                           : ": DType");
    }
    case EntityKind::Local:
        return std::string(entity.is_parameter ? ""
                           : entity.is_mutable ? "var "
                                               : "let ") +
               name + ": " + type_text(entity.type);
    }
    return name;
}

void Checker::collect_symbols() {
    std::map<EntityId, std::uint32_t> index_of;
    const auto symbol_index = [&](EntityId id) {
        const auto found = index_of.find(id);
        if (found != index_of.end()) {
            return found->second;
        }
        const Entity& entity = entities_[id];
        SymbolInfo symbol;
        symbol.name = std::string(entity.name);
        symbol.span = entity.span;
        symbol.detail = describe(id);
        switch (entity.kind) {
        case EntityKind::Module:
            symbol.kind = SymbolKind::Module;
            break;
        case EntityKind::Const:
            symbol.kind = SymbolKind::Const;
            break;
        case EntityKind::TypeAlias:
            symbol.kind = SymbolKind::TypeAlias;
            break;
        case EntityKind::Struct:
            symbol.kind = SymbolKind::Struct;
            break;
        case EntityKind::Enum:
            symbol.kind = SymbolKind::Enum;
            break;
        case EntityKind::Function: {
            const auto& decl =
                std::get<ast::FunctionDecl>(modules_[entity.module]->item(entity.item).data);
            symbol.kind = decl.kind == ast::FunctionKind::Fn   ? SymbolKind::Function
                          : decl.kind == ast::FunctionKind::Op ? SymbolKind::Op
                                                               : SymbolKind::Entry;
            break;
        }
        case EntityKind::Block:
            symbol.kind = SymbolKind::Block;
            break;
        case EntityKind::Member: {
            const auto& decl =
                std::get<ast::MemberDecl>(modules_[entity.module]->item(entity.item).data);
            symbol.kind = decl.kind == ast::MemberKind::Param    ? SymbolKind::Param
                          : decl.kind == ast::MemberKind::Buffer ? SymbolKind::Buffer
                                                                 : SymbolKind::Sub;
            break;
        }
        case EntityKind::GenericDim:
            symbol.kind = SymbolKind::GenericDim;
            break;
        case EntityKind::GenericPack:
            symbol.kind = SymbolKind::GenericPack;
            break;
        case EntityKind::GenericDType:
            symbol.kind = SymbolKind::GenericDType;
            break;
        case EntityKind::Local:
            symbol.kind = entity.is_parameter ? SymbolKind::Parameter : SymbolKind::Local;
            break;
        }
        result_.symbols.push_back(std::move(symbol));
        const auto index = static_cast<std::uint32_t>(result_.symbols.size() - 1);
        index_of.emplace(id, index);
        return index;
    };

    for (EntityId id = 0; id < entities_.size(); ++id) {
        if (!entities_[id].name.empty() && entities_[id].span.file != invalid_file_id) {
            symbol_index(id);
        }
    }
    for (const auto& [span, entity] : refs_) {
        if (index_of.contains(entity)) {
            result_.references.push_back({span, index_of.at(entity)});
        }
    }
}

} // namespace linnet::sema

namespace linnet::sema {

namespace {
const ExprFacts empty_expr_facts;
const StmtFacts empty_stmt_facts;
} // namespace

const ExprFacts& Model::expr(std::uint32_t module, ast::ExprId id) const {
    const auto found = exprs[module].find(id);
    return found == exprs[module].end() ? empty_expr_facts : found->second;
}

const StmtFacts& Model::stmt(std::uint32_t module, ast::StmtId id) const {
    const auto found = stmts[module].find(id);
    return found == stmts[module].end() ? empty_stmt_facts : found->second;
}

} // namespace linnet::sema
