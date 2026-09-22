#include "checker.hpp"

#include <algorithm>

namespace linnet::sema {

void Checker::collect_manifests() {
    for (EntityId entity = 0; entity < entities_.size(); ++entity) {
        const Entity& block = entities_[entity];
        if (block.kind != EntityKind::Block || block.parent != no_entity) {
            continue;
        }
        ManifestBlock manifest;
        manifest.name = std::string(block.name);
        for (const ast::Name& segment : modules_[block.module]->module_path) {
            manifest.module += manifest.module.empty() ? "" : ".";
            manifest.module += segment.text;
        }
        for (const GenericInfo& generic : decls_[entity].generics) {
            manifest.generics.push_back((generic.kind == GenericKind::Pack ? "*" : "") +
                                        std::string(generic.name));
        }
        std::vector<EntityId> active;
        collect_manifest(entity, {}, "", {}, active, manifest);
        result_.manifests.push_back(std::move(manifest));
    }
}

void Checker::collect_manifest(EntityId block,
                               const Substitution& substitution,
                               const std::string& prefix,
                               const std::vector<std::string>& repeat,
                               std::vector<EntityId>& active,
                               ManifestBlock& out) {
    // A block that contains itself, directly or through an array, has no
    // finite manifest; the cycle is already an error elsewhere.
    if (std::find(active.begin(), active.end(), block) != active.end()) {
        return;
    }
    active.push_back(block);
    const Entity& declaration = entities_[block];
    const auto& decl =
        std::get<ast::BlockDecl>(modules_[declaration.module]->item(declaration.item).data);
    for (const ast::ItemId item_id : decl.members) {
        const ast::Item& item = modules_[declaration.module]->item(item_id);
        const auto* member = std::get_if<ast::MemberDecl>(&item.data);
        if (member == nullptr) {
            continue;
        }
        const Scope& scope = decls_[block].scope;
        const auto found = scope.find(member->name.text);
        if (found == scope.end() || entities_[found->second].type == no_type) {
            continue;
        }
        const TypeId type = types_.substitute(entities_[found->second].type, substitution);
        const TypeData& data = types_.get(type);
        const std::string path = prefix + std::string(member->name.text);

        if (member->kind == ast::MemberKind::Sub) {
            std::vector<std::string> inner_repeat = repeat;
            std::string inner_prefix = path;
            const TypeData* element = &data;
            if (data.kind == TypeKind::Array) {
                inner_repeat.push_back(dims_.to_string(data.value));
                inner_prefix += "[*]";
                element = &types_.get(data.elements.front());
            }
            if (element->kind == TypeKind::Block) {
                collect_manifest(element->decl,
                                 substitution_of(*element),
                                 inner_prefix + ".",
                                 inner_repeat,
                                 active,
                                 out);
            }
            continue;
        }

        ManifestEntry entry;
        entry.path = path;
        entry.kind = member->kind == ast::MemberKind::Param ? "param" : "buffer";
        entry.repeat = repeat;
        entry.is_optional = data.kind == TypeKind::Optional;
        const TypeData& tensor = entry.is_optional ? types_.get(data.elements.front()) : data;
        if (tensor.kind != TypeKind::Tensor) {
            continue;
        }
        entry.dtype = types_.to_string(tensor.dtype);
        for (const ShapeElem& unit : tensor.shape) {
            entry.shape.push_back(unit.is_pack ? "*" + std::string(dims_.symbol_name(unit.pack))
                                               : dims_.to_string(unit.dim));
        }
        out.entries.push_back(std::move(entry));
    }
    active.pop_back();
}

} // namespace linnet::sema
