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
        walk_manifest(*model_, types_, entity, [&](const ManifestTensor& found) {
            ManifestEntry entry;
            entry.path = found.path;
            entry.kind = found.member->is_state    ? "state"
                         : found.member->is_buffer ? "buffer"
                                                   : "param";
            for (const shape::Poly& length : *found.repeat) {
                entry.repeat.push_back(dims_.to_string(length));
            }
            entry.is_optional = found.is_optional;
            entry.dtype = types_.to_string(found.tensor->dtype);
            for (const ShapeElem& unit : found.tensor->shape) {
                entry.shape.push_back(unit.is_pack ? "*" + std::string(dims_.symbol_name(unit.pack))
                                                   : dims_.to_string(unit.dim));
            }
            manifest.entries.push_back(std::move(entry));
        });
        result_.manifests.push_back(std::move(manifest));
    }
}

namespace {

struct ManifestWalk {
    const Model& model;
    TypeStore& types;
    const std::function<void(const ManifestTensor&)>& visit;
    std::vector<EntityId> active;
    std::vector<shape::Poly> repeat;

    void block(EntityId block,
               const Substitution& substitution,
               const std::string& prefix,
               bool within_optional) {
        if (std::ranges::find(active, block) != active.end()) {
            return;
        }
        active.push_back(block);
        for (const auto& [entity, name] : block_members(model, block)) {
            const Entity& member = model.entities[entity];
            if (member.type == no_type) {
                continue;
            }
            const TypeData& data = types.get(types.substitute(member.type, substitution));
            const std::string path = prefix + std::string(name);
            // An optional `sub` is present or absent as a whole: everything
            // in it is optional.
            const bool is_optional = data.kind == TypeKind::Optional;
            const TypeData& inner = is_optional ? types.get(data.elements.front()) : data;
            if (data.kind == TypeKind::Array) {
                const TypeData& element = types.get(data.elements.front());
                if (element.kind == TypeKind::Block) {
                    repeat.push_back(data.value);
                    this->block(element.decl,
                                substitution_of(model.decls.at(element.decl), element),
                                path + "[*].",
                                within_optional);
                    repeat.pop_back();
                }
            } else if (inner.kind == TypeKind::Block) {
                this->block(inner.decl,
                            substitution_of(model.decls.at(inner.decl), inner),
                            path + ".",
                            within_optional || is_optional);
            } else if (inner.kind == TypeKind::Tensor) {
                visit({path, &member, &inner, &repeat, within_optional || is_optional});
            }
        }
        active.pop_back();
    }
};

} // namespace

std::vector<std::pair<EntityId, std::string_view>> block_members(const Model& model,
                                                                 EntityId block) {
    // The scope is in hash order; members were declared in id order.
    std::vector<std::pair<EntityId, std::string_view>> members;
    for (const auto& [name, entity] : model.decls.at(block).scope) {
        if (model.entities[entity].kind == EntityKind::Member) {
            members.emplace_back(entity, name);
        }
    }
    std::ranges::sort(members, {}, &std::pair<EntityId, std::string_view>::first);
    return members;
}

void walk_manifest(const Model& model,
                   TypeStore& types,
                   EntityId root,
                   const std::function<void(const ManifestTensor&)>& visit) {
    ManifestWalk{model, types, visit, {}, {}}.block(root, {}, "", false);
}

} // namespace linnet::sema
