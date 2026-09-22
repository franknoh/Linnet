#include "linnet/package/manifest.hpp"

#include "linnet/package/toml.hpp"

#include <fstream>
#include <iterator>
#include <optional>
#include <utility>

namespace linnet {

bool is_valid_package_name(std::string_view name) {
    if (name.empty() || name == "std" || name == "crate") {
        return false;
    }
    const char first = name.front();
    if (!((first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') || first == '_')) {
        return false;
    }
    for (const char c : name) {
        const bool is_word =
            (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if (!is_word) {
            return false;
        }
    }
    return true;
}

std::expected<PackageManifest, std::string> parse_manifest(std::string_view text,
                                                           const std::filesystem::path& directory) {
    const auto document = toml::parse(text);
    if (!document) {
        return std::unexpected("line " + std::to_string(document.error().line) + ": " +
                               document.error().message);
    }
    const toml::Value root{*document, 0};
    const toml::Value* package = root.find("package");
    if (package == nullptr || package->as_table() == nullptr) {
        return std::unexpected("missing [package] table");
    }
    PackageManifest manifest;
    const auto required = [&](const char* key, std::string& out) -> std::optional<std::string> {
        const toml::Value* value = package->find(key);
        if (value == nullptr || value->as_string() == nullptr) {
            return std::string("[package] needs a `") + key + "` string";
        }
        out = *value->as_string();
        return std::nullopt;
    };
    for (const auto& [key, out] : {std::pair{"name", &manifest.name},
                                   std::pair{"version", &manifest.version},
                                   std::pair{"language", &manifest.language}}) {
        if (auto problem = required(key, *out)) {
            return std::unexpected(*problem);
        }
    }
    if (manifest.name.empty()) {
        return std::unexpected("[package] name must not be empty");
    }
    if (manifest.language != supported_language_version) {
        return std::unexpected("language version `" + manifest.language +
                               "` is not supported; this toolchain implements `" +
                               supported_language_version + "`");
    }

    if (const toml::Value* dependencies = root.find("dependencies")) {
        if (dependencies->as_table() == nullptr) {
            return std::unexpected("[dependencies] must be a table");
        }
        for (const auto& [key, value] : *dependencies->as_table()) {
            if (!is_valid_package_name(key)) {
                return std::unexpected("`" + key +
                                       "` cannot name a dependency; use identifier "
                                       "characters, and not `std` or `crate`");
            }
            const toml::Value* path = value.find("path");
            if (value.as_table() == nullptr || path == nullptr || path->as_string() == nullptr ||
                value.as_table()->size() != 1) {
                return std::unexpected("dependency `" + key +
                                       "` must be `{ path = \"...\" }`; other sources are not "
                                       "supported yet");
            }
            manifest.dependencies[key] = directory / *path->as_string();
        }
    }
    return manifest;
}

std::expected<PackageManifest, std::string> read_manifest(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::unexpected(path.generic_string() + ": cannot open file");
    }
    const std::string text{std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>()};
    auto manifest = parse_manifest(text, path.parent_path());
    if (!manifest) {
        return std::unexpected(path.generic_string() + ": " + manifest.error());
    }
    return manifest;
}

std::string default_manifest(std::string_view name) {
    return "[package]\nname = \"" + std::string(name) + "\"\nversion = \"0.1.0\"\nlanguage = \"" +
           supported_language_version + "\"\n\n[dependencies]\n";
}

} // namespace linnet
