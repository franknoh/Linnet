#pragma once

#include <expected>
#include <filesystem>
#include <map>
#include <string>

namespace linnet {

// The contents of a `linnet.toml`:
//
//   [package]
//   name = "example-model"
//   version = "0.1.0"
//   language = "0.1"
//
//   [dependencies]
//   foo = { path = "../foo" }
//
// Only path dependencies exist so far. Paths are relative to the manifest's
// directory. Reading a manifest never runs anything.
struct PackageManifest {
    std::string name;
    std::string version;
    std::string language;
    std::map<std::string, std::filesystem::path> dependencies; // key -> package directory
};

inline constexpr const char* supported_language_version = "0.1";

// A key that can name a dependency: identifier characters only, and not one
// of the reserved import roots `std` and `crate`.
bool is_valid_package_name(std::string_view name);

std::expected<PackageManifest, std::string> parse_manifest(std::string_view text,
                                                           const std::filesystem::path& directory);
std::expected<PackageManifest, std::string> read_manifest(const std::filesystem::path& path);

// Text of a fresh manifest for `linnet init`.
std::string default_manifest(std::string_view name);

} // namespace linnet
