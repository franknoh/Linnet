#pragma once

#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace linnet {

// Where a git dependency comes from: a repository, which commit of it (a
// tag, a branch, a revision, or the default branch), and the package's
// directory inside it.
struct GitSource {
    std::string url;       // what `git clone` is given
    std::string reference; // "tag=v1", "branch=main", "rev=4f2c...", or "" (the default branch)
    std::filesystem::path subdir;

    // The source as the lock file records it: the URL, then `#` and the
    // reference when there is one.
    std::string key() const;
};

// A dependency: a directory beside the package, or a git repository.
struct Dependency {
    std::filesystem::path path; // a path dependency's directory
    std::optional<GitSource> git;
};

// The contents of a `linnet.toml`:
//
//   [package]
//   name = "example-model"
//   version = "0.1.0"
//   language = "0.1"
//
//   [dependencies]
//   local = { path = "../local" }
//   layers = { github = "owner/repo", tag = "v0.2" }
//   cards = { hf = "owner/repo", subdir = "models/x" }
//   other = { git = "https://example.com/repo.git", rev = "4f2c..." }
//
// Paths are relative to the manifest's directory. A git dependency names one
// of `tag`, `branch` or `rev` (or none: the default branch), and `subdir`
// when the package is not at the repository's root. Reading a manifest never
// runs anything.
struct PackageManifest {
    std::string name;
    std::string version;
    std::string language;
    std::map<std::string, Dependency> dependencies; // key -> where it comes from
};

inline constexpr const char* supported_language_version = "0.1";

// A key that can name a dependency: identifier characters only, and not one
// of the reserved import roots `std` and `crate`.
bool is_valid_package_name(std::string_view name);

// The host of a git URL (`github.com` for `https://github.com/a/b.git` and
// `git@github.com:a/b.git`), lowercased; empty for a local `file://` URL.
std::string git_host(std::string_view url);

// Whether a git URL is on GitHub or the Hugging Face Hub, the hosts a
// dependency comes from without a warning.
bool is_known_git_host(std::string_view url);

std::expected<PackageManifest, std::string> parse_manifest(std::string_view text,
                                                           const std::filesystem::path& directory);
std::expected<PackageManifest, std::string> read_manifest(const std::filesystem::path& path);

// Text of a fresh manifest for `linnet init`.
std::string default_manifest(std::string_view name);

} // namespace linnet
