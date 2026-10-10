#pragma once

#include "linnet/package/manifest.hpp"

#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>

namespace linnet {

// Where Linnet keeps what it downloads: `$LINNET_HOME`, else `~/.linnet`.
std::filesystem::path linnet_home();

// A git repository's place under `<home>/git`, named as the Hugging Face
// cache names a repository: `github.com--owner--repo` for
// `https://github.com/owner/repo.git`. It holds `db`, a bare clone, and
// `snapshots/<commit>`, the files of each commit used.
std::filesystem::path repository_directory(std::string_view url);

// The directory a git dependency's package is in, once `commit` is checked out.
std::filesystem::path package_directory(const GitSource& source, std::string_view commit);

// `linnet.lock`: the commit each git dependency resolved to, by source
// (`GitSource::key`). Written beside the root package's `linnet.toml`, so
// that every build of the package reads the same files.
using Lockfile = std::map<std::string, std::string>;

std::expected<Lockfile, std::string> read_lockfile(const std::filesystem::path& path);
std::string lockfile_text(const Lockfile& lock);

struct FetchOptions {
    bool update = false;    // resolve every reference again, ignoring the lock
    bool offline = false;   // use only what the cache holds
    bool write_lock = true; // record what was resolved in linnet.lock
    // Progress and warnings, a line each.
    std::function<void(std::string_view)> note;
};

// Every git dependency reachable from the package at `root` (through path
// and git dependencies alike), checked out under `linnet_home()` and
// recorded in `<root>/linnet.lock`. Returns each source's checkout
// directory, by `GitSource::key`. A dependency not on GitHub or the Hugging
// Face Hub is fetched with a warning. Nothing fetched is ever run.
std::expected<std::map<std::string, std::filesystem::path>, std::string>
fetch_dependencies(const std::filesystem::path& root, const FetchOptions& options);

// Whether the package at `root` reaches any git dependency.
bool has_git_dependencies(const std::filesystem::path& root);

} // namespace linnet
