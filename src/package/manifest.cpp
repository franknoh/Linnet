#include "linnet/package/manifest.hpp"

#include "linnet/package/toml.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

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

namespace {

bool is_repository_word(std::string_view word) {
    return !word.empty() && word != "." && word != ".." &&
           std::all_of(word.begin(), word.end(), [](char c) {
               return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '-' || c == '_' ||
                      c == '.';
           });
}

// `owner/repo`, or for the Hugging Face Hub `datasets/owner/repo` and
// `spaces/owner/repo`.
bool is_repository(std::string_view name, bool hub) {
    std::vector<std::string_view> words;
    std::size_t start = 0;
    for (std::size_t slash = name.find('/'); slash != std::string_view::npos;
         slash = name.find('/', start)) {
        words.push_back(name.substr(start, slash - start));
        start = slash + 1;
    }
    words.push_back(name.substr(start));
    const bool kind = words.size() == 3 && hub && (words[0] == "datasets" || words[0] == "spaces");
    return (words.size() == 2 || kind) &&
           std::all_of(words.begin(), words.end(), is_repository_word);
}

// What `git clone` may be given: an https, http, ssh or file URL, or the
// `git@host:path` form; never one of git's command-running transports.
bool is_git_url(std::string_view url) {
    for (const std::string_view scheme : {"https://", "http://", "ssh://", "file://"}) {
        if (url.starts_with(scheme) && url.size() > scheme.size()) {
            return true;
        }
    }
    const std::size_t at = url.find('@');
    const std::size_t colon = url.find(':');
    return at != std::string_view::npos && colon != std::string_view::npos && at < colon &&
           !url.starts_with("-") && url.find("::") == std::string_view::npos;
}

// A tag, branch or revision: something `git` reads as a name, never an option.
bool is_reference(std::string_view value) {
    return !value.empty() && !value.starts_with("-") &&
           std::none_of(value.begin(), value.end(), [](char c) {
               return std::isspace(static_cast<unsigned char>(c)) != 0 ||
                      std::iscntrl(static_cast<unsigned char>(c)) != 0;
           });
}

std::expected<Dependency, std::string> parse_dependency(const std::string& key,
                                                        const toml::Value& value,
                                                        const std::filesystem::path& directory) {
    const toml::Table* table = value.as_table();
    const std::string usage = "dependency `" + key +
                              "` must be `{ path = \"...\" }`, `{ github = \"owner/repo\" }`, "
                              "`{ hf = \"owner/repo\" }` or `{ git = \"<url>\" }`";
    if (table == nullptr) {
        return std::unexpected(usage);
    }
    const auto text = [&](const char* name) -> std::optional<std::string> {
        const toml::Value* found = value.find(name);
        return found != nullptr && found->as_string() != nullptr
                   ? std::optional(*found->as_string())
                   : std::nullopt;
    };
    for (const auto& [name, field] : *table) {
        if (field.as_string() == nullptr) {
            std::string message = "dependency `" + key;
            message += "`: `";
            message += name;
            message += "` must be a string";
            return std::unexpected(message);
        }
    }
    Dependency dependency;
    if (const auto path = text("path")) {
        if (table->size() != 1) {
            return std::unexpected("dependency `" + key +
                                   "`: a path dependency takes `path` alone");
        }
        dependency.path = directory / *path;
        return dependency;
    }
    GitSource source;
    std::size_t sources = 0;
    if (const auto github = text("github")) {
        if (!is_repository(*github, false)) {
            return std::unexpected("dependency `" + key + "`: `github` is `owner/repo`");
        }
        source.url = "https://github.com/" + *github + ".git";
        ++sources;
    }
    if (const auto hub = text("hf")) {
        if (!is_repository(*hub, true)) {
            return std::unexpected("dependency `" + key +
                                   "`: `hf` is `owner/repo` (or `datasets/...`, `spaces/...`)");
        }
        source.url = "https://huggingface.co/" + *hub;
        ++sources;
    }
    if (const auto url = text("git")) {
        if (!is_git_url(*url)) {
            return std::unexpected("dependency `" + key +
                                   "`: `git` must be an https, ssh or file URL");
        }
        source.url = *url;
        ++sources;
    }
    if (sources != 1) {
        return std::unexpected(usage);
    }
    std::size_t references = 0;
    for (const char* kind : {"tag", "branch", "rev"}) {
        if (const auto reference = text(kind)) {
            if (!is_reference(*reference)) {
                return std::unexpected("dependency `" + key + "`: `" + kind + "` is not a name");
            }
            source.reference = std::string(kind) + "=" + *reference;
            ++references;
        }
    }
    if (references > 1) {
        return std::unexpected("dependency `" + key + "` takes one of `tag`, `branch` and `rev`");
    }
    if (const auto subdir = text("subdir")) {
        const std::filesystem::path relative(*subdir);
        if (relative.is_absolute() || relative.has_root_name() ||
            std::any_of(relative.begin(), relative.end(), [](const std::filesystem::path& part) {
                return part == "..";
            })) {
            return std::unexpected("dependency `" + key +
                                   "`: `subdir` must stay inside the repository");
        }
        source.subdir = relative;
    }
    const std::size_t known = 1 + references + (source.subdir.empty() ? 0 : 1);
    if (table->size() != known) {
        return std::unexpected("dependency `" + key +
                               "` takes a source and only `tag`, `branch`, `rev` or `subdir`");
    }
    dependency.git = std::move(source);
    return dependency;
}

} // namespace

// One commit serves every directory of it, so the key leaves `subdir` out.
std::string GitSource::key() const {
    return reference.empty() ? url : url + "#" + reference;
}

std::string git_host(std::string_view url) {
    std::string_view rest = url;
    if (const std::size_t scheme = rest.find("://"); scheme != std::string_view::npos) {
        if (rest.substr(0, scheme) == "file") {
            return {};
        }
        rest.remove_prefix(scheme + 3);
        if (const std::size_t at = rest.find('@');
            at != std::string_view::npos && at < rest.find('/')) {
            rest.remove_prefix(at + 1);
        }
        rest = rest.substr(0, rest.find_first_of("/:"));
    } else {
        // `user@host:path`
        rest.remove_prefix(rest.find('@') + 1);
        rest = rest.substr(0, rest.find(':'));
    }
    std::string host(rest);
    std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return host;
}

bool is_known_git_host(std::string_view url) {
    const std::string host = git_host(url);
    return host == "github.com" || host == "huggingface.co" || host == "hf.co";
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
            auto dependency = parse_dependency(key, value, directory);
            if (!dependency) {
                return std::unexpected(dependency.error());
            }
            manifest.dependencies[key] = std::move(*dependency);
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
