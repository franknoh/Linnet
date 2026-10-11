#include "linnet/package/fetch.hpp"

#include "linnet/package/toml.hpp"
#include "linnet/support/files.hpp"
#include "linnet/support/text.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <random>
#include <set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#include <stdlib.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#ifdef __APPLE__
#include <crt_externs.h>
#else
#include <unistd.h>
#endif
#endif

namespace linnet {

namespace fs = std::filesystem;

namespace {

using Environment = std::vector<std::pair<std::string, std::string>>;

char** inherited_environment() {
#ifdef _WIN32
    return _environ;
#elif defined(__APPLE__)
    return *_NSGetEnviron();
#else
    return environ;
#endif
}

// Runs `arguments` (the program first, found on PATH) with `extra` set in
// its environment, as a program of its own: no shell reads the words, so a
// URL or a name from a manifest is never a command. Returns the exit status,
// or -1 when the program could not run.
int run(const std::vector<std::string>& arguments, const Environment& extra) {
    std::vector<std::string> environment;
    std::set<std::string> replaced;
    for (const auto& [name, value] : extra) {
        std::string entry = name;
        entry += "=";
        entry += value;
        environment.push_back(std::move(entry));
        replaced.insert(name);
    }
    for (char* const* entry = inherited_environment(); entry != nullptr && *entry != nullptr;
         ++entry) {
        const std::string text(*entry);
        if (!replaced.contains(text.substr(0, text.find('=')))) {
            environment.push_back(text);
        }
    }
    std::vector<std::string> words;
    words.reserve(arguments.size());
    for (const std::string& argument : arguments) {
#ifdef _WIN32
        words.push_back(command_line_word(argument));
#else
        words.push_back(argument);
#endif
    }
    std::vector<char*> argv;
    argv.reserve(words.size() + 1);
    for (std::string& word : words) {
        argv.push_back(word.data());
    }
    argv.push_back(nullptr);
    std::vector<char*> envp;
    envp.reserve(environment.size() + 1);
    for (std::string& entry : environment) {
        envp.push_back(entry.data());
    }
    envp.push_back(nullptr);
#ifdef _WIN32
    const intptr_t status = _spawnvpe(_P_WAIT, arguments.front().c_str(), argv.data(), envp.data());
    return status == -1 ? -1 : static_cast<int>(status);
#else
    pid_t child = 0;
    if (posix_spawnp(&child, argv.front(), nullptr, nullptr, argv.data(), envp.data()) != 0) {
        return -1;
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) {
            return -1;
        }
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

// `git` with what every call here needs: no prompt for credentials (a
// missing repository fails instead of waiting), large files left as their
// Git LFS pointers (a package needs its `.linnet` files, not a checkpoint),
// and no transport that runs commands.
int git(std::vector<std::string> arguments, Environment extra = {}) {
    arguments.insert(arguments.begin(),
                     {"git", "-c", "protocol.ext.allow=never", "-c", "core.autocrlf=false"});
    extra.emplace_back("GIT_TERMINAL_PROMPT", "0");
    extra.emplace_back("GIT_LFS_SKIP_SMUDGE", "1");
    return run(arguments, extra);
}

std::string random_suffix() {
    std::random_device device;
    return std::to_string(device()) + std::to_string(device());
}

std::string trimmed(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.pop_back();
    }
    return text;
}

// The commit `revision` names in the bare clone `db`, if it names one.
std::optional<std::string> commit_of(const fs::path& db, const std::string& revision) {
    const fs::path output = db / ("resolved-" + random_suffix());
    const int status = git({"--git-dir=" + db.string(),
                            "log",
                            "-1",
                            "--format=%H",
                            "--output=" + output.string(),
                            "--end-of-options",
                            revision});
    std::optional<std::string> text = read_file(output);
    std::error_code error;
    fs::remove(output, error);
    if (status != 0 || !text) {
        return std::nullopt;
    }
    std::string commit = trimmed(*text);
    return commit.size() == 40 ? std::optional(commit) : std::nullopt;
}

// What a source's reference names in a clone: a tag or a branch as it is
// there, a revision as given, the default branch as HEAD.
std::string revision_of(const GitSource& source) {
    const std::string& reference = source.reference;
    if (reference.starts_with("tag=")) {
        return "refs/tags/" + reference.substr(4);
    }
    if (reference.starts_with("branch=")) {
        return "refs/heads/" + reference.substr(7);
    }
    if (reference.starts_with("rev=")) {
        return reference.substr(4);
    }
    return "HEAD";
}

class Fetcher {
public:
    Fetcher(const FetchOptions& options, Lockfile locked)
        : options_(options), locked_(std::move(locked)) {}

    // The commit a source resolves to, checked out in the cache.
    std::expected<std::string, std::string> resolve(const GitSource& source) {
        const std::string key = source.key();
        if (const auto found = resolved_.find(key); found != resolved_.end()) {
            return found->second;
        }
        const fs::path repository = repository_directory(source.url);
        const fs::path db = repository / "db";
        std::optional<std::string> commit;
        if (const auto found = locked_.find(key); found != locked_.end() && !options_.update) {
            commit = found->second;
        }
        if (commit && fs::is_directory(repository / "snapshots" / *commit)) {
            resolved_[key] = *commit;
            return *commit;
        }
        if (options_.offline) {
            return std::unexpected("`" + key + "` is not in " + repository.generic_string() +
                                   "; run `linnet fetch` online (not with --offline or "
                                   "LINNET_OFFLINE=1)");
        }
        auto cloned = clone(source.url, db);
        if (!cloned) {
            return std::unexpected(cloned.error());
        }
        if (commit) {
            if (!commit_of(db, *commit) && !refresh(source.url, db, *commit)) {
                return std::unexpected("commit " + *commit + " of `" + source.url +
                                       "` (in linnet.lock) is not in the repository; "
                                       "`linnet update` resolves it again");
            }
        } else {
            if (!*cloned && !refresh(source.url, db, {})) {
                return std::unexpected("cannot fetch `" + source.url + "`");
            }
            const std::string revision = revision_of(source);
            commit = commit_of(db, revision);
            if (!commit && source.reference.starts_with("rev=") &&
                refresh(source.url, db, revision)) {
                commit = commit_of(db, revision);
            }
            if (!commit) {
                return std::unexpected("`" + key + "` names no commit of the repository");
            }
        }
        auto checked_out = check_out(repository, db, *commit);
        if (!checked_out) {
            return std::unexpected(checked_out.error());
        }
        resolved_[key] = *commit;
        return *commit;
    }

    const Lockfile& resolved() const { return resolved_; }

private:
    void note(const std::string& line) const {
        if (options_.note) {
            options_.note(line);
        }
    }

    // Before reaching a repository over the network: a warning, once, when
    // it is not on GitHub or the Hugging Face Hub.
    void reach(const std::string& url) {
        if (!is_known_git_host(url) && warned_.insert(url).second) {
            note("warning: `" + url +
                 "` is not on GitHub or the Hugging Face Hub; Linnet reads only its .linnet files");
        }
    }

    // A bare clone of `url` at `db`, made when there is none; true when it
    // was made now (and so holds every branch and tag already).
    std::expected<bool, std::string> clone(const std::string& url, const fs::path& db) {
        std::error_code error;
        if (fs::is_directory(db, error)) {
            return false;
        }
        fs::create_directories(db.parent_path(), error);
        const fs::path staging = db.parent_path() / ("db-" + random_suffix());
        reach(url);
        note("fetching " + url);
        if (git({"clone", "--bare", "--quiet", "--", url, staging.string()}) != 0) {
            fs::remove_all(staging, error);
            return std::unexpected("cannot clone `" + url + "`");
        }
        fs::rename(staging, db, error);
        if (error) {
            fs::remove_all(staging, error); // another process made it first
        }
        return true;
    }

    // Every branch and tag of `url` again, or one revision when given.
    bool refresh(const std::string& url, const fs::path& db, const std::string& revision) {
        reach(url);
        note("updating " + url);
        std::vector<std::string> arguments = {
            "--git-dir=" + db.string(), "fetch", "--quiet", "--force", "--tags", "--", url};
        if (revision.empty()) {
            arguments.emplace_back("+refs/heads/*:refs/heads/*");
        } else {
            arguments.push_back(revision);
        }
        return git(arguments) == 0;
    }

    // `commit`'s files at `<repository>/snapshots/<commit>`, written to a
    // staging directory first so that a snapshot is never half written.
    static std::expected<void, std::string>
    check_out(const fs::path& repository, const fs::path& db, const std::string& commit) {
        const fs::path snapshot = repository / "snapshots" / commit;
        std::error_code error;
        if (fs::is_directory(snapshot, error)) {
            return {};
        }
        const fs::path staging = repository / "snapshots" / (".staging-" + random_suffix());
        const fs::path tree = staging / "tree";
        fs::create_directories(tree, error);
        const int status = git({"-c",
                                "core.bare=false",
                                "--git-dir=" + db.string(),
                                "--work-tree=" + tree.string(),
                                "checkout",
                                "--quiet",
                                "--force",
                                commit,
                                "--",
                                "."},
                               {{"GIT_INDEX_FILE", (staging / "index").string()}});
        if (status == 0) {
            fs::rename(tree, snapshot, error);
        }
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        if (status != 0 || (error && !fs::is_directory(snapshot, ignored))) {
            return std::unexpected("cannot check out " + commit + " in " + db.generic_string());
        }
        return {};
    }

    const FetchOptions& options_;
    Lockfile locked_;
    Lockfile resolved_;
    std::set<std::string> warned_;
};

std::string toml_string(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out + "\"";
}

} // namespace

fs::path linnet_home() {
    if (const char* chosen = std::getenv("LINNET_HOME"); chosen != nullptr && *chosen != '\0') {
        return chosen;
    }
#ifdef _WIN32
    const char* user = std::getenv("USERPROFILE");
#else
    const char* user = std::getenv("HOME");
#endif
    return fs::path(user != nullptr ? user : ".") / ".linnet";
}

fs::path repository_directory(std::string_view url) {
    std::string_view rest = url;
    std::string name;
    if (const std::size_t scheme = rest.find("://"); scheme != std::string_view::npos) {
        if (rest.substr(0, scheme) == "file") {
            name = "file";
        }
        rest.remove_prefix(scheme + 3);
    }
    if (const std::size_t at = rest.find('@');
        at != std::string_view::npos && at < rest.find('/')) {
        rest.remove_prefix(at + 1);
    }
    if (rest.ends_with("/")) {
        rest.remove_suffix(1);
    }
    if (rest.ends_with(".git")) {
        rest.remove_suffix(4);
    }
    std::string part;
    const auto flush = [&] {
        if (!part.empty()) {
            name += name.empty() ? "" : "--";
            name += part;
            part.clear();
        }
    };
    for (const char c : rest) {
        if (c == '/' || c == ':') {
            flush();
        } else {
            const bool plain = std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '.' ||
                               c == '-' || c == '_';
            part += plain ? c : '_';
        }
    }
    flush();
    return linnet_home() / "git" / name;
}

fs::path package_directory(const GitSource& source, std::string_view commit) {
    return repository_directory(source.url) / "snapshots" / std::string(commit) / source.subdir;
}

std::expected<Lockfile, std::string> read_lockfile(const fs::path& path) {
    const std::optional<std::string> text = read_file(path);
    if (!text) {
        return Lockfile{};
    }
    // `linnet fetch` writes the file; one it cannot read is best written anew.
    const auto problem = [&](const std::string& what) {
        return std::unexpected(path.generic_string() + ": " + what +
                               " (delete it and run `linnet fetch` to write it again)");
    };
    auto document = toml::parse_version_one(*text);
    if (!document) {
        return problem(document.error());
    }
    const toml::Value root{std::move(*document), 0};
    Lockfile lock;
    if (const toml::Value* entries = root.find("git")) {
        if (entries->as_array() == nullptr) {
            return problem("`git` must be [[git]] entries");
        }
        for (const toml::Value& entry : *entries->as_array()) {
            const toml::Value* source = entry.find("source");
            const toml::Value* commit = entry.find("commit");
            if (source == nullptr || source->as_string() == nullptr || commit == nullptr ||
                commit->as_string() == nullptr || commit->as_string()->size() != 40) {
                return problem("each [[git]] entry needs a `source` and a 40-character `commit`");
            }
            lock[*source->as_string()] = *commit->as_string();
        }
    }
    return lock;
}

std::string lockfile_text(const Lockfile& lock) {
    std::string text = "# Written by `linnet fetch`: the commit each git dependency resolved to.\n"
                       "version = 1\n";
    for (const auto& [source, commit] : lock) {
        text += "\n[[git]]\nsource = ";
        text += toml_string(source);
        text += "\ncommit = ";
        text += toml_string(commit);
        text += "\n";
    }
    return text;
}

namespace {

// Every package reachable from `root`, each once: `visit` takes its
// manifest (or why it could not be read) and returns the directories of the
// packages to go on to.
template <typename Visit>
void walk_packages(const fs::path& root, Visit visit) {
    std::vector<fs::path> queue = {root};
    std::set<std::string> visited;
    while (!queue.empty()) {
        const fs::path directory = queue.back();
        queue.pop_back();
        if (visited.insert(path_key(directory)).second) {
            for (fs::path& next : visit(read_manifest(directory / "linnet.toml"))) {
                queue.push_back(std::move(next));
            }
        }
    }
}

} // namespace

Fetched fetch_dependencies(const fs::path& root, const FetchOptions& options) {
    const fs::path lock_path = root / "linnet.lock";
    Fetched fetched;
    auto locked = read_lockfile(lock_path);
    if (!locked) {
        fetched.failures.push_back(locked.error());
        return fetched;
    }
    Fetcher fetcher(options, *locked);
    walk_packages(root, [&](const std::expected<PackageManifest, std::string>& manifest) {
        std::vector<fs::path> next;
        if (!manifest) {
            fetched.failures.push_back(manifest.error());
            return next;
        }
        for (const auto& [key, dependency] : manifest->dependencies) {
            if (!dependency.git) {
                next.push_back(dependency.path);
                continue;
            }
            auto commit = fetcher.resolve(*dependency.git);
            if (!commit) {
                fetched.failures.push_back("dependency `" + key + "`: " + commit.error());
                continue;
            }
            fetched.checkouts[dependency.git->key()] =
                repository_directory(dependency.git->url) / "snapshots" / *commit;
            next.push_back(package_directory(*dependency.git, *commit));
        }
        return next;
    });
    const std::string text = lockfile_text(fetcher.resolved());
    const std::optional<std::string> before = read_file(lock_path);
    // A failed dependency's pin stays in the lock until it resolves again.
    if (options.write_lock && fetched.failures.empty() && before != text &&
        (before || !fetcher.resolved().empty())) {
        std::ofstream stream(lock_path, std::ios::binary | std::ios::trunc);
        stream << text;
        if (!stream) {
            fetched.failures.push_back("cannot write " + lock_path.generic_string());
        }
    }
    return fetched;
}

bool has_git_dependencies(const fs::path& root) {
    bool found = false;
    walk_packages(root, [&](const std::expected<PackageManifest, std::string>& manifest) {
        std::vector<fs::path> next;
        for (const auto& [key, dependency] :
             manifest ? manifest->dependencies : std::map<std::string, Dependency>{}) {
            found = found || dependency.git.has_value();
            next.push_back(dependency.path);
        }
        return found ? std::vector<fs::path>{} : next;
    });
    return found;
}

} // namespace linnet
