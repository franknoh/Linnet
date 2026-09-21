#include "linnet/module/loader.hpp"

#include "linnet/syntax/parser.hpp"

#include <map>
#include <optional>
#include <string>

namespace linnet {

namespace fs = std::filesystem;

std::vector<const ast::Ast*> Program::module_pointers() const {
    std::vector<const ast::Ast*> pointers;
    pointers.reserve(modules.size());
    for (const ast::Ast& module : modules) {
        pointers.push_back(&module);
    }
    return pointers;
}

fs::path find_package_root(const fs::path& file) {
    std::error_code error;
    fs::path directory = fs::weakly_canonical(file, error).parent_path();
    while (!directory.empty()) {
        if (fs::is_regular_file(directory / "linnet.toml", error)) {
            return directory;
        }
        if (directory == directory.root_path()) {
            break;
        }
        directory = directory.parent_path();
    }
    return {};
}

namespace {

class Loader {
public:
    Loader(SourceManager& sources, const LoaderOptions& options, DiagnosticSink& sink)
        : sources_(sources), options_(options), sink_(sink) {}

    Program run(std::span<const fs::path> files) {
        for (const fs::path& file : files) {
            load(file, true);
        }
        // `load` appends to the queue while it is being walked.
        for (std::uint32_t module = 0; module < program_.modules.size(); ++module) {
            const std::size_t use_count = program_.modules[module].uses.size();
            for (std::uint32_t use = 0; use < use_count; ++use) {
                const auto file = file_of_import(module, program_.modules[module].uses[use].path);
                const auto target = file ? load(*file, false) : std::nullopt;
                if (target) {
                    program_.imports[{module, use}] = *target;
                }
            }
        }
        return std::move(program_);
    }

private:
    // Imports are made of identifiers, so they cannot contain separators or
    // `..` and always stay below their root directory.
    static fs::path relative_file(const std::vector<ast::Name>& path, std::size_t from) {
        fs::path file;
        for (std::size_t i = from; i < path.size(); ++i) {
            file /= std::string(path[i].text);
        }
        file += ".linnet";
        return file;
    }

    std::optional<fs::path> file_of_import(std::uint32_t module,
                                           const std::vector<ast::Name>& path) const {
        if (path.empty() || path.front().text.empty()) {
            return std::nullopt;
        }
        if (path.front().text == "std") {
            if (options_.std_root.empty() || path.size() < 2) {
                return std::nullopt;
            }
            return options_.std_root / relative_file(path, 1);
        }
        if (path.front().text == "crate") {
            const fs::path& root = package_roots_[module];
            if (root.empty()) {
                return std::nullopt;
            }
            return path.size() == 1 ? root / "src" / "lib.linnet"
                                    : root / "src" / relative_file(path, 1);
        }
        return std::nullopt; // dependencies are not resolved yet
    }

    std::optional<std::uint32_t> load(const fs::path& file, bool is_requested) {
        std::error_code error;
        const fs::path canonical = fs::weakly_canonical(file, error);
        const std::string key = (error ? file : canonical).generic_string();
        if (const auto found = loaded_.find(key); found != loaded_.end()) {
            return found->second;
        }
        if (!is_requested && !fs::is_regular_file(file, error)) {
            return std::nullopt; // reported as an unknown module by the analysis
        }
        const auto id = sources_.load_file(file);
        if (!id) {
            Diagnostic diagnostic;
            diagnostic.message = id.error();
            sink_.report(std::move(diagnostic));
            return std::nullopt;
        }
        program_.modules.push_back(parse(sources_, *id, sink_));
        package_roots_.push_back(find_package_root(file));
        const auto index = static_cast<std::uint32_t>(program_.modules.size() - 1);
        loaded_.emplace(key, index);
        return index;
    }

    SourceManager& sources_;
    const LoaderOptions& options_;
    DiagnosticSink& sink_;
    Program program_;
    std::vector<fs::path> package_roots_;
    std::map<std::string, std::uint32_t> loaded_;
};

} // namespace

Program load_program(SourceManager& sources,
                     std::span<const fs::path> files,
                     const LoaderOptions& options,
                     DiagnosticSink& sink) {
    return Loader(sources, options, sink).run(files);
}

} // namespace linnet
