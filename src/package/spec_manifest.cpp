#include "linnet/package/spec_manifest.hpp"

#include "linnet/package/toml.hpp"
#include "linnet/support/files.hpp"

#include <optional>
#include <utility>

namespace linnet {

std::expected<std::vector<SpecCase>, std::string>
read_spec_manifest(const std::filesystem::path& path) {
    const std::string where = path.generic_string() + ":";
    const std::optional<std::string> text = read_file(path);
    if (!text) {
        return std::unexpected(where + " cannot read file");
    }
    auto document = toml::parse_version_one(*text);
    if (!document) {
        return std::unexpected(where + " " + document.error());
    }
    const toml::Value root{std::move(*document), 0};
    const toml::Value* cases = root.find("case");
    if (cases == nullptr || cases->as_array() == nullptr) {
        return std::unexpected(where + " no [[case]] entries");
    }

    std::vector<SpecCase> result;
    for (const toml::Value& entry : *cases->as_array()) {
        const std::string at = where + std::to_string(entry.line) + ": ";
        const toml::Value* file = entry.find("file");
        const toml::Value* expect = entry.find("expect");
        const toml::Value* code = entry.find("code");
        if (file == nullptr || file->as_string() == nullptr) {
            return std::unexpected(at + "each case needs a `file` string");
        }
        if (expect == nullptr || expect->as_string() == nullptr ||
            (*expect->as_string() != "ok" && *expect->as_string() != "error")) {
            return std::unexpected(at + "`expect` must be \"ok\" or \"error\"");
        }
        SpecCase spec_case{*file->as_string(), *expect->as_string() == "error", {}};
        if (spec_case.expects_error) {
            if (code == nullptr || code->as_string() == nullptr) {
                return std::unexpected(at + "an error case needs a `code` string");
            }
            spec_case.code = *code->as_string();
        } else if (code != nullptr) {
            return std::unexpected(at + "an ok case cannot have a `code`");
        }
        result.push_back(std::move(spec_case));
    }
    return result;
}

} // namespace linnet
