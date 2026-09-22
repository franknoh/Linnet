#include "linnet/package/spec_manifest.hpp"

#include "linnet/package/toml.hpp"

#include <fstream>
#include <iterator>

namespace linnet {

std::expected<std::vector<SpecCase>, std::string>
read_spec_manifest(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::unexpected(path.generic_string() + ": cannot open file");
    }
    const std::string text{std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>()};
    const auto document = toml::parse(text);
    const std::string where = path.generic_string() + ":";
    if (!document) {
        return std::unexpected(where + std::to_string(document.error().line) + ": " +
                               document.error().message);
    }
    const toml::Value root{*document, 0};
    const toml::Value* version = root.find("version");
    if (version == nullptr || version->as_integer() == nullptr || *version->as_integer() != 1) {
        return std::unexpected(where + " `version = 1` is required");
    }
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
