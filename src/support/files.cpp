#include "linnet/support/files.hpp"

#include <fstream>
#include <iterator>
#include <system_error>

namespace linnet {

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return std::nullopt;
    }
    std::string text{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        return std::nullopt;
    }
    return text;
}

std::string path_key(const std::filesystem::path& path) {
    std::error_code error;
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(path, error);
    return (error ? path : canonical).generic_string();
}

} // namespace linnet
