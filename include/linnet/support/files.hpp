#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace linnet {

// A file's bytes, or none when it cannot be read.
std::optional<std::string> read_file(const std::filesystem::path& path);

// The one spelling of a path every map of files is keyed by: canonical as
// far as it exists, `path` itself when that fails.
std::string path_key(const std::filesystem::path& path);

} // namespace linnet
