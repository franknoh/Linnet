#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace linnet {

// One entry of a `spec-tests/manifest.toml`:
//
//   [[case]]
//   file = "invalid/101_shape_mismatch.linnet"
//   expect = "error"      # or "ok"
//   code = "E2201"        # required when expect = "error"
struct SpecCase {
    std::string file;
    bool expects_error = false;
    std::string code;
};

// Reads and validates a manifest. The error message names the offending line.
std::expected<std::vector<SpecCase>, std::string>
read_spec_manifest(const std::filesystem::path& path);

} // namespace linnet
