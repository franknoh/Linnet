#pragma once

#include <cstdint>
#include <string_view>

namespace linnet {

inline constexpr char32_t replacement_character = 0xFFFD;

struct DecodedCodePoint {
    char32_t value;       // replacement_character when the sequence is invalid
    std::uint32_t length; // bytes consumed; always >= 1
    bool valid;
};

// Decodes the code point starting at `offset`, which must be < text.size().
// Malformed input (stray continuation bytes, truncated or overlong sequences,
// surrogates, values above U+10FFFF) consumes exactly one byte, so callers
// always make forward progress.
DecodedCodePoint decode_utf8(std::string_view text, std::size_t offset);

// Number of UTF-16 code units needed for a code point.
constexpr std::uint32_t utf16_length(char32_t code_point) {
    return code_point >= 0x10000 ? 2 : 1;
}

} // namespace linnet
