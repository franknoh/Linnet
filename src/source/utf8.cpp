#include "linnet/source/utf8.hpp"

namespace linnet {

namespace {

constexpr bool is_continuation(unsigned char byte) {
    return (byte & 0xC0U) == 0x80U;
}

} // namespace

DecodedCodePoint decode_utf8(std::string_view text, std::size_t offset) {
    constexpr DecodedCodePoint invalid{replacement_character, 1, false};

    const auto byte_at = [&](std::size_t index) { return static_cast<unsigned char>(text[index]); };
    const unsigned char lead = byte_at(offset);
    if (lead < 0x80U) {
        return {lead, 1, true};
    }

    std::uint32_t length = 0;
    char32_t value = 0;
    char32_t minimum = 0;
    if ((lead & 0xE0U) == 0xC0U) {
        length = 2;
        value = lead & 0x1FU;
        minimum = 0x80;
    } else if ((lead & 0xF0U) == 0xE0U) {
        length = 3;
        value = lead & 0x0FU;
        minimum = 0x800;
    } else if ((lead & 0xF8U) == 0xF0U) {
        length = 4;
        value = lead & 0x07U;
        minimum = 0x10000;
    } else {
        return invalid;
    }

    if (text.size() - offset < length) {
        return invalid;
    }
    for (std::uint32_t i = 1; i < length; ++i) {
        const unsigned char byte = byte_at(offset + i);
        if (!is_continuation(byte)) {
            return invalid;
        }
        value = (value << 6) | (byte & 0x3FU);
    }

    const bool is_surrogate = value >= 0xD800 && value <= 0xDFFF;
    if (value < minimum || value > 0x10FFFF || is_surrogate) {
        return invalid;
    }
    return {value, length, true};
}

} // namespace linnet
