#pragma once

#include <ranges>
#include <string>
#include <string_view>

namespace linnet {

// The shortest decimal that reads back as the same double, with a `.0` when
// it would otherwise read as an integer.
std::string shortest_float(double value);

// One argument of a Windows command line, as the C runtime splits the line
// back into words: quoted when it is empty or has a space, a tab or a quote,
// with each backslash before a quote doubled.
std::string command_line_word(std::string_view argument);

// `items` between `separator`s, each as `spell` writes it.
template <std::ranges::input_range Range, typename Spell>
std::string join(const Range& items, std::string_view separator, Spell spell) {
    std::string out;
    bool is_first = true;
    for (const auto& item : items) {
        if (!is_first) {
            out += separator;
        }
        is_first = false;
        out += spell(item);
    }
    return out;
}

// `items` (strings) between `separator`s.
template <std::ranges::input_range Range>
std::string join(const Range& items, std::string_view separator) {
    return join(items, separator, [](const auto& item) -> const auto& { return item; });
}

} // namespace linnet
