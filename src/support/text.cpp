#include "linnet/support/text.hpp"

#include <cstdlib>
#include <sstream>

namespace linnet {

std::string shortest_float(double value) {
    std::string text;
    for (int precision = 6; precision <= 17; ++precision) {
        std::ostringstream stream;
        stream.precision(precision);
        stream << value;
        text = stream.str();
        if (precision == 17 || std::strtod(text.c_str(), nullptr) == value) {
            break;
        }
    }
    if (text.find_first_of(".eEni") == std::string::npos) {
        text += ".0";
    }
    return text;
}

} // namespace linnet
