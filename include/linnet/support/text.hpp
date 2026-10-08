#pragma once

#include <string>

namespace linnet {

// The shortest decimal that reads back as the same double, with a `.0` when
// it would otherwise read as an integer.
std::string shortest_float(double value);

} // namespace linnet
