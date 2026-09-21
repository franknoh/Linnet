#pragma once

// Stable diagnostic codes. Ranges follow the specification:
//   E1xxx parse/module   E2xxx type/shape   E3xxx tensor algebra
//   E4xxx block/parameter   E5xxx package/binding
// A code is never reused for an unrelated condition.

namespace linnet::codes {

// Lexical
inline constexpr const char* invalid_character = "E1001";
inline constexpr const char* unterminated_comment = "E1002";
inline constexpr const char* invalid_string = "E1003";
inline constexpr const char* reserved_identifier = "E1004";
inline constexpr const char* invalid_number = "E1005";

// Syntax
inline constexpr const char* unexpected_token = "E1101";
inline constexpr const char* missing_module_decl = "E1102";
inline constexpr const char* unsupported_syntax = "E1103";

} // namespace linnet::codes
