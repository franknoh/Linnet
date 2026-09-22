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

// Names and modules
inline constexpr const char* unknown_symbol = "E1201";
inline constexpr const char* unknown_module = "E1202";
inline constexpr const char* private_item = "E1203";
inline constexpr const char* duplicate_item = "E1204";
inline constexpr const char* duplicate_name = "E1205";
inline constexpr const char* import_cycle = "E1206";
inline constexpr const char* shadows_prelude = "E1207";
inline constexpr const char* wrong_symbol_kind = "E1208";

// Types
inline constexpr const char* type_mismatch = "E2101";
inline constexpr const char* bad_arguments = "E2102";
inline constexpr const char* dtype_mismatch = "E2103";
inline constexpr const char* invalid_operand = "E2104";
inline constexpr const char* literal_out_of_range = "E2105";
inline constexpr const char* not_callable = "E2106";
inline constexpr const char* unknown_member = "E2107";
inline constexpr const char* assign_immutable = "E2108";
inline constexpr const char* assign_changes_type = "E2109";
inline constexpr const char* cannot_infer = "E2110";
inline constexpr const char* bad_generic_arguments = "E2111";
inline constexpr const char* dtype_constraint = "E2112";
inline constexpr const char* none_needs_context = "E2113";
inline constexpr const char* condition_not_bool = "E2114";
inline constexpr const char* branch_mismatch = "E2115";
inline constexpr const char* logical_not_bool = "E2116";
inline constexpr const char* invalid_pattern = "E2117";
inline constexpr const char* non_exhaustive_match = "E2118";
inline constexpr const char* recursion = "E2119";
inline constexpr const char* missing_return = "E2120";
inline constexpr const char* invalid_static_for = "E2121";
inline constexpr const char* cyclic_definition = "E2122";
inline constexpr const char* not_compile_time = "E2123";
inline constexpr const char* not_implemented = "E2190";

// Shapes
inline constexpr const char* contraction_mismatch = "E2201";
inline constexpr const char* shape_mismatch = "E2202";
inline constexpr const char* reshape_count = "E2203";
inline constexpr const char* rank_mismatch = "E2204";
inline constexpr const char* divisor_not_positive = "E2205";
inline constexpr const char* constraint_unsatisfied = "E2206";
inline constexpr const char* unproven_broadcast = "E2207";
inline constexpr const char* negative_dimension = "E2208";
inline constexpr const char* invalid_slice = "E2209";
inline constexpr const char* invalid_axis = "E2210";

// Tensor algebra
inline constexpr const char* unbound_index = "E3101";
inline constexpr const char* unused_reduction_index = "E3102";
inline constexpr const char* index_without_domain = "E3103";
inline constexpr const char* invalid_index_expression = "E3104";
inline constexpr const char* duplicate_index = "E3105";
inline constexpr const char* pack_index_misuse = "E3106";

// Blocks and parameters
inline constexpr const char* invalid_member_type = "E4101";
inline constexpr const char* param_payload = "E4102";

// Packages
inline constexpr const char* invalid_manifest = "E5001";

// Lints
inline constexpr const char* unused_import = "W1001";
inline constexpr const char* unused_local = "W1002";
inline constexpr const char* unused_member = "W1003";

} // namespace linnet::codes
