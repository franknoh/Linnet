# Diagnostic codes

Every message carries a stable code. A code is never reused for another
condition, so tests and editors can match on it. `E` codes are errors and
fail `linnet check`; `W` codes are warnings and fail only under `--strict`
(`linnet lint`).

## Lexing and parsing

| Code | Reported for |
| --- | --- |
| E1001 | a character or byte sequence that cannot appear in source |
| E1002 | a block comment without its closing `*/` |
| E1003 | a malformed string literal or escape sequence |
| E1004 | a reserved word used as a name |
| E1005 | a malformed numeric literal |
| E1101 | an unexpected token; the message says what was expected |
| E1102 | a file that does not start with a `module` declaration |
| E1103 | syntax that is reserved but not supported |

## Names and modules

| Code | Reported for |
| --- | --- |
| E1201 | a name that is not declared in scope |
| E1202 | an import of a module that cannot be found |
| E1203 | use of an item that is not `pub` from another module |
| E1204 | two items with one name in a module |
| E1205 | two parameters, generics, fields, members, or variants with one name |
| E1206 | modules that import each other |
| E1207 | a declaration that reuses a prelude name |
| E1208 | a name of the wrong kind, such as a constant used as a type |

## Types and values

| Code | Reported for |
| --- | --- |
| E2101 | a value whose type differs from the required type |
| E2102 | wrong number of arguments, an unknown keyword, or a missing argument |
| E2103 | operands or arguments with different dtypes |
| E2104 | an operator applied to a type it does not support |
| E2105 | a literal that does not fit its dtype |
| E2106 | a call of something that is not a function |
| E2107 | a field, member, method, or variant that does not exist |
| E2108 | assignment to something that is neither a `var` nor a `state` member |
| E2109 | an assignment that would change a variable's type |
| E2110 | a generic parameter that cannot be inferred |
| E2111 | wrong number or kind of generic arguments |
| E2112 | a dtype that does not satisfy a generic constraint |
| E2113 | `none` without an optional type from context |
| E2114 | an `if`, `select`, or `while` condition that is not a scalar `bool` |
| E2115 | branches of different types |
| E2116 | `&&`, `||`, or `!` on something other than scalar `bool` |
| E2117 | a pattern that cannot match the value |
| E2118 | a `match` that does not cover every case |
| E2119 | a function that calls itself, directly or indirectly |
| E2120 | a function that does not end with `return`, or a `return` inside a loop |
| E2121 | `static for` over something that is not a structural array or compile-time range |
| E2122 | a constant, alias, or struct defined in terms of itself |
| E2123 | an expression that is not a compile-time integer where one is required |
| E2190 | a language feature this toolchain does not implement yet |

## Shapes

| Code | Reported for |
| --- | --- |
| E2201 | an index used with incompatible extents |
| E2202 | a tensor whose shape differs from the required shape |
| E2203 | a `reshape` that cannot be proven to keep the element count |
| E2204 | the wrong number of indices for a tensor |
| E2205 | a division whose divisor cannot be proven positive |
| E2206 | a `where` constraint that cannot be proven at a call |
| E2207 | shapes that cannot be proven to broadcast |
| E2208 | a dimension that is provably negative |
| E2209 | an invalid index or slice bound |
| E2210 | an invalid axis or permutation |

## Index notation

| Code | Reported for |
| --- | --- |
| E3101 | an index that is neither an output nor bound by a reduction |
| E3102 | a reduction index that indexes no tensor |
| E3103 | an output index that indexes no tensor |
| E3104 | an expression that is not valid in index notation |
| E3105 | an index listed twice |
| E3106 | a pack index used where a plain index is needed, or the reverse |

## Blocks and packages

| Code | Reported for |
| --- | --- |
| E4101 | a `param`, `buffer`, `state`, or `sub` with a type it cannot have |
| E4102 | a `param` with a value in source |
| E5001 | a `linnet.toml` that cannot be read or is malformed |

## Warnings

| Code | Reported for |
| --- | --- |
| W1001 | an import that is never used |
| W1002 | a local binding that is never used; prefix its name with `_` to keep it |
| W1003 | a block member that its block never uses |
