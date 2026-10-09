# Diagnostic codes

Codes are stable and never reused. Errors (`E`) fail `linnet check`;
warnings (`W`) fail only under `--strict` (`linnet lint`).

## Syntax

| Code | Meaning |
| --- | --- |
| E1001 | invalid character or bytes |
| E1002 | unclosed block comment |
| E1003 | malformed string or escape |
| E1004 | reserved word as a name |
| E1005 | malformed number |
| E1101 | unexpected token |
| E1102 | missing leading `module` |
| E1103 | reserved, unsupported syntax |

## Names

| Code | Meaning |
| --- | --- |
| E1201 | undeclared name |
| E1202 | module not found |
| E1203 | non-`pub` item from another module |
| E1204 | duplicate item in a module |
| E1205 | duplicate parameter, generic, field, member, or variant |
| E1206 | import cycle |
| E1207 | prelude name redeclared |
| E1208 | name of the wrong kind |

## Types

| Code | Meaning |
| --- | --- |
| E2101 | value of the wrong type |
| E2102 | wrong, missing, or unknown-keyword arguments |
| E2103 | different dtypes; use an explicit `cast` |
| E2104 | operator unsupported for the type |
| E2105 | literal out of its dtype's range |
| E2106 | call of a non-function |
| E2107 | no such field, member, method, or variant |
| E2108 | assignment to a non-`var`, non-`state` target |
| E2109 | assignment changes a variable's type |
| E2110 | uninferable generic; write `f<...>(...)` |
| E2111 | wrong generic arguments |
| E2112 | dtype fails a generic constraint |
| E2113 | `none` without an optional type |
| E2114 | condition not a scalar `bool` |
| E2115 | branches of different types |
| E2116 | `&&`, `\|\|`, or `!` operand not a scalar `bool` |
| E2117 | pattern cannot match |
| E2118 | `match` misses a case |
| E2119 | recursion |
| E2120 | no final `return`, or `return` inside a loop |
| E2121 | `static for` over something not a structural array or compile-time range |
| E2122 | self-referential constant, alias, or struct |
| E2123 | compile-time integer required |
| E2124 | `yield` outside the end of a `for` that is a binding's value, or such a `for` elsewhere |
| E2125 | `grad` on something other than an op, or on an op that does not return one floating tensor or scalar, takes a parameter that is not a tensor or scalar, or has no floating tensor parameter |
| E2126 | an op's `kernel` that does not take the op's parameters or write what it returns, or a call to a kernel |
| E2127 | a kernel rule broken: a tensor in memory outside `load`/`store`, `return` in a kernel, `store` outside one, slices in a memory index |
| E2190 | unimplemented feature |

## Shapes

| Code | Meaning |
| --- | --- |
| E2201 | incompatible extents for an index |
| E2202 | wrong shape |
| E2203 | `reshape` element count unprovable; add `where H % N == 0` |
| E2204 | wrong index count |
| E2205 | divisor not provably positive; add `where N > 0` |
| E2206 | `where` unprovable at a call; restate it in the caller |
| E2207 | broadcast unprovable |
| E2208 | provably negative dimension |
| E2209 | invalid index or slice bound |
| E2210 | invalid axis or permutation |

## Index notation

| Code | Meaning |
| --- | --- |
| E3101 | index neither output nor reduced |
| E3102 | unused reduction index |
| E3103 | unused output index |
| E3104 | invalid in index notation |
| E3105 | index listed twice |
| E3106 | pack index for a plain one, or the reverse |

## Blocks and packages

| Code | Meaning |
| --- | --- |
| E4101 | member of a type it cannot have |
| E4102 | `param` with a value in source |
| E5001 | `linnet.toml` unreadable or malformed |

## Warnings

| Code | Meaning |
| --- | --- |
| W1001 | unused import |
| W1002 | unused local; prefix it with `_` to keep it |
| W1003 | unused `param`, `buffer`, or `sub` |
