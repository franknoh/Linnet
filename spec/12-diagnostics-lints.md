# 12. Diagnostics and lints

Errors and lints use stable codes.

## 12.1 Diagnostic structure

A diagnostic SHOULD contain a stable code, severity, concise message, and primary source span, and MAY add labeled secondary spans, notes, and machine-applicable fix-its. Example:

```text
error E2201: incompatible contraction dimensions

  --> model.linnet:14:20
   |
14 |     let y[m, n] = sum[k] a[m, k] * b[k, n]
   |                    ^^^^^^                ^
   |                    `k` is 4096 here      `k` is 5120 here
   |
   = note: reduction index `k` must have one consistent domain
```

## 12.2 Error versus lint

An error means the program has no valid Linnet semantics; a lint flags valid but suspicious, redundant, non-portable, or poorly styled source.

## 12.3 Lints

Recommended lints, with reference-implementation codes:

- unused import (`W1001`);
- unused local (`W1002`);
- unused parameter or buffer declaration (`W1003`, any unused block member: `param`, `buffer`, `state`, `sub`);
- redundant cast;
- redundant reshape/permute;
- suspicious broad broadcast;
- shadowing where readability suffers;
- non-canonical module/file naming;
- optional parameter always force-unwrapped, after future option utilities;
- numerically suspicious forms, once numerical analysis exists.

Locals and block members whose names begin with `_` are exempt. Lints are reported only for error-free programs.

## 12.4 `--strict`

`linnet check --strict` promotes configured warnings to errors; `linnet lint` fails on every warning. Strict mode MUST NOT change program semantics, dtype rules, shape rules, optimizer legality, or backend output.

## 12.5 Stable codes

Implementations SHOULD NOT reuse a documented stable-release code for an unrelated condition. Suggested ranges:

```text
E1xxx parse/module errors
E2xxx type/shape errors
E3xxx tensor algebra errors
E4xxx block/parameter errors
E5xxx package/binding errors
W1xxx style/readability
W2xxx portability/numerical warnings
```
