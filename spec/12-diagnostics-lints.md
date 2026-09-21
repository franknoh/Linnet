# 12. Diagnostics, Lints, and Strict Mode

## 12.1 Diagnostic structure

A diagnostic SHOULD contain:

- stable diagnostic code;
- severity;
- concise message;
- primary source span;
- optional secondary labeled spans;
- optional notes;
- optional machine-applicable fix-it.

Example:

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

Errors mean the program has no valid Linnet semantics.

Lints report valid but suspicious, redundant, non-portable, or style-problematic source.

## 12.3 Initial lint categories

Recommended lints:

- unused import;
- unused local;
- unused parameter or buffer declaration;
- redundant cast;
- redundant reshape/permute;
- suspicious broad broadcast;
- shadowing where readability suffers;
- non-canonical module/file naming;
- parameter declared optional but always force-unwrapped after future option utilities;
- numerically suspicious forms once numerical analysis is introduced.

## 12.4 `--strict`

`linnet check --strict` promotes configured warnings to errors for CI.

Strict mode MUST NOT change program semantics, dtype rules, shape rules, optimizer legality, or backend output. It changes only acceptance policy for lint severities.

## 12.5 Stable codes

Once a diagnostic code is documented in a stable release, implementations SHOULD avoid reusing it for an unrelated condition.

Suggested ranges:

```text
E1xxx parse/module errors
E2xxx type/shape errors
E3xxx tensor algebra errors
E4xxx block/parameter errors
E5xxx package/binding errors
W1xxx style/readability
W2xxx portability/numerical warnings
```
