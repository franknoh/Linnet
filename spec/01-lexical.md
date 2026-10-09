# 1. Lexical structure

Linnet source is UTF-8 text with no statement terminators.

## 1.1 Encoding

Source files MUST be UTF-8. Comments and string literals MAY contain any Unicode scalar value.

Identifiers are ASCII-only. A future version MAY introduce Unicode identifiers only with explicit normalization and confusable-handling rules.

## 1.2 File extension

The canonical source extension is:

```text
.linnet
```

## 1.3 Identifiers

```text
identifier := [A-Za-z_][A-Za-z0-9_]*
```

Identifiers are case-sensitive.

Recommended style:

- modules, functions, operations, fields: `snake_case`
- blocks and types: `PascalCase`
- compile-time constants: `PascalCase` or `SCREAMING_SNAKE_CASE`; the formatter enforces neither
- generic dimensions: short `PascalCase` symbols such as `B`, `S`, `H`, `In`, `Out`

## 1.4 Keywords

Reserved keywords:

```text
module use pub crate self super as
const type struct enum
fn op block entry
param buffer state sub
let var return
if else match
static for in while yield
where
true false none some
extern
```

`grad` is a keyword only after an op's body (§7.2); elsewhere it is an identifier.

Words reserved for future syntax MUST NOT be accepted as identifiers:

```text
async await effect unsafe macro trait impl derive
rng mut ref kernel device
```

## 1.5 Comments

Line comments:

```text
// comment
```

Block comments nest:

```text
/* outer
   /* inner */
*/
```

Reserved documentation comments:

```text
/// item documentation
/** item documentation */
```

A parser MAY retain them in the syntax tree.

## 1.6 Literals

Integer literals:

```text
0
42
1_000_000
0xff
0b1010
```

`_` may appear only between two digits. A decimal literal other than `0` has no leading zero. Hexadecimal (`0x`) and binary (`0b`) forms are integers only.

Floating literals:

```text
0.0
1.5
1e-5
3.141_592
```

A floating literal has a fractional part, an exponent (`e` or `E`, optionally signed), or both. Each digit run follows the integer `_` rule.

Boolean literals:

```text
true
false
```

Strings are UTF-8:

```text
"model.layers.0.weight"
```

Raw strings are reserved for future versions.

## 1.7 Literal typing

A numeric literal has no fixed width; its context gives it a type:

```text
let x: f32 = 1.0
let n: i32 = 4
```

Without context, an integer literal defaults to `i64` and a floating literal to `f64`.

A literal converts only to a scalar type that can represent its value. An integer literal adopts any `Numeric` dtype; a floating literal adopts only a `Float` dtype. Neither converts to `bool`. Integer literals are limited to the `i64` range.

`Dim` generic parameters, unannotated integer `const`s, and arithmetic over them are contextual too: `x * H` is valid for a floating tensor `x`. An unannotated `let` gives a constant integer the type `i64`, but a symbolic dimension such as `H / 2` stays a compile-time integer usable in shapes.

## 1.8 Whitespace and statement termination

Whitespace only separates tokens. Linnet does **not** require semicolons.

Valid statements:

- `let` declarations;
- `var` declarations;
- assignment to a local `var` or to a `state` member of the enclosing block;
- `return`;
- `static for`;
- `while`;
- declarations inside structural scopes.

A formatter MUST emit one logical statement per line, except where a multiline expression is more readable.
