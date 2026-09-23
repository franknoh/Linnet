# 1. Lexical Structure

## 1.1 Encoding

Source files MUST be UTF-8.

Comments and string literals MAY contain arbitrary Unicode scalar values. Identifiers in the initial language version are restricted to ASCII to reduce confusable-identifier risks and simplify portable tooling.

A future language version MAY introduce Unicode identifiers only with explicit normalization and confusable-handling rules.

## 1.2 File extension

The canonical source extension is:

```text
.linnet
```

The language name is **Linnet**. File extension and language name are intentionally different.

## 1.3 Identifiers

```text
identifier := [A-Za-z_][A-Za-z0-9_]*
```

Identifiers are case-sensitive.

Recommended style:

- modules, functions, operations, fields: `snake_case`
- block/type names: `PascalCase`
- compile-time constants: `PascalCase` or `SCREAMING_SNAKE_CASE`, formatter does not enforce either
- generic dimensions: short `PascalCase` symbols such as `B`, `S`, `H`, `In`, `Out`

## 1.4 Keywords

The following are reserved:

```text
module use pub crate self super as
const type struct enum
fn op block entry
param buffer state sub
let var return
if else match
static for in while
where
true false none some
extern
```

The following words are reserved for future syntax and MUST NOT be accepted as user identifiers:

```text
async await effect unsafe macro trait impl derive
rng mut ref yield kernel device
```

## 1.5 Comments

Line comments:

```text
// comment
```

Nested block comments are supported:

```text
/* outer
   /* inner */
*/
```

Documentation comments are reserved for future structured documentation:

```text
/// item documentation
/** item documentation */
```

A parser MAY retain them in the syntax tree before documentation semantics are implemented.

## 1.6 Literals

Integer literals:

```text
0
42
1_000_000
0xff
0b1010
```

Floating literals:

```text
0.0
1.5
1e-5
3.141_592
```

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

Numeric literals are initially context-sensitive literal values, not immediately fixed-width values.

Examples:

```text
let x: f32 = 1.0
let n: i32 = 4
```

If no context exists:

- an integer literal defaults to `i64`;
- a floating literal defaults to `f64`.

A literal conversion is valid only if its value is representable by the target scalar type. This rule does not constitute general implicit dtype conversion between tensor values.

An integer literal adopts any `Numeric` dtype; a floating literal adopts only a `Float` dtype. Neither converts to `bool`. Integer literals are limited to the range of `i64`.

Compile-time integers behave like integer literals: a `Dim` generic parameter, an unannotated integer `const`, and arithmetic over them are contextual in the same way, so `x * H` is valid for a floating tensor `x`. An unannotated `let` gives a constant integer the type `i64`, while a symbolic dimension such as `H / 2` stays a compile-time integer that can be used in shapes.

## 1.8 Whitespace and statement termination

Whitespace is not semantically significant except that it separates tokens.

Linnet does **not** require semicolons. Statements are syntactically self-delimiting because the language deliberately excludes unrestricted expression statements in the initial version.

The following are valid statements:

- `let` declarations;
- `var` declarations;
- assignment to a local `var`;
- `return`;
- `static for`;
- declarations inside structural scopes.

A formatter MUST emit one logical statement per line except where a multiline expression is more readable.
