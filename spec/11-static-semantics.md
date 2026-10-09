# 11. Static semantics

Programs must pass these checks before lowering.

## 11.1 Name resolution

Every identifier reference MUST resolve unambiguously to a declaration visible in lexical or module scope.

## 11.2 Definite declaration

Locals must be declared before use, and `var` locals must be initialized.

## 11.3 Returns

Every reachable return path in a non-unit function MUST return a value of the declared type; the final expression is never returned implicitly.

## 11.4 Dtype equality

Arithmetic tensor operands MUST have identical dtypes after explicit casts and contextual literal typing. Backend-specific implicit promotion MUST NOT leak into Linnet semantics.

## 11.5 Broadcasting

Broadcast relationships MUST be statically provable (§4.6).

## 11.6 Index expressions

In a tensor comprehension:

- every output index must have a known domain;
- every non-output index must be bound by a reduction;
- every reduction index must occur in the reduced expression;
- uses of one index must have compatible domains;
- an index repeated in one tensor selects a diagonal and requires compatible axes.

## 11.7 Effects

`fn`, `op`, and `entry` bodies have no hidden effects. Structural composition and local SSA-style reassignment are not runtime effects. `state` reads and assignments (§9.3) are explicit effects; each function's state footprint is known statically.

## 11.8 Recursion

Direct or indirect recursive calls are errors (§8.6).

## 11.9 Exhaustive match

`match` over an optional or enum MUST be exhaustive. Duplicate and unreachable arms SHOULD be diagnosed.

## 11.10 Compile-time values

Dimensions, generic shape parameters, block-array lengths, and `static for` iteration counts are compile-time values, which runtime tensor contents MUST NOT influence.
