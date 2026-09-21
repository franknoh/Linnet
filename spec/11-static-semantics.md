# 11. Static Semantics

## 11.1 Name resolution

Every identifier reference MUST resolve unambiguously to a declaration visible in lexical/module scope.

Unknown and ambiguous names are errors.

## 11.2 Definite declaration

Locals must be declared before use.

`var` locals must have an initial value; uninitialized locals are not permitted.

## 11.3 Return checking

Every reachable return path in a non-unit function MUST return a value matching the declared return type.

Implicit return of the final expression is not part of the initial language version. `return` is explicit.

## 11.4 Dtype equality

Tensor operands for arithmetic operations MUST have identical dtypes after explicit casts and contextual literal typing.

A backend-specific implicit promotion MUST NOT leak into Linnet semantics.

## 11.5 Broadcast checking

Broadcast relationships MUST be statically provable under the rules in the shape specification.

## 11.6 Index-expression checking

For a tensor comprehension:

- every output index must have a known domain;
- every non-output index used in the expression must be bound by a reduction;
- every reduction index must occur in the reduced expression unless a future explicit-range syntax provides its domain;
- uses of the same index must have compatible domains;
- repeated indices in one tensor imply diagonal selection and require compatible axes.

## 11.7 Purity and effects

Initial `fn`, `op`, and `entry` bodies have no hidden effects. Structural composition and local SSA-style reassignment do not count as runtime effects.

## 11.8 Recursive cycles

Direct or indirect recursive calls are errors in the initial language version.

## 11.9 Exhaustive match

`match` over an optional or enum MUST be exhaustive. Duplicate and unreachable arms SHOULD be diagnosed.

## 11.10 Compile-time versus runtime values

Dimensions, generic shape parameters, block-array lengths, and `static for` iteration counts are compile-time structural values.

Runtime tensor contents MUST NOT influence them.

This separation is required for deterministic static shape checking and portable backend lowering.
