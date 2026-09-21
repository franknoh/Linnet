# Linnet Executable Specification Tests

These fixtures are intended to become the first test corpus consumed by the real parser/type checker.

## Contract

- Every file under `valid/` MUST parse and pass static semantic checking.
- Every file under `invalid/` MUST fail with the diagnostic code listed in `manifest.toml`.
- Files under `diagnostics/` are expected stderr snapshots demonstrating desired diagnostic quality. Exact whitespace may evolve before the first stable CLI release, but diagnostic codes and semantic content should remain consistent.
- `fixtures/` contains small package/module trees used for import tests.

The test runner should eventually support:

```bash
linnet spec-test spec-tests/
```

or an internal equivalent used by CTest.

The language specification is normative. Tests that disagree with the specification should be fixed rather than silently defining alternate semantics.
