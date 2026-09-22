# Linnet Executable Specification Tests

These fixtures are intended to become the first test corpus consumed by the real parser/type checker.

## Contract

- Every file under `valid/` MUST parse and pass static semantic checking.
- Every file under `invalid/` MUST fail with the diagnostic code listed in `manifest.toml`.
- Files under `diagnostics/` are stderr snapshots: `diagnostics/<name>.stderr` must equal the errors rendered for `invalid/<name>.linnet`, byte for byte, with paths relative to this directory. Regenerate one with `linnet check --no-color invalid/<name>.linnet 2> diagnostics/<name>.stderr` from this directory and review the change like any other.
- `fixtures/` contains small package/module trees used for import tests.

Run the suite with:

```bash
linnet spec-test --std spec-tests/fixtures/std spec-tests/
```

`ctest` runs the same cases through the test executables and the command.

The language specification is normative. Tests that disagree with the specification should be fixed rather than silently defining alternate semantics.
