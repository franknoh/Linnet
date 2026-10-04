"""`linnet.torch.flex.fresh_copy`: a contiguous copy of its own, gradients
passed through unchanged, eager and traced alike."""

from __future__ import annotations

import torch

from linnet.torch.flex import fresh_copy


def _sliced(base: torch.Tensor) -> torch.Tensor:
    # As a prompt pass's step rows are: sliced off at an offset, permuted.
    return base[:, 2:6].permute(1, 0, 2)


def test_fresh_copy_is_a_contiguous_copy_with_the_gradient_passed_through() -> None:
    base = torch.randn(3, 8, 4, requires_grad=True)
    view = _sliced(base)
    out = fresh_copy(view)
    assert out.is_contiguous() and not view.is_contiguous()
    assert out.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
    torch.testing.assert_close(out, view)
    (out * 3).sum().backward()
    expected = torch.zeros_like(base)
    expected[:, 2:6] = 3
    assert base.grad is not None
    torch.testing.assert_close(base.grad, expected)


def test_fresh_copy_traces() -> None:
    base = torch.randn(3, 8, 4)
    traced = torch.compile(
        lambda t: fresh_copy(_sliced(t)) * 2, backend="aot_eager", fullgraph=True
    )
    torch.testing.assert_close(traced(base), _sliced(base) * 2)
