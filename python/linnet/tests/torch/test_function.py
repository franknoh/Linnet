"""Functions: module-level entries (`examples/09-functions`) as PyTorch
functions. Interpreted and as generated source, they agree with PyTorch's own
losses and with hand-written preprocessing and rewards, bind their generics
from each call's inputs, and are differentiated by autograd -- into a Linnet
model's parameters too."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional

from linnet.plan import PlanError
from linnet.torch import load, load_function

REPO = Path(__file__).resolve().parents[4]
STDLIB = REPO / "stdlib"
EXAMPLE = REPO / "examples" / "09-functions" / "functions.linnet"


@pytest.fixture(autouse=True)
def _compiler() -> None:
    if "LINNET_BIN" not in os.environ:
        for candidate in ("build/debug/linnet", "build/release/linnet"):
            if (REPO / candidate).exists():
                os.environ["LINNET_BIN"] = str(REPO / candidate)
                break
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_entropy_and_its_gradient_match_pytorch(compile: bool, dtype: torch.dtype) -> None:
    cross_entropy = load_function(EXAMPLE, "cross_entropy", std_root=STDLIB, compile=compile)
    exact = dtype == torch.float32
    rtol, atol = (1e-5, 1e-6) if exact else (1e-2, 1e-2)
    # A second batch size binds `B` again: one function serves both.
    for batch in (6, 3):
        logits = torch.randn(batch, 11).to(dtype).requires_grad_(True)
        labels = torch.randint(0, 11, (batch,))
        loss = cross_entropy(logits, labels)
        assert loss.shape == () and loss.dtype == torch.float32
        loss.backward()
        reference = logits.detach().clone().requires_grad_(True)
        expected = functional.cross_entropy(reference.float(), labels)
        expected.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.testing.assert_close(loss, expected, rtol=rtol, atol=atol)
        assert logits.grad is not None and reference.grad is not None
        torch.testing.assert_close(logits.grad, reference.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("compile", [False, True])
def test_images_are_normalized_channels_first(compile: bool) -> None:
    normalize = load_function(EXAMPLE, "normalize_images", std_root=STDLIB, compile=compile)
    pixels = torch.randint(0, 256, (2, 5, 4, 3), dtype=torch.uint8)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    got = normalize(pixels, mean, std)
    expected = ((pixels.float() / 255.0 - mean) / std).permute(0, 3, 1, 2)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("compile", [False, True])
def test_a_policy_gradient_through_token_log_probs(compile: bool) -> None:
    log_probs = load_function(EXAMPLE, "token_log_probs", std_root=STDLIB, compile=compile)
    reward = load_function(EXAMPLE, "match_reward", std_root=STDLIB, compile=compile)
    logits = torch.randn(3, 5, 7, requires_grad=True)
    tokens = torch.randint(0, 7, (3, 5))
    reference = tokens.clone()
    reference[0, 1] = (reference[0, 1] + 1) % 7
    reference[2] = (reference[2] + 3) % 7
    lengths = torch.tensor([5, 2, 4])

    # The reward: the fraction of each reference reproduced, less a penalty
    # (a Python number for the scalar input) for the fraction missed.
    rewards = reward(tokens, reference, lengths, 0.5)
    inside = torch.arange(5)[None, :] < lengths[:, None]
    matched = ((tokens == reference) & inside).sum(-1).float()
    expected = (matched - 0.5 * (lengths - matched)) / lengths
    torch.testing.assert_close(rewards, expected)

    loss = -(log_probs(logits, tokens).sum(-1) * rewards).mean()
    loss.backward()
    leaf = logits.detach().clone().requires_grad_(True)
    chosen = torch.log_softmax(leaf, -1).gather(-1, tokens[..., None])[..., 0]
    expected_loss = -(chosen.sum(-1) * expected).mean()
    expected_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(loss, expected_loss)
    assert logits.grad is not None and leaf.grad is not None
    torch.testing.assert_close(logits.grad, leaf.grad)


@pytest.mark.parametrize("compile", [False, True])
def test_a_model_trains_on_a_linnet_loss(compile: bool) -> None:
    model = load(
        EXAMPLE,
        generics={"In": 4, "Classes": 3},
        root="Classifier",
        std_root=STDLIB,
        compile=compile,
        trainable=True,
    )
    cross_entropy = load_function(EXAMPLE, "cross_entropy", std_root=STDLIB, compile=compile)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.5)
    x = torch.randn(32, 4)
    labels = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()

    # The function on the model's output, and the model's own `loss` entry,
    # which calls the same function inside the model's program.
    loss = cross_entropy(model(x), labels)
    torch.testing.assert_close(model.run_entry("loss", [x, labels]), loss)
    weight = dict(model.named_parameters())["root.head.weight"]
    expected = functional.cross_entropy(functional.linear(x, weight), labels)
    torch.testing.assert_close(loss, expected)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    first = loss.item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = cross_entropy(model(x), labels)
        loss.backward()
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert loss.item() < first * 0.7, (first, loss.item())


POSITIONS = """\
module tests.positions

pub entry positions<N: Dim>(offset: i64) -> Tensor[N; i64] {
    let position = iota<i64>(N)
    let shifted[n] = position[n] + offset
    return shifted
}
"""


@pytest.mark.parametrize("compile", [False, True])
def test_a_generic_the_inputs_leave_open_is_given_by_name(tmp_path: Path, compile: bool) -> None:
    source = tmp_path / "positions.linnet"
    source.write_text(POSITIONS, encoding="utf-8")
    positions = load_function(source, compile=compile)  # the only function: no name needed
    assert positions.name == "positions" and positions.inputs == ["offset"]
    assert positions(3, N=4).tolist() == [3, 4, 5, 6]
    with pytest.raises(PlanError, match="cannot determine `N`"):
        positions(3)


def test_naming_and_typing_mistakes_are_reported() -> None:
    with pytest.raises(PlanError, match="4 module-level entries"):
        load_function(EXAMPLE, std_root=STDLIB)
    with pytest.raises(PlanError, match="no module-level entry `forward`"):
        load_function(EXAMPLE, "forward", std_root=STDLIB)
    cross_entropy = load_function(EXAMPLE, "cross_entropy", std_root=STDLIB, compile=False)
    with pytest.raises(PlanError, match="`T` of `cross_entropy` is float, not i64"):
        cross_entropy(torch.zeros(2, 3, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(PlanError, match="takes 2 inputs, got 1"):
        cross_entropy(torch.zeros(2, 3))
    with pytest.raises(PlanError, match="has size 4 where `B` is 2"):
        cross_entropy(torch.zeros(2, 3), torch.zeros(4, dtype=torch.int64))


def test_the_generated_source_is_a_function_of_the_inputs_alone() -> None:
    cross_entropy = load_function(EXAMPLE, "cross_entropy", std_root=STDLIB, compile=True)
    cross_entropy(torch.randn(2, 3), torch.tensor([0, 2]))
    text = cross_entropy.generated_source()
    assert "# cross_entropy from module examples.functions" in text
    assert "PARAMETERS = []" in text and "def main(in_logits, in_labels" in text
