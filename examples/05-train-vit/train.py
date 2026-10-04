"""Trains the ViT example (`examples/02-vit`) from scratch, in seconds on a
CPU: which of four textures fills a 32x32 image. The Linnet model is an
ordinary `torch.nn.Module`, trained by a plain PyTorch loop."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from linnet.torch import load

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "02-vit" / "vit.linnet"
STDLIB = HERE.parents[1] / "stdlib"
GENERICS = {
    "Height": 32,
    "Width": 32,
    "Channels": 1,
    "Patch": 8,
    "D": 64,
    "Heads": 4,
    "Inner": 128,
    "Layers": 2,
    "Classes": 4,
    "T": "f32",
}


def images(count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Faint noise under one of four textures, at a random period and phase:
    horizontal stripes, vertical stripes, a checkerboard, or dots."""
    labels = torch.randint(0, 4, (count,), generator=generator)
    y = torch.arange(32)[:, None].expand(32, 32)
    x = torch.arange(32)[None, :].expand(32, 32)
    pixels = torch.rand(count, 1, 32, 32, generator=generator) * 0.3
    for i in range(count):
        period = int(torch.randint(4, 7, (1,), generator=generator))
        phase = int(torch.randint(0, period, (1,), generator=generator))
        textures = [
            (y + phase) % period < period // 2,
            (x + phase) % period < period // 2,
            ((y + phase) // 3 + (x + phase) // 3) % 2 == 0,
            ((y + phase) % period == 0) & ((x + phase) % period == 0),
        ]
        pixels[i, 0][textures[int(labels[i])]] = 1.0
    return pixels, labels


def main(steps: int = 200, batch: int = 64, lr: float = 1e-3, device: str = "cpu") -> float:
    # Without weights the parameters are zeros: a model trained from scratch
    # starts from an initialization of its own.
    model = load(
        SOURCE, generics=GENERICS, std_root=STDLIB, device=device, trainable=True, compile=True
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.zero_()
            elif ".norm" in name or name.startswith("root.norm"):
                parameter.fill_(1.0)
            else:
                torch.nn.init.trunc_normal_(parameter, std=0.02)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(0)
    test_images, test_labels = images(1024, torch.Generator().manual_seed(1))
    accuracy = 0.0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        pixels, labels = images(batch, generator)
        loss = torch.nn.functional.cross_entropy(model(pixels.to(device)), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                predicted = model(test_images.to(device)).argmax(-1).cpu()
            accuracy = float((predicted == test_labels).float().mean())
            print(
                f"step {step}: loss {loss.item():.3f}, held-out accuracy {accuracy:.3f}"
                f" ({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    options = parser.parse_args()
    main(options.steps, options.batch, options.lr, options.device)
