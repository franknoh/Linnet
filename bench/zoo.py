"""Collects the model zoo's measurements into `results/zoo.json`, which the
documentation site's Benchmarks page renders beside `latest.json`.

The zoo (https://github.com/franknoh/nest) measures every model it hosts with
real weights against the stacks people already use -- transformers,
diffusers, sentence-transformers, vLLM -- and writes one
`models/<name>/bench.json` per model. This keeps the rows the page draws and
nothing else, so the site builds without a zoo checkout:

    python bench/zoo.py ../nest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The one number each kind of model is judged by, in the order tried.
PRIMARY = ["decode_tok_s", "latency_ms", "step_ms", "encode_ms"]


def _task(tags: list[str]) -> str:
    if "text-generation" in tags:
        return "decoder"
    if "image-generation" in tags:
        return "diffusion"
    if "vision" in tags:
        return "vision"
    if "audio" in tags:
        return "audio"
    return "encoder"


def collect(nest: Path) -> dict[str, object]:
    index = json.loads((nest / "index.json").read_text(encoding="utf-8"))
    models: list[dict[str, object]] = []
    environment: dict[str, str] = {}
    date = ""
    for card in index["models"]:
        path = nest / "models" / card["name"] / "bench.json"
        if not path.exists():
            continue
        bench = json.loads(path.read_text(encoding="utf-8"))
        environment = environment or bench.get("environment", {})
        date = max(date, bench.get("date", ""))
        rows = []
        for method in bench["methods"]:
            metrics = {
                k: round(v, 4)
                for k, v in (method.get("metrics") or {}).items()
                if isinstance(v, (int, float)) and k != "first_token"
            }
            rows.append(
                {
                    "key": method["key"],
                    "method": method["method"],
                    "kind": "linnet" if "linnet" in method["key"] else "reference",
                    "metrics": metrics,
                    "max_abs_diff": method.get("max_abs_diff"),
                    "notes": method.get("notes") or "",
                    "error": (method.get("error") or "").split("\n")[0][:160] or None,
                }
            )
        metric = next(
            (m for m in PRIMARY if any(m in r["metrics"] for r in rows)),
            None,
        )
        models.append(
            {
                "name": card["name"],
                "title": card["title"],
                "task": _task(card["tags"]),
                "parameters": card.get("parameters"),
                "reference": bench.get("reference"),
                "primary": metric,
                "workload": bench.get("workload", {}),
                "rows": rows,
            }
        )
    return {"date": date, "environment": environment, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("nest", type=Path, help="a checkout of the model zoo")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "zoo.json")
    args = parser.parse_args()
    summary = collect(args.nest)
    args.out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{len(summary['models'])} models -> {args.out}")


if __name__ == "__main__":
    main()
