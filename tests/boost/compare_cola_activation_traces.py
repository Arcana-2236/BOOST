import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _safe_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a = a.detach().float()
    b = b.detach().float()
    d = (a - b).abs()
    max_abs = float(d.max().item()) if d.numel() > 0 else 0.0
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
    max_rel = float((d / denom).max().item()) if d.numel() > 0 else 0.0
    return max_abs, max_rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare activation traces dumped from CoLA model.")
    parser.add_argument("--trace-a", required=True)
    parser.add_argument("--trace-b", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    a: Dict[str, torch.Tensor] = _safe_load(args.trace_a)
    b: Dict[str, torch.Tensor] = _safe_load(args.trace_b)

    keys_a = set(a.keys())
    keys_b = set(b.keys())
    missing_a = sorted(keys_b - keys_a)
    missing_b = sorted(keys_a - keys_b)
    common = sorted(keys_a & keys_b)

    lines: List[str] = [
        f"trace_a={Path(args.trace_a)}",
        f"trace_b={Path(args.trace_b)}",
        f"atol={args.atol} rtol={args.rtol}",
        f"n_common={len(common)} n_missing_in_a={len(missing_a)} n_missing_in_b={len(missing_b)}",
        "",
        "Per-activation diffs:",
    ]
    failures: List[str] = []

    for k in common:
        ta = a[k]
        tb = b[k]
        if tuple(ta.shape) != tuple(tb.shape):
            failures.append(f"{k}: shape mismatch A={tuple(ta.shape)} B={tuple(tb.shape)}")
            lines.append(f"- {k}: shape_mismatch")
            continue
        abs_d, rel_d = _tensor_diff(ta, tb)
        lines.append(f"- {k}: max_abs={abs_d:.6e}, max_rel={rel_d:.6e}")
        if abs_d > args.atol and rel_d > args.rtol:
            failures.append(f"{k}: abs={abs_d:.6e} rel={rel_d:.6e}")

    if missing_a:
        failures.append(f"missing in A: {missing_a[:20]}")
    if missing_b:
        failures.append(f"missing in B: {missing_b[:20]}")

    lines.append("")
    if failures:
        lines.append("Failures:")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")
        lines.append("FAIL")
    else:
        lines.append("PASS")

    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
