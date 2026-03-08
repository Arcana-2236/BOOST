import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a = a.detach().float()
    b = b.detach().float()
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0
    return max_abs, max_rel


def _compare_tensor_dict(
    name: str, da: Dict[str, torch.Tensor], db: Dict[str, torch.Tensor], atol: float, rtol: float
) -> Tuple[float, float, str, str, int]:
    keys_a = set(da.keys())
    keys_b = set(db.keys())
    if keys_a != keys_b:
        missing_a = sorted(keys_b - keys_a)
        missing_b = sorted(keys_a - keys_b)
        raise AssertionError(
            f"{name}: key mismatch. missing_in_A={missing_a[:10]} missing_in_B={missing_b[:10]}"
        )

    max_abs = 0.0
    max_rel = 0.0
    worst_key_abs = ""
    worst_key_rel = ""
    n_exceed = 0
    for k in sorted(keys_a):
        ta = da[k]
        tb = db[k]
        if tuple(ta.shape) != tuple(tb.shape):
            raise AssertionError(f"{name}:{k} shape mismatch A={tuple(ta.shape)} B={tuple(tb.shape)}")
        abs_d, rel_d = _tensor_diff(ta, tb)
        if abs_d > max_abs:
            max_abs = abs_d
            worst_key_abs = k
        if rel_d > max_rel:
            max_rel = rel_d
            worst_key_rel = k
        if abs_d > atol and rel_d > rtol:
            n_exceed += 1
    return max_abs, max_rel, worst_key_abs, worst_key_rel, n_exceed


def _safe_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch without weights_only argument.
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict step-0 tensor dump artifacts from two runs.")
    parser.add_argument("--dump-a", required=True)
    parser.add_argument("--dump-b", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    a = _safe_load(args.dump_a)
    b = _safe_load(args.dump_b)

    lines = []
    failures: List[str] = []
    lines.append(f"dump_a={Path(args.dump_a)}")
    lines.append(f"dump_b={Path(args.dump_b)}")
    lines.append(f"atol={args.atol} rtol={args.rtol}")
    lines.append("mode=compare-all (do not stop at first mismatch)")

    # 1) input & weight after init before iter 0
    try:
        input_stats = _compare_tensor_dict(
            "input_batch_before_iter0",
            a["input_batch_before_iter0"],
            b["input_batch_before_iter0"],
            atol=0.0,
            rtol=0.0,
        )
        if input_stats[4] > 0:
            failures.append(
                f"input_batch_before_iter0 has {input_stats[4]} tensors exceeding strict equality."
            )
    except Exception as exc:
        input_stats = None
        failures.append(f"input_batch_before_iter0: {exc}")

    try:
        wpre_stats = _compare_tensor_dict(
            "weights_after_init_before_iter0",
            a["weights_after_init_before_iter0"],
            b["weights_after_init_before_iter0"],
            atol=args.atol,
            rtol=args.rtol,
        )
        if wpre_stats[4] > 0:
            failures.append(
                f"weights_after_init_before_iter0 has {wpre_stats[4]} tensors exceeding thresholds."
            )
    except Exception as exc:
        wpre_stats = None
        failures.append(f"weights_after_init_before_iter0: {exc}")

    # 2) forward output tensor before computing loss
    try:
        fa = a.get("forward_output_before_loss")
        fb = b.get("forward_output_before_loss")
        if fa is None or fb is None:
            raise AssertionError("missing. Enable COLA_STRICT_TENSOR_DUMP_INCLUDE_LOGITS=1.")
        f_abs, f_rel = _tensor_diff(fa, fb)
        if f_abs > args.atol and f_rel > args.rtol:
            failures.append(f"forward_output_before_loss diverged abs={f_abs:.6e} rel={f_rel:.6e}")
    except Exception as exc:
        f_abs, f_rel = float("nan"), float("nan")
        failures.append(f"forward_output_before_loss: {exc}")

    # 3) lm_loss
    try:
        lm_a = float(a["lm_loss"])
        lm_b = float(b["lm_loss"])
        lm_abs = abs(lm_a - lm_b)
        lm_rel = lm_abs / max(abs(lm_a), abs(lm_b), 1e-12)
        if lm_abs > args.atol and lm_rel > args.rtol:
            failures.append(f"lm_loss diverged A={lm_a:.8e} B={lm_b:.8e} abs={lm_abs:.6e} rel={lm_rel:.6e}")
    except Exception as exc:
        lm_a, lm_b, lm_abs, lm_rel = float("nan"), float("nan"), float("nan"), float("nan")
        failures.append(f"lm_loss: {exc}")

    # 4) grad_norm, lr
    try:
        gn_a = float(a["grad_norm"])
        gn_b = float(b["grad_norm"])
        gn_abs = abs(gn_a - gn_b)
        gn_rel = gn_abs / max(abs(gn_a), abs(gn_b), 1e-12)
        lr_a = float(a["lr"])
        lr_b = float(b["lr"])
        lr_abs = abs(lr_a - lr_b)
        lr_rel = lr_abs / max(abs(lr_a), abs(lr_b), 1e-12)
        if gn_abs > args.atol and gn_rel > args.rtol:
            failures.append(f"grad_norm diverged A={gn_a:.8e} B={gn_b:.8e} abs={gn_abs:.6e} rel={gn_rel:.6e}")
        if lr_abs > args.atol and lr_rel > args.rtol:
            failures.append(f"lr diverged A={lr_a:.8e} B={lr_b:.8e} abs={lr_abs:.6e} rel={lr_rel:.6e}")
    except Exception as exc:
        gn_a, gn_b, gn_abs, gn_rel = float("nan"), float("nan"), float("nan"), float("nan")
        lr_a, lr_b, lr_abs, lr_rel = float("nan"), float("nan"), float("nan"), float("nan")
        failures.append(f"grad_norm/lr: {exc}")

    # 5) grad
    try:
        g_stats = _compare_tensor_dict(
            "grad_before_step0_update",
            a["grad_before_step0_update"],
            b["grad_before_step0_update"],
            atol=args.atol,
            rtol=args.rtol,
        )
        if g_stats[4] > 0:
            failures.append(f"grad_before_step0_update has {g_stats[4]} tensors exceeding thresholds.")
    except Exception as exc:
        g_stats = None
        failures.append(f"grad_before_step0_update: {exc}")

    # 6) weight after step 0 optimizer update
    try:
        wpost_stats = _compare_tensor_dict(
            "weight_after_step0_optimizer_update",
            a["weight_after_step0_optimizer_update"],
            b["weight_after_step0_optimizer_update"],
            atol=args.atol,
            rtol=args.rtol,
        )
        if wpost_stats[4] > 0:
            failures.append(
                f"weight_after_step0_optimizer_update has {wpost_stats[4]} tensors exceeding thresholds."
            )
    except Exception as exc:
        wpost_stats = None
        failures.append(f"weight_after_step0_optimizer_update: {exc}")

    lines.append("")
    lines.append("Checks:")
    if input_stats is not None:
        lines.append(
            f"- input_batch_before_iter0: max_abs={input_stats[0]:.6e}, max_rel={input_stats[1]:.6e}, worst_abs={input_stats[2]}, n_exceed={input_stats[4]}"
        )
    else:
        lines.append("- input_batch_before_iter0: ERROR")
    if wpre_stats is not None:
        lines.append(
            f"- weights_after_init_before_iter0: max_abs={wpre_stats[0]:.6e}, max_rel={wpre_stats[1]:.6e}, worst_abs={wpre_stats[2]}, n_exceed={wpre_stats[4]}"
        )
    else:
        lines.append("- weights_after_init_before_iter0: ERROR")
    lines.append(f"- forward_output_before_loss: max_abs={f_abs:.6e}, max_rel={f_rel:.6e}")
    lines.append(f"- lm_loss: A={lm_a:.8e} B={lm_b:.8e} abs={lm_abs:.6e} rel={lm_rel:.6e}")
    lines.append(f"- grad_norm: A={gn_a:.8e} B={gn_b:.8e} abs={gn_abs:.6e} rel={gn_rel:.6e}")
    lines.append(f"- lr: A={lr_a:.8e} B={lr_b:.8e} abs={lr_abs:.6e} rel={lr_rel:.6e}")
    if g_stats is not None:
        lines.append(
            f"- grad_before_step0_update: max_abs={g_stats[0]:.6e}, max_rel={g_stats[1]:.6e}, worst_abs={g_stats[2]}, n_exceed={g_stats[4]}"
        )
    else:
        lines.append("- grad_before_step0_update: ERROR")
    if wpost_stats is not None:
        lines.append(
            f"- weight_after_step0_optimizer_update: max_abs={wpost_stats[0]:.6e}, max_rel={wpost_stats[1]:.6e}, worst_abs={wpost_stats[2]}, n_exceed={wpost_stats[4]}"
        )
    else:
        lines.append("- weight_after_step0_optimizer_update: ERROR")
    lines.append("")
    if failures:
        lines.append("Failures:")
        for item in failures:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("FAIL")
    else:
        lines.append("PASS")

    report = "\n".join(lines)
    print(report)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
