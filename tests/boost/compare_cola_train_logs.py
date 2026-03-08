import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


ITER_RE = re.compile(r"iteration:\s*(\d+)\s*/")
KV_RE = re.compile(r"([a-zA-Z_]+):\s*([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")


def _parse_log(log_path: Path) -> Dict[int, Dict[str, float]]:
    metrics_by_step: Dict[int, Dict[str, float]] = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        if "iteration:" not in line:
            continue
        m_iter = ITER_RE.search(line)
        if not m_iter:
            continue
        step = int(m_iter.group(1))
        values: Dict[str, float] = {}
        for m_kv in KV_RE.finditer(line):
            key = m_kv.group(1)
            # Keep only stable training-signal keys.
            if key not in {"lm_loss", "grad_norm", "lr"}:
                continue
            try:
                values[key] = float(m_kv.group(2))
            except ValueError:
                continue
        if values:
            metrics_by_step[step] = values
    return metrics_by_step


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _format(v: float) -> str:
    return f"{v:.6e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two CoLA training logs and report first divergence step."
    )
    parser.add_argument("--log-a", required=True, help="Path to baseline log (e.g. TP1).")
    parser.add_argument("--log-b", required=True, help="Path to comparison log (e.g. TP4).")
    parser.add_argument("--keys", default="lm_loss,grad_norm,lr", help="Comma-separated metric keys.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for divergence.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for divergence.")
    parser.add_argument(
        "--report-dir",
        default="/home/zhengyangwang/nanotron/.logging/log-compare",
        help="Directory to store the comparison report.",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional report filename. Defaults to auto-generated timestamped name.",
    )
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    log_a = Path(args.log_a)
    log_b = Path(args.log_b)

    steps_a = _parse_log(log_a)
    steps_b = _parse_log(log_b)
    common_steps = sorted(set(steps_a.keys()) & set(steps_b.keys()))

    if not common_steps:
        raise SystemExit("No overlapping iteration lines found between logs.")

    report_lines = []
    report_lines.append(f"Parsed steps: A={len(steps_a)} B={len(steps_b)} common={len(common_steps)}")
    report_lines.append(f"Keys: {', '.join(keys)}")
    report_lines.append(f"Tolerances: atol={args.atol} rtol={args.rtol}")

    first_divergence: Optional[Dict[str, float]] = None
    max_stats: Dict[str, Dict[str, float]] = {
        k: {"max_abs": 0.0, "max_rel": 0.0, "step_abs": -1, "step_rel": -1} for k in keys
    }

    for step in common_steps:
        va = steps_a[step]
        vb = steps_b[step]
        for key in keys:
            if key not in va or key not in vb:
                continue
            a = va[key]
            b = vb[key]
            abs_d = abs(a - b)
            rel_d = _rel_diff(a, b)
            if abs_d > max_stats[key]["max_abs"]:
                max_stats[key]["max_abs"] = abs_d
                max_stats[key]["step_abs"] = step
            if rel_d > max_stats[key]["max_rel"]:
                max_stats[key]["max_rel"] = rel_d
                max_stats[key]["step_rel"] = step

            # mark first divergence when either abs or rel breaks threshold
            if first_divergence is None and abs_d > args.atol and rel_d > args.rtol:
                first_divergence = {
                    "step": step,
                    "key": key,
                    "a": a,
                    "b": b,
                    "abs": abs_d,
                    "rel": rel_d,
                }

    report_lines.append("")
    report_lines.append("Max diff summary:")
    for key in keys:
        s = max_stats[key]
        report_lines.append(
            f"- {key}: "
            f"max_abs={_format(s['max_abs'])} (step={int(s['step_abs'])}), "
            f"max_rel={_format(s['max_rel'])} (step={int(s['step_rel'])})"
        )

    if first_divergence is None:
        report_lines.append("")
        report_lines.append("No divergence found within thresholds on common steps.")
    else:
        report_lines.append("")
        report_lines.append("First divergence:")
        report_lines.append(
            f"- step={int(first_divergence['step'])} key={first_divergence['key']} "
            f"A={_format(first_divergence['a'])} B={_format(first_divergence['b'])} "
            f"abs={_format(first_divergence['abs'])} rel={_format(first_divergence['rel'])}"
        )

    # Print to stdout and also save a report file under .logging.
    text = "\n".join(report_lines)
    print(text)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.report_name:
        report_name = args.report_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"compare_{ts}.txt"
    report_path = report_dir / report_name
    report_path.write_text(text + "\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
