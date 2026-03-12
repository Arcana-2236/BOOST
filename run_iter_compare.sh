#!/usr/bin/env bash
set -euo pipefail

# Run three training commands, save logs under .logging/iter_comp,
# then summarize average iteration time (ms) after 2-step warmup.

wandb disabled
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT_DIR=".logging/iter_comp"
mkdir -p "${OUT_DIR}"

run_cmd() {
  local name="$1"
  shift
  local log="${OUT_DIR}/${name}.log"

  echo "=== Running ${name} ==="
  CUDA_DEVICE_MAX_CONNECTIONS=1 "$@" 2>&1 | tee "${log}"
  echo "=== Finished ${name}, log saved to ${log} ==="
}

# 1) Pretrain run
run_cmd "fr_llama7b" \
  torchrun --nproc_per_node=4 run_train.py \
  --config-file examples/config_llama_7b.yaml

# 2) CoLA finetune run
run_cmd "btp_cola7b" \
  torchrun --nproc_per_node=4 examples/cola/train_cola.py \
  --config-file examples/cola/config_cola_llama_7b.yaml

# 3) Vanilla CoLA 1B TP=4 run
run_cmd "vanilla_cola_7b" \
  torchrun --nproc_per_node=4 examples/cola/train_vanilla_cola.py \
  --config-file examples/cola/config_cola_llama_7b.yaml

python - "$OUT_DIR" << 'PY'
import sys
import pathlib
import re
import statistics

log_dir = pathlib.Path(sys.argv[1])

pattern = re.compile(
    r"iteration:\s*(\d+)\s*/\s*(\d+)\s*\|.*?elapsed_time_per_iteration_ms:\s*([\d.]+)([KkMm]?)"
)


def parse_file(path: pathlib.Path):
    iters = []
    try:
        text = path.read_text(errors="ignore")
    except Exception as e:
        print(f"- {path.name}: failed to read log ({e})")
        return None

    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        it = int(m.group(1))
        val = float(m.group(3))
        suffix = m.group(4)
        if suffix in ("K", "k"):
            val *= 1_000.0
        elif suffix in ("M", "m"):
            val *= 1_000_000.0
        iters.append((it, val))

    # Require at least 3 iterations to have anything after 2-step warmup
    if len(iters) <= 2:
        return None

    # Drop first 2 iterations (warmup)
    filtered = [v for i, v in iters if i > 2]
    if not filtered:
        return None

    avg = statistics.mean(filtered)
    return avg, len(filtered)


runs = [
    ("fr_llama7b", "fr_llama7b.log"),
    ("btp_cola7b", "btp_cola7b.log"),
    ("vanilla_cola_7b", "vanilla_cola_7b.log"),
]

lines = []
lines.append("Average iteration time (ms) after 2-step warmup:")
for label, fname in runs:
    p = log_dir / fname
    if not p.is_file():
        lines.append(f"- {label}: log file missing at {p}")
        continue
    res = parse_file(p)
    if res is None:
        lines.append(f"- {label}: insufficient iteration data")
        continue
    avg, n = res
    lines.append(f"- {label}: {avg:.2f} ms over {n} iterations")

text = "\n".join(lines)
print("\n" + text)

result_path = log_dir / "result.txt"
try:
    result_path.write_text(text + "\n")
except Exception as e:
    print(f"\nWarning: failed to write result.txt ({e})")
PY

