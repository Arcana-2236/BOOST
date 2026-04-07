#!/usr/bin/env bash
set -euo pipefail

# Compare iteration runtime for FullRank, Vanilla CoLA TP, and BTP CoLA TP
# at fixed micro-batch-size=1 for:
#   3B: TP=2
#   7B: TP=4
#
# Outputs:
#   - logs:    .logging/iter_comp_mbz1/*.log
#   - configs: .logging/iter_comp_mbz1/configs/*.yaml
#   - table:   .logging/iter_comp_mbz1/iter_compare_mbz1_table.txt

cd "$(dirname "${BASH_SOURCE[0]}")"

export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

OUT_DIR=".logging/iter_comp_mbz1"
CFG_DIR="${OUT_DIR}/configs"
mkdir -p "${OUT_DIR}" "${CFG_DIR}"

BASE_FR_CFG="examples/config_llama_7b.yaml"
BASE_COLA_CFG="examples/cola/config_cola_llama_7b.yaml"

make_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local tp="$3"
  local mbz="$4"
  local hidden="$5"
  local intermediate="$6"
  local n_layers="$7"
  local n_heads="$8"
  local n_kv_heads="$9"
  local attn_rank="${10}"
  local mlp_rank="${11}"
  local is_cola="${12}"
  local run_name="${13}"

  python - "$base_cfg" "$out_cfg" "$tp" "$mbz" "$hidden" "$intermediate" \
    "$n_layers" "$n_heads" "$n_kv_heads" "$attn_rank" "$mlp_rank" "$is_cola" "$run_name" <<'PY'
import pathlib
import re
import sys

(
    base_cfg,
    out_cfg,
    tp,
    mbz,
    hidden,
    intermediate,
    n_layers,
    n_heads,
    n_kv_heads,
    attn_rank,
    mlp_rank,
    is_cola,
    run_name,
) = sys.argv[1:]

text = pathlib.Path(base_cfg).read_text()

def replace_key(txt: str, key: str, value: str) -> str:
    pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
    out, count = re.subn(pattern, rf"\g<1>{value}", txt, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"Could not find key '{key}' in {base_cfg}")
    return out

# Parallelism / token settings
text = replace_key(text, "tp", tp)
text = replace_key(text, "micro_batch_size", mbz)
text = replace_key(text, "batch_accumulation_per_replica", "1")
text = replace_key(text, "val_check_interval", "-1")
text = replace_key(text, "limit_val_batches", "0")

# Keep tokenizer local/offline-friendly.
text = replace_key(text, "tokenizer_name_or_path", "robot-test/dummy-tokenizer-wordlevel")

# Model shape
text = replace_key(text, "hidden_size", hidden)
text = replace_key(text, "intermediate_size", intermediate)
text = replace_key(text, "num_hidden_layers", n_layers)
text = replace_key(text, "num_attention_heads", n_heads)
text = replace_key(text, "num_key_value_heads", n_kv_heads)

if is_cola == "1":
    text = replace_key(text, "attn_rank", attn_rank)
    text = replace_key(text, "mlp_rank", mlp_rank)

text = replace_key(text, "run", run_name)

pathlib.Path(out_cfg).write_text(text)
PY
}

run_cmd() {
  local name="$1"
  shift
  local log_path="${OUT_DIR}/${name}.log"
  echo
  echo "=== Running ${name} ==="
  "$@" 2>&1 | tee "${log_path}"
  echo "=== Finished ${name}; log: ${log_path} ==="
}

# model|tp|hidden|intermediate|layers|heads|kv_heads|rank
SPECS=(
  "3B|2|3072|8192|26|24|24|768"
  "7B|4|4096|11008|32|32|32|1024"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r model tp hidden intermediate n_layers n_heads n_kv_heads rank <<< "${spec}"

  fr_cfg="${CFG_DIR}/config_llama_${model}_tp${tp}_mbz1.yaml"
  cola_cfg="${CFG_DIR}/config_cola_llama_${model}_tp${tp}_mbz1.yaml"

  make_cfg "${BASE_FR_CFG}" "${fr_cfg}" "${tp}" "1" \
    "${hidden}" "${intermediate}" "${n_layers}" "${n_heads}" "${n_kv_heads}" \
    "${rank}" "${rank}" "0" "iter_cmp_fr_${model}_tp${tp}_mbz1_%date_%jobid"

  make_cfg "${BASE_COLA_CFG}" "${cola_cfg}" "${tp}" "1" \
    "${hidden}" "${intermediate}" "${n_layers}" "${n_heads}" "${n_kv_heads}" \
    "${rank}" "${rank}" "1" "iter_cmp_cola_${model}_tp${tp}_mbz1_%date_%jobid"

  run_cmd "fr_${model,,}_tp${tp}_mbz1" \
    torchrun --nproc_per_node="${tp}" run_train.py --config-file "${fr_cfg}"

  run_cmd "btp_${model,,}_tp${tp}_mbz1" \
    torchrun --nproc_per_node="${tp}" examples/cola/train_cola.py --config-file "${cola_cfg}"

  run_cmd "vanilla_${model,,}_tp${tp}_mbz1" \
    torchrun --nproc_per_node="${tp}" examples/cola/train_vanilla_cola.py --config-file "${cola_cfg}"
done

python - "$OUT_DIR" <<'PY'
import pathlib
import re
import statistics
import sys

out_dir = pathlib.Path(sys.argv[1])
result_path = out_dir / "iter_compare_mbz1_table.txt"

pattern = re.compile(
    r"iteration:\s*(\d+)\s*/\s*(\d+)\s*\|.*?elapsed_time_per_iteration_ms:\s*([\d.]+)([KkMm]?)"
)

def parse_avg_ms(path: pathlib.Path):
    if not path.is_file():
        return None, "missing log"
    text = path.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        it = int(m.group(1))
        val = float(m.group(3))
        suffix = m.group(4).lower()
        if suffix == "k":
            val *= 1_000.0
        elif suffix == "m":
            val *= 1_000_000.0
        rows.append((it, val))
    if len(rows) <= 2:
        return None, "insufficient data"
    vals = [v for it, v in rows if it > 2]  # drop 2-step warmup
    if not vals:
        return None, "insufficient post-warmup data"
    return statistics.mean(vals), f"n={len(vals)}"

specs = [
    ("3B", 2),
    ("7B", 4),
]

def fnum(x, nd=2):
    return "N/A" if x is None else f"{x:.{nd}f}"

lines = []
lines.append("Iteration Runtime Summary (ms), after 2-step warmup")
lines.append("=" * 112)
header = (
    f"{'Model':<8} {'TP':<4} {'MBZ':<5} "
    f"{'FullRank':<12} {'Vanilla TP':<12} {'BTP':<12} "
    f"{'BTP/FR':<10} {'BTP/Vanilla':<12}"
)
lines.append(header)
lines.append("-" * len(header))

notes = []
for model, tp in specs:
    key = model.lower()
    fr, fr_note = parse_avg_ms(out_dir / f"fr_{key}_tp{tp}_mbz1.log")
    btp, btp_note = parse_avg_ms(out_dir / f"btp_{key}_tp{tp}_mbz1.log")
    van, van_note = parse_avg_ms(out_dir / f"vanilla_{key}_tp{tp}_mbz1.log")

    btp_vs_fr = (fr / btp) if (fr is not None and btp is not None and btp > 0) else None
    btp_vs_van = (van / btp) if (van is not None and btp is not None and btp > 0) else None

    lines.append(
        f"{model:<8} {tp:<4} {1:<5} "
        f"{fnum(fr):<12} {fnum(van):<12} {fnum(btp):<12} "
        f"{fnum(btp_vs_fr, 3):<10} {fnum(btp_vs_van, 3):<12}"
    )

    notes.append(f"{model}: FR({fr_note}), Vanilla({van_note}), BTP({btp_note})")

lines.append("")
lines.append("Notes:")
for n in notes:
    lines.append(f"- {n}")

text = "\n".join(lines) + "\n"
print("\n" + text)
result_path.write_text(text)
print(f"Saved table to: {result_path}")
PY

