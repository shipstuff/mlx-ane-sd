#!/bin/bash
# Measure SoC power during: (a) pure-MLX SD, (b) heterogeneous ANE+MLX SD
# Both generating ~500 tokens on Gemma-3-12B bf16 so sustained load dominates
# warmup noise.
set -e

PY=/Users/carl/models/mlx-venv/bin/python
OUT_DIR=/tmp/power_compare
mkdir -p "$OUT_DIR"

# Inline Python runners — driven externally by powermetrics
cat > /tmp/sustained_mlx_sd.py << 'EOF'
import time
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
target, tok = load("mlx-community/gemma-3-12b-it-bf16")
draft, _ = load("mlx-community/gemma-3-270m-it-bf16")
sampler = make_sampler(temp=0.0)
print("READY", flush=True)
time.sleep(5)
for prompt in ["The capital of France is", "Write a Python function", "def fibonacci(n):"]:
    tokens = []
    for resp in stream_generate(target, tok, prompt, max_tokens=150, sampler=sampler,
                                 draft_model=draft, num_draft_tokens=12):
        tokens.append(resp.token)
print(f"DONE {len(tokens)} tok", flush=True)
EOF

cat > /tmp/sustained_ane_sd.py << 'EOF'
import subprocess, sys, time
subprocess.Popen([sys.executable, "/Users/carl/projects/mlx-ane-sd/scripts/phaseB_sequential_optimized.py",
                  "--num-draft", "12", "--max-new-tokens", "450",
                  "--prompt", "The capital of France is",
                  "--skip-baseline", "--quiet"], env={**__import__('os').environ})
EOF

run_powermetrics() {
    local label=$1
    local pyscript=$2
    echo "=== $label ==="
    # Start the Python process in the background
    $PY $pyscript > $OUT_DIR/${label}.log 2>&1 &
    local pypid=$!
    # Wait until we see READY or give up after 120s
    for i in {1..120}; do
        if grep -q "READY" $OUT_DIR/${label}.log 2>/dev/null; then
            break
        fi
        sleep 1
    done
    sleep 2  # warmup settle
    # Record powermetrics for 30 seconds while the python runs
    sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power \
        -i 500 --sample-count 60 > $OUT_DIR/${label}_power.txt 2>&1
    # Wait for python to finish
    wait $pypid || true
    echo "$label done"
}

# Run the actual heterogeneous one via the existing Phase B.1 script but with
# a longer max-tokens and READY signal
cat > /tmp/phaseb_ready.py << 'EOF'
import sys, time, os
sys.path.insert(0, '/Users/carl/projects/mlx-ane-sd/scripts')
# Import from phaseB
import phaseB_sequential_optimized as pb
from mlx_lm import load

target_model_name = "mlx-community/gemma-3-12b-it-bf16"
target = pb.MLXTarget(target_model_name)
draft = pb.ANEDraft()
prompt_ids = target.tokenizer.encode("The capital of France is", add_special_tokens=True)
print("READY", flush=True)
time.sleep(5)
# Run sustained workload
for _ in range(3):
    gen, stats = pb.run_sd(draft, target, prompt_ids, 150, 12, verbose=False)
print("DONE", flush=True)
EOF

run_powermetrics "mlx_only" "/tmp/sustained_mlx_sd.py"
sleep 3
run_powermetrics "heterogeneous" "/tmp/phaseb_ready.py"

echo ""
echo "--- Parsing results ---"
for label in mlx_only heterogeneous; do
    if [ -f $OUT_DIR/${label}_power.txt ]; then
        echo ""
        echo "=== $label ==="
        $PY - <<PYPARSE
import re
text = open("$OUT_DIR/${label}_power.txt").read()
cpu = [int(m.group(1)) for m in re.finditer(r"^CPU Power: (\d+) mW", text, re.MULTILINE)]
gpu = [int(m.group(1)) for m in re.finditer(r"^GPU Power: (\d+) mW", text, re.MULTILINE)]
ane = [int(m.group(1)) for m in re.finditer(r"^ANE Power: (\d+) mW", text, re.MULTILINE)]
combined = [c+g+a for c,g,a in zip(cpu, gpu, ane)]
def stats(v, name):
    if not v: return f"  {name}: no data"
    import statistics as st
    return f"  {name}: mean={st.mean(v)/1000:.2f}W p50={sorted(v)[len(v)//2]/1000:.2f}W max={max(v)/1000:.2f}W n={len(v)}"
print(stats(cpu, "CPU    "))
print(stats(gpu, "GPU    "))
print(stats(ane, "ANE    "))
print(stats(combined, "SoC    "))
PYPARSE
    fi
done
