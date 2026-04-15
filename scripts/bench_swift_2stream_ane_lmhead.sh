#!/bin/bash
# Re-run the 2-stream contention bench with ANE-hosted lm_head.
# Compares to the GPU-lmhead baseline from bench_swift_2stream.sh.
cd /Users/carl/projects/mlx-ane-sd/swift-bench
BIN=.build/release/dflash-sd
MLMODELC=/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc
LMHEAD=/tmp/lmhead_qwen3/lmhead_lut6.mlmodelc

# Warmup
$BIN --prompt "Hello world" --max-new 20 --draft $MLMODELC --ane-lmhead $LMHEAD --json > /dev/null 2>&1 &
wait $!

echo "=== 1 stream baseline (ANE lm_head) ==="
$BIN --prompt "The capital of France is Paris, which is known for" --max-new 100 --draft $MLMODELC --ane-lmhead $LMHEAD --json > /tmp/solo_ane.json 2>&1
python3 -c "
import json
d = json.load(open('/tmp/solo_ane.json'))
p = d['phases']
print(f\"solo: {d['tok_per_s_decode']:.2f} tok/s, {d['tokens']} tok, {d['cycles']} cyc\")
print(f\"  target_verify: {p['target_verify']['meanMs']:.1f}ms/cyc\")
print(f\"  draft_lmhead:  {p['draft_lmhead']['meanMs']:.2f}ms/cyc (ANE)\")
print(f\"  draft_predict: {p['draft_predict']['meanMs']:.1f}ms/cyc (ANE)\")
"

echo ""
echo "=== 2 streams in parallel (ANE lm_head) ==="
START=$(python3 -c "import time; print(time.time())")
$BIN --prompt "The capital of France is Paris, which is known for" --max-new 100 --draft $MLMODELC --ane-lmhead $LMHEAD --json > /tmp/s1_ane.json 2>&1 &
pid1=$!
$BIN --prompt "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who" --max-new 100 --draft $MLMODELC --ane-lmhead $LMHEAD --json > /tmp/s2_ane.json 2>&1 &
pid2=$!
wait $pid1 $pid2
END=$(python3 -c "import time; print(time.time())")
ELAPSED=$(python3 -c "print($END - $START)")

python3 <<PYEOF
import json
s1 = json.load(open('/tmp/s1_ane.json'))
s2 = json.load(open('/tmp/s2_ane.json'))
solo = json.load(open('/tmp/solo_ane.json'))

total_tok = s1['tokens'] + s2['tokens']
elapsed = $ELAPSED
agg_tps = total_tok / elapsed
speedup = agg_tps / solo['tok_per_s_decode']

print(f"stream1: {s1['tok_per_s_decode']:.2f} tok/s decode, {s1['tokens']} tok, {s1['cycles']} cyc")
print(f"stream2: {s2['tok_per_s_decode']:.2f} tok/s decode, {s2['tokens']} tok, {s2['cycles']} cyc")
print(f"2-stream aggregate: {total_tok}/{elapsed:.2f}s = {agg_tps:.2f} tok/s")
print(f"vs solo ANE lm_head: {speedup:.3f}x aggregate")
print()
print("Per-phase behavior under contention:")
for label, d in [('SOLO', solo), ('STREAM1 parallel', s1), ('STREAM2 parallel', s2)]:
    p = d['phases']
    print(f"  {label}: tv={p['target_verify']['meanMs']:.1f}ms  "
          f"dlh={p['draft_lmhead']['meanMs']:.2f}ms  "
          f"dp={p['draft_predict']['meanMs']:.1f}ms  "
          f"cycles={d['cycles']}")

print()
print("Compare to GPU-lm_head baseline: 1.28x aggregate")
print(f"ANE lm_head gain: {(speedup/1.28 - 1)*100:+.1f}%")
PYEOF
