#!/bin/bash
# Run two Swift dflash-sd instances in parallel, measure aggregate tok/s.
cd /Users/carl/projects/mlx-ane-sd/swift-bench
BIN=.build/release/dflash-sd
MLMODELC=/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc

# Warmup both
$BIN --prompt "Hello world" --max-new 20 --draft $MLMODELC --json > /dev/null 2>&1 &
pid1=$!
wait $pid1

echo "=== 1 stream baseline ==="
/usr/bin/time -p $BIN --prompt "The capital of France is Paris, which is known for" --max-new 100 --draft $MLMODELC --json 2>&1 | tee /tmp/solo.json | head -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(f\"solo: {d['tok_per_s_decode']:.2f} tok/s, {d['tokens']} tok, {d['cycles']} cyc\")"

echo ""
echo "=== 2 streams in parallel ==="
START=$(python3 -c "import time; print(time.time())")
$BIN --prompt "The capital of France is Paris, which is known for" --max-new 100 --draft $MLMODELC --json > /tmp/s1.json 2>&1 &
pid1=$!
$BIN --prompt "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who" --max-new 100 --draft $MLMODELC --json > /tmp/s2.json 2>&1 &
pid2=$!
wait $pid1 $pid2
END=$(python3 -c "import time; print(time.time())")
ELAPSED=$(python3 -c "print($END - $START)")

python3 <<PYEOF
import json
s1 = json.load(open('/tmp/s1.json'))
s2 = json.load(open('/tmp/s2.json'))
total_tok = s1['tokens'] + s2['tokens']
elapsed = $ELAPSED
print(f"stream1: {s1['tok_per_s_decode']:.2f} tok/s solo-measured, {s1['tokens']} tok, {s1['cycles']} cyc")
print(f"stream2: {s2['tok_per_s_decode']:.2f} tok/s solo-measured, {s2['tokens']} tok, {s2['cycles']} cyc")
print(f"2-stream aggregate: {total_tok}/{elapsed:.2f}s = {total_tok/elapsed:.2f} tok/s")
print(f"(solo aggregate would be: 2x solo tok/s)")
PYEOF
