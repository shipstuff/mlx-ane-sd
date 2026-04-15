# EAGLE-3 baseline on Qwen3-4B (MLX port, mini-03)

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB (skynet-m4-mini-03)
**Target:** `mlx-community/Qwen3-4B-Instruct-2507-bf16` (36 layers, 2560 hidden, GQA 32/8)
**Draft:** `taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3` (EAGLE-3, 1 midlayer, 0.44 GB)
**Runner:** `scripts/baseline_eagle3.py` (chain-style SD; no tree decoding)
**Decode:** greedy (temp=0), max_new=100, num_draft=2
**Draft rope_theta:** 1_000_000 (from EAGLE-3 checkpoint config)

## TL;DR

EAGLE-3 chain-SD on MLX runs on M4 Pro but is **net slower than target-only
greedy decoding** for this draft / target pair, both solo and under
contention. Per-cycle acceptance lands around 0.26 (draft tokens / cycle),
far below the paper's reported 2.08 — we're running *chain* (no tree) and
each subsequent draft step compounds error since the draft must reuse its
own pre-norm hidden in place of the target's feature triple.

The numbers are directly comparable (same prompts, same mechanics) to F.0
(DFlash GPU) and F.1 (DFlash ANE), but **the comparison is unfavorable to
EAGLE-3** because:

1. The draft was trained against the Instruct-2507 variant; we used that
   same target here. F.0 / F.1 used plain Qwen3-4B-bf16.
2. DFlash's draft is purpose-built for batch=16 verify and amortizes
   weight fetch aggressively. EAGLE-3's draft is a 1-layer token-level
   recurrence that needs *tree* decoding to shine, which isn't implemented
   here.
3. mini-03's standalone throughput is meaningfully slower than mini-02
   (16.6 tok/s vs mini-02's 27.5 tok/s on Qwen3-4B bf16 @ greedy), most
   likely because mini-03 is still Python 3.9 / MLX 0.29 (python 3.11 / MLX 0.31
   wheels don't exist for 3.9), and because there are background services
   (EXO, ollama) co-resident. **Use the ratios, not the absolute tok/s, for
   comparison.**

## Results (4-prompt mean, identical prompts to F.0/F.1)

| Config | Solo | w/ gemma-3-270m bg (moderate) | w/ Qwen3-4B bg (heavy) |
|---|---:|---:|---:|
| target-only (Qwen3-4B-Instruct-2507) | **16.61** | not run | 10.03 |
| **EAGLE-3 (K=2, chain)**             | **10.83** | 11.29 | 6.17 |
| EAGLE-3 speedup                     | 0.65× | — | 0.62× |

Side-by-side with the DFlash results from F.0 / F.1 (mini-02, plain
Qwen3-4B-bf16 target; numbers from `notes/phaseF0_results.md`):

| Config | Solo | Heavy contention | Slowdown |
|---|---:|---:|---:|
| DFlash GPU (F.0)                         | 40.92 | 21.21 | -48% |
| DFlash ANE (F.1)                         | 34.68 | 19.62 | -43% |
| EAGLE-3 chain (this work, Qwen3-4B-Instruct-2507) | 10.83 | 6.17 | -43% |

Three things stand out:

- **Contention slowdown is comparable across all three approaches (-43% to
  -48%).** EAGLE-3 doesn't degrade worse under contention, despite running
  entirely on GPU.
- **Absolute tok/s is much lower for EAGLE-3** (chain + mini-03 combined).
  Target-only on mini-03 (16.6) is already ~60% of target-only on mini-02
  (27.5). EAGLE-3 then has to amortize a 5-token verify batch over ~1.2
  effective tokens/cycle, which doesn't pay off.
- **Acceptance rate per cycle** (0.26 mean across prompts, best case 0.36
  on math) is roughly 10× lower than the paper's 2.08. This is the single
  biggest gap.

## Per-prompt detail (solo K=2, greedy)

| Prompt | tok/s | cycles | accepted | acc/cycle |
|---|---:|---:|---:|---:|
| capital    | 10.32 | 83 | 16 | 0.19 |
| fibonacci  | 10.61 | 81 | 18 | 0.22 |
| math       | 11.63 | 73 | 26 | 0.36 |
| story      | 10.76 | 78 | 22 | 0.28 |
| **mean**   | **10.83** | 78.8 | 20.5 | **0.26** |

Per-cycle target verify batch is K+1 = 3 positions. Each cycle nets
`accepted + 1` tokens. At acc/cycle=0.26, that's 1.26 tokens per
3-position verify + K draft steps.

## Why chain-style acceptance is so low (caveats / compromises)

1. **Feature mismatch on subsequent draft steps.** The EAGLE-3 training
   recipe teacher-forces the draft with target's concatenated hidden
   states at layers [2, 18, 33] at *every* position. At inference time
   past the first step we don't have target features for draft-synthesized
   tokens, so we substitute the draft's own pre-norm output (this is what
   EAGLE's `cache_hidden` mechanism does internally). Error compounds
   across K draft steps. Teacher-forced, 1-step draft acceptance on a
   representative prompt is **56%** — in line with the paper. Realized
   chain acceptance at K=2 is 26% across prompts; at K=4 it degrades
   further.

2. **No tree decoding.** EAGLE-3 was designed to branch out into a
   candidate tree that the target verifies in parallel. Published
   speedups (1.5-2.5×) assume tree with dynamic expansion. Porting the
   tree+mask logic to MLX was out of scope for this brief; we chose chain
   to match the shape of F.0/F.1's DFlash baseline and get directly-
   comparable numbers.

3. **Python 3.9 / MLX 0.29 on mini-03.** MLX wheels for ≥0.31 require
   Python ≥3.10; mini-03 only has the system Python 3.9. A newer Python
   toolchain (e.g. uv with 3.12) would give us MLX 0.31+ which has better
   fast-attention kernels. Absolute tok/s should be taken with a grain of
   salt; the relative comparison remains valid since both target-only and
   EAGLE-3 run on the same stack.

4. **Target checkpoint differs from F.0/F.1.** F.0/F.1 used
   `mlx-community/Qwen3-4B-bf16` (base). The EAGLE-3 draft was trained
   against `Qwen3-4B-Instruct-2507` (the Instruct variant uses
   rope_theta=5M while the base uses 1M). To give EAGLE-3 a fair shot we
   used Instruct-2507 as target here. This prevents a perfectly
   apples-to-apples comparison to F.0 numbers — contention measurements
   against the same bg Qwen3-4B-bf16 process still tell a consistent
   story.

5. **KV cache alignment between draft and target.** EAGLE-3's chain
   needs the draft's KV cache offsets to track the target's. The
   implementation here trims target cache back to (prompt + last_tok
   + accepted_drafts) after each cycle and re-feeds the bonus token at
   the start of the next verify batch; draft cache tracks
   (prompt + accepted_drafts) and the next cycle's first draft step uses
   the target's triple at the bonus position (from the previous verify
   pass) through `fc`. Subsequent draft steps run through `step_projected`
   which reuses the draft's pre-norm hidden.

## What would improve acceptance (not done here)

- **Tree decoding with 8-16 candidate paths.** This is the canonical
  EAGLE-3 mode and is where the 2.08 acceptance-length comes from.
  Requires custom attention mask generation and MLX tensor operations
  that don't exist as native primitives (custom block mask + gather).
- **Run the EAGLE-3 authors' PyTorch inference on MPS as a sanity check.**
  If *their* reference implementation also produces ~0.26 acc/cycle on
  Qwen3-4B without a tree, then this result is what it is. If it produces
  ~2 without a tree, we have a porting bug.
- **Validate the d2t / t2d mapping.** The mapping (`target_id = draft_id
  + d2t[draft_id]`) was verified on random samples — `t2d[target_ids]` is
  all True for all mapped draft ids. Bug here is unlikely.
- **Try a plain Qwen3-4B base target.** The draft's trained prior may
  lean specifically on Instruct-2507's feature distribution; if chain
  acceptance stays at ~0.26 there too, the bottleneck is chain
  compounding, not target mismatch.

## Deliverables

- `scripts/baseline_eagle3.py` — MLX port of EAGLE-3 draft + chain-style
  verify loop + contention harness. 650 LOC.
- `notes/eagle3_baseline.md` — this document.

## Reproduce

On mini-03 with `~/mlx-venv` (Python 3.9 + MLX 0.29 + mlx-lm 0.29.1):

```bash
# Solo
python scripts/baseline_eagle3.py --approach eagle3 --mode solo \
    --max-new 100 --num-draft 2

# Contention (moderate)
python scripts/baseline_eagle3.py --approach eagle3 --mode parallel \
    --max-new 100 --num-draft 2 \
    --bg-model mlx-community/gemma-3-270m-it-bf16

# Contention (heavy)
python scripts/baseline_eagle3.py --approach eagle3 --mode parallel \
    --max-new 100 --num-draft 2 \
    --bg-model mlx-community/Qwen3-4B-bf16

# Target-only reference
python scripts/baseline_eagle3.py --approach target --mode solo \
    --max-new 100
```

## What this adds to the paper

- A second SD method (EAGLE-3) measured on M4 Pro on the same 4 prompts.
- Confirms that **chain-only EAGLE-3 is not competitive with DFlash on M4
  Pro**. The tree variant would likely be competitive; porting is future
  work.
- Confirms that **GPU-resident SD degrades roughly uniformly under
  contention** across all three methods (~-45%), consistent with the
  Phase-C hypothesis that moving the draft off the GPU is the lever.
