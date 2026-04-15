# Month research plan: DFlash-on-Apple-Silicon → publishable artifact

**Start:** 2026-04-14
**Duration:** 4 weeks
**Resources:** mini-02 (primary dev), mini-03 (agent helpers), RTX 5000 (training), no cloud compute

## Research narrative

**"Block-diffusion speculative decoding on Apple Silicon: a port that uses 10%
of the ANE, freeing the GPU for N-way concurrent serving."**

Current state: F.1 ANE port runs 100% on ANE at 79-85% of the GPU baseline.
The undervalued property: ANE is barely saturated (~3.5 of 38 TOPS). One ANE
draft can serve multiple concurrent SD streams — a deployment regime where
F.1 decisively beats F.0. That's the research artifact.

## Phase structure

```
Week 1   mini-02: Multi-stream serving (primary thrust)
         mini-03 agent A: EAGLE-3 Qwen3-4B baseline
         mini-03 agent B: Full LUT × cache-size grid
         mini-03 agent C: Multi-function ANE variant
         RTX 5000 agent: Port z-lab training code, Gemma-3-1B sanity run

Week 2   mini-02: Swift native runner + super-block fusion on DFlash
         mini-03 agent: consolidate Week 1 agent results
         RTX 5000 agent: Gemma-3-4B full training

Week 3   Convergence: integrate Gemma-3-4B draft into F.1 ANE pipeline
         Benchmark the new pair (Gemma-3-4B + trained draft) in solo,
         contention, and multi-stream regimes
         mini-03 agent: long-running benchmark sweep

Week 4   Paper writing + repo cleanup + public release
         Submission to arxiv or similar
```

## Expected deliverables

**End of Week 1:**
- Multi-stream serving infrastructure working on mini-02
- Throughput curves for F.0 and F.1 at N=1..8 concurrent streams
- EAGLE-3 baseline numbers on Qwen3-4B (for context in paper)
- Full LUT × cache-size matrix filled in
- Multi-function variant compiled and benchmarked
- Gemma-3-1B DFlash sanity-trained draft

**End of Week 2:**
- Swift native runner: F.1 solo ≥ F.0 solo (target to beat)
- Super-block fusion applied to DFlash (if it works, another +15-30%)
- Gemma-3-4B DFlash draft trained, uploaded to HF Hub

**End of Week 3:**
- F.1 ANE port validated on Gemma-3-4B target (second target family)
- Multi-stream serving results across both targets
- Paper outline drafted

**End of Week 4:**
- Paper submitted
- Public repo + weights + benchmark tools released

## Non-goals

- **Gemma-3-12B training**: won't fit on RTX 5000, no cloud. If
  sponsorship lands later, the Gemma-3-4B training code will transfer.
- **Tree speculation**: validation showed -55% on our workload. Proper
  tree attention is 1+ week of engineering for modest expected gain.
  Parked as future work.
- **Power/energy**: explicitly deprioritized by the user.
- **Beating F.0 on solo tok/s only**: if Swift runner + fusion gets us
  there, great; if not, multi-stream is still the winning angle.

## Parallelism map

### mini-02 (current machine, primary dev)
- Weeks 1-2: Multi-stream serving + Swift native runner + fusion
- Week 3-4: Integration + paper writing

### mini-03 (helper machine, agents run here)
- Week 1: 3 parallel agents (EAGLE-3, LUT grid, multi-function) — see below
- Week 2: Consolidate their outputs into the paper's benchmark tables
- Week 3: Long-running benchmark sweep (multi-stream × multiple targets)

### RTX 5000
- See `notes/agent_briefs/rtx5000_gemma3_draft_training.md`

### Convergence checkpoints

After Week 1: user reviews all agent outputs + mini-02 progress. Adjust
Week 2 scope if needed.

After Week 2: Gemma-3-4B draft should be ready. User decides whether
Week 3 integrates it or if training needs more time.

End of Week 3: paper outline review before Week 4 writing sprint.

## Agent briefs (detailed)

Three separate briefs below; copy/paste to start each agent.

---

### mini-03 agent A: EAGLE-3 baseline on Qwen3-4B

(See `notes/agent_briefs/mini03_eagle3_baseline.md`)

### mini-03 agent B: LUT × cache-size grid

(See `notes/agent_briefs/mini03_lut_cache_grid.md`)

### mini-03 agent C: multi-function ANE variant

(See `notes/agent_briefs/mini03_multifunction.md`)

### RTX 5000 agent: Gemma-3 draft training

(See `notes/agent_briefs/rtx5000_gemma3_draft_training.md`)

## Success criteria for the paper

**Narrative coherence** — does the paper tell a clear story?
- Yes: "ANE port + multi-stream serving is the right deployment for block-diffusion SD on Apple Silicon, here's a 2.3× advantage when N>3"
- No: "We tried a bunch of things, some worked, here are scattered benchmarks"

**Novelty** — what's the contribution?
- Primary: first ANE port of block-diffusion SD with acceptance parity
- Secondary: multi-stream serving architecture that exploits ANE's unused capacity
- Tertiary: characterization of where DDTree-style approaches do/don't transfer

**Reproducibility** — can others replicate?
- Public code, weights, benchmark harness, reference numbers for all tables

If Week 4 ends and ANY of these are not in good shape, extend by 1 week
rather than ship prematurely.
