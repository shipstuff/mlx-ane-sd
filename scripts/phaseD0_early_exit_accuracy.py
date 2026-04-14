"""Phase D.0: measure early-exit top-κ accuracy on Gemma-3-12B bf16.

The Mirror-SD paper (Section 3) uses mid-layer logits p^(ℓ_e) as a proxy
for the final layer distribution p^(N). The draft is seeded with the
Top-κ tokens from that proxy and runs speculatively in parallel with
the target's remaining layers.

The entire scheme hinges on "is the mid-layer top-κ a good predictor of
the final top-1?" This script measures exactly that on Gemma-3-12B across
diverse prompts, at multiple candidate ℓ_e depths.

Output: per-depth top-1 match rate and top-κ containment rate. This tells
us whether κ=1 (a simple cross-cycle scheme) is viable or whether we must
build tree speculation (Phase E) before Mirror-SD.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load


PROMPTS = [
    "The capital of France is Paris, which is known for its rich history, art, and culture. Some famous landmarks include",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# To compute the 10th Fibonacci number",
    "In quantum mechanics, the Heisenberg uncertainty principle states that the position and momentum of a particle cannot",
    "The recipe called for two cups of flour, one teaspoon of baking soda, and a pinch of salt. She mixed them together",
    "The lawsuit alleged that the company had engaged in deceptive practices by misrepresenting the efficacy of its product",
    "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who",
    "The economic impact of climate change on coastal communities has been well documented. Rising sea levels threaten",
    "Solving for x: 2x + 5 = 17. Subtracting 5 from both sides gives 2x = 12. Dividing both sides by 2 yields",
]


def forward_with_early_exits(model, prompt_ids, exit_layers):
    """Run Gemma-3-12B forward, capturing hidden states at each requested
    early-exit layer. Returns dict: {layer_idx: logits (T, V)} for each
    requested layer, plus final logits under key 'final'.
    """
    lm = model.language_model
    inner = lm.model  # Gemma3Model
    args = inner.args
    lm_head = lm.lm_head

    from mlx_lm.models.gemma3_text import create_attention_mask

    inputs = mx.array([prompt_ids])
    h = inner.embed_tokens(inputs)
    h = h * mx.array(args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

    # No cache — we want parallel prefill forward for accuracy measurement
    cache = [None] * len(inner.layers)
    global_mask = create_attention_mask(h, cache[args.sliding_window_pattern - 1])
    sliding_mask = None
    if args.sliding_window_pattern > 1:
        sliding_mask = create_attention_mask(h, cache[0], window_size=args.sliding_window)

    out_logits = {}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        is_global = i % args.sliding_window_pattern == args.sliding_window_pattern - 1
        mask = global_mask if is_global else sliding_mask
        h = layer(h, mask, c)

        if (i + 1) in exit_layers:
            h_normed = inner.norm(h)
            logits_i = lm_head(h_normed).astype(mx.float32)
            mx.eval(logits_i)
            out_logits[i + 1] = np.asarray(logits_i[0])

    h_final = inner.norm(h)
    logits_final = lm_head(h_final).astype(mx.float32)
    mx.eval(logits_final)
    out_logits["final"] = np.asarray(logits_final[0])
    return out_logits


def top_k_indices(logits_row, k):
    # Returns indices of top-k values, unsorted order doesn't matter for containment
    return np.argpartition(-logits_row, k - 1)[:k]


def measure_prompt(model, tokenizer, prompt, exit_layers, k_values):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    logits_by_layer = forward_with_early_exits(model, prompt_ids, exit_layers)

    final_logits = logits_by_layer["final"]  # (T, V)
    T = final_logits.shape[0]
    final_top1 = final_logits.argmax(axis=-1)  # (T,)

    per_depth = {}
    for depth in exit_layers:
        proxy_logits = logits_by_layer[depth]  # (T, V)
        proxy_top1 = proxy_logits.argmax(axis=-1)
        # top-1 match rate
        top1_match = (proxy_top1 == final_top1).mean()
        # top-k containment rate: is final_top1 in proxy_topK?
        containment = {}
        for k in k_values:
            topk = np.argpartition(-proxy_logits, k - 1, axis=-1)[:, :k]
            hit = np.array([final_top1[t] in topk[t] for t in range(T)])
            containment[k] = hit.mean()
        per_depth[depth] = dict(top1=float(top1_match), containment={k: float(v) for k, v in containment.items()})
    return per_depth, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-3-12b-it-bf16")
    ap.add_argument("--exit-layers", type=int, nargs="+",
                    default=[12, 18, 24, 30, 36, 42])
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"loaded in {time.perf_counter()-t0:.1f}s")

    num_layers = len(model.language_model.model.layers)
    print(f"Total layers: {num_layers}")
    print(f"Exit layers to measure: {args.exit_layers}")
    print(f"k values: {args.k_values}")
    print()

    # Aggregate across prompts
    agg = {d: {"top1_sum": 0.0, "containment_sum": {k: 0.0 for k in args.k_values},
               "positions": 0} for d in args.exit_layers}

    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] prompt: {prompt[:60]!r}...")
        per_depth, T = measure_prompt(model, tokenizer, prompt,
                                       args.exit_layers, args.k_values)
        for d, stats in per_depth.items():
            agg[d]["top1_sum"] += stats["top1"] * T
            for k in args.k_values:
                agg[d]["containment_sum"][k] += stats["containment"][k] * T
            agg[d]["positions"] += T
        print(f"  positions: {T}")

    print("\n=== Early-exit accuracy across {} positions ===\n".format(agg[args.exit_layers[0]]["positions"]))
    hdr = f"{'depth':>6} | {'top-1':>7} |" + "".join(f" top-{k:<2}  |" for k in args.k_values)
    print(hdr)
    print("-" * len(hdr))
    for d in args.exit_layers:
        N = agg[d]["positions"]
        top1 = agg[d]["top1_sum"] / N
        row = f"{d:>6} | {top1:>7.1%} |"
        for k in args.k_values:
            c = agg[d]["containment_sum"][k] / N
            row += f" {c:>6.1%} |"
        print(row)

    print("\nInterpretation:")
    print("  top-1 = P(layer_d argmax == final argmax)")
    print("  top-k = P(final argmax in layer_d top-k)")
    print("  For Mirror-SD viability without tree spec (kappa=1):")
    print("    need top-1 >= 70% at some reasonable depth")
    print("  With tree spec (kappa=8), need top-8 containment >= 90%")


if __name__ == "__main__":
    main()
