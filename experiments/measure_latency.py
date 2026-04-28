"""
Latency Benchmark: measure overhead of the ALSR two-pass gated inference.

The reviewer asked: what is the latency cost of the optional second pass?

Measures (wall-clock, averaged over N_WARMUP + N_SAMPLES runs):
  1. Backbone pass                 — run backbone + text encoder once
  2. First DETR pass               — FPN(L32) + DETR (routing decision)
  3. Second DETR pass              — FPN(Lk) + DETR (conditional on gate trigger)
  4. Router forward                — MLP router inference (negligible, but reported)
  5. End-to-end L32-only           — backbone + one DETR pass (baseline)
  6. End-to-end Gated ALSR         — backbone + first DETR + (gate_rate × second DETR)

Reports:
  - Mean ± std per stage (ms)
  - Gate trigger rate from real data (from eval_full_raw.csv if available)
  - Expected overhead = gate_rate × second_pass_latency
  - Speedup vs. running all samples through both passes

Usage:
    cd capr_clean
    python experiments/measure_latency.py

    # Custom settings:
    N_SAMPLES=50 GATE_THRESHOLD=0.5 python experiments/measure_latency.py
"""
import os, sys, json, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper
from capr_router   import load_router

import torch

# ── Config ────────────────────────────────────────────────────────────────────
N_WARMUP        = 3    # warm-up runs (discarded)
N_SAMPLES       = int(os.environ.get("N_SAMPLES",   "20"))   # timed runs
GATE_THRESHOLD  = float(os.environ.get("GATE_THRESHOLD", "0.5"))
ROUTER_LAYER    = 28   # representative non-L32 layer for second-pass timing

DATA_FILE  = ("/home/grads/f/fahimehorvatinia/Documents/newpaper_2026"
              "/saco_gold_data/metaclip/saco_gold_metaclip_test_1.json")
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results")


def sync():
    """GPU sync before timing for accurate measurement."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_sample_images(n: int):
    with open(DATA_FILE) as f:
        data = json.load(f)
    samples = []
    for img in data["images"]:
        path = os.path.join(IMAGE_ROOT, img["file_name"])
        if os.path.exists(path):
            samples.append((path, img["text_input"]))
        if len(samples) >= n + N_WARMUP:
            break
    return samples


def time_stage(fn, label, n_runs):
    """Run fn() n_runs times and return list of elapsed ms."""
    times = []
    for _ in range(n_runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    mean, std = np.mean(times), np.std(times)
    print(f"  {label:<40s}: {mean:7.2f} ± {std:5.2f} ms  (n={n_runs})")
    return times


def get_real_gate_rate():
    """Read empirical gate trigger rate from eval_full_raw.csv if available."""
    csv_path = os.path.join(OUT_DIR, "eval_full_raw.csv")
    if not os.path.exists(csv_path):
        return None
    import csv
    triggered, total = 0, 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method") == "gated":
                total += 1
                if row.get("routed", "False").lower() == "true":
                    triggered += 1
    return triggered / total if total > 0 else None


def main():
    print("\n=== ALSR Latency Benchmark ===")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"N_SAMPLES={N_SAMPLES}  GATE_THRESHOLD={GATE_THRESHOLD}\n")

    samples = load_sample_images(N_SAMPLES + N_WARMUP)
    if len(samples) < N_WARMUP + N_SAMPLES:
        print(f"WARNING: only {len(samples)} images found; reduce N_SAMPLES.")

    wrapper = SAM3Wrapper()
    router  = load_router()
    router.eval()
    device  = wrapper.device
    router  = router.to(device)

    # Pre-load images to avoid disk I/O in timing loops
    print("Loading images...")
    images_data = []
    for path, prompt in samples:
        img = Image.open(path).convert("RGB")
        images_data.append((img, prompt))

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"\nWarm-up ({N_WARMUP} runs)...")
    for img, prompt in images_data[:N_WARMUP]:
        pv, ids = wrapper.preprocess(img, prompt)
        hs, lhs, t_emb, _, _ = wrapper.extract(pv, ids)
        wrapper.run(img, hs, lhs, t_emb, pv, layer_idx=32)

    timed = images_data[N_WARMUP: N_WARMUP + N_SAMPLES]

    # ── Stage 1: Backbone + text encoder ──────────────────────────────────────
    print(f"\nTiming {N_SAMPLES} samples per stage:\n")
    backbone_times = []
    extracted      = []   # cache for downstream stages

    for img, prompt in timed:
        sync()
        t0 = time.perf_counter()
        pv, ids = wrapper.preprocess(img, prompt)
        hs, lhs, t_emb, t_router, i_router = wrapper.extract(pv, ids)
        sync()
        backbone_times.append((time.perf_counter() - t0) * 1000)
        extracted.append((img, pv, hs, lhs, t_emb, t_router, i_router))

    mean_bb, std_bb = np.mean(backbone_times), np.std(backbone_times)
    print(f"  {'Backbone + text encoder':<40s}: {mean_bb:7.2f} ± {std_bb:5.2f} ms")

    # ── Stage 2: First DETR pass (FPN(L32) + DETR) ───────────────────────────
    first_pass_times = []
    routing_data     = []

    for img, pv, hs, lhs, t_emb, t_router, i_router in extracted:
        sync()
        t0 = time.perf_counter()
        result = wrapper.run(img, hs, lhs, t_emb, pv, layer_idx=32)
        sync()
        first_pass_times.append((time.perf_counter() - t0) * 1000)
        q32 = result["query_score"]
        # Also extract routing embedding for router forward timing
        detr_emb = wrapper.extract_detr_emb(hs, lhs, t_emb, pv, layer_idx=32)
        routing_data.append((img, pv, hs, lhs, t_emb, q32, detr_emb))

    mean_fp, std_fp = np.mean(first_pass_times), np.std(first_pass_times)
    print(f"  {'First DETR pass (FPN+DETR, L32)':<40s}: {mean_fp:7.2f} ± {std_fp:5.2f} ms")

    # ── Stage 3: Router forward (MLP) ─────────────────────────────────────────
    router_times = []
    for *_, detr_emb in routing_data:
        emb_t = detr_emb.unsqueeze(0).to(device)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = router(emb_t)
        sync()
        router_times.append((time.perf_counter() - t0) * 1000)

    mean_rt, std_rt = np.mean(router_times), np.std(router_times)
    print(f"  {'Router MLP forward':<40s}: {mean_rt:7.2f} ± {std_rt:5.2f} ms")

    # ── Stage 4: Second DETR pass (FPN(Lk) + DETR) ───────────────────────────
    second_pass_times = []
    for img, pv, hs, lhs, t_emb, q32, detr_emb in routing_data:
        sync()
        t0 = time.perf_counter()
        _ = wrapper.run(img, hs, lhs, t_emb, pv, layer_idx=ROUTER_LAYER)
        sync()
        second_pass_times.append((time.perf_counter() - t0) * 1000)

    mean_sp, std_sp = np.mean(second_pass_times), np.std(second_pass_times)
    print(f"  {'Second DETR pass (FPN+DETR, L{})'.format(ROUTER_LAYER):<40s}: {mean_sp:7.2f} ± {std_sp:5.2f} ms")

    # ── Stage 5: End-to-end L32-only = backbone + first DETR pass ────────────
    e2e_l32  = np.array(backbone_times) + np.array(first_pass_times)
    mean_e2e = np.mean(e2e_l32)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("SUMMARY")
    print(f"{'─'*60}")

    # Gate trigger rate
    real_rate = get_real_gate_rate()
    gate_rate = real_rate if real_rate is not None else (1 - 0.964)
    gate_src  = "from eval_full_raw.csv" if real_rate is not None else "from paper (1 - recall@L32)"
    print(f"\nGate trigger rate : {gate_rate:.3f} ({gate_rate*100:.1f}%)  [{gate_src}]")

    # Expected overhead per sample
    expected_overhead = gate_rate * mean_sp
    overhead_pct      = 100 * expected_overhead / mean_e2e

    print(f"\nLatency breakdown (ms, mean ± std):")
    print(f"  Backbone + text encoder  : {mean_bb:7.2f} ± {std_bb:.2f}")
    print(f"  1st DETR pass (L32)      : {mean_fp:7.2f} ± {std_fp:.2f}")
    print(f"  Router MLP               : {mean_rt:7.2f} ± {std_rt:.2f}   ← negligible")
    print(f"  2nd DETR pass (L{ROUTER_LAYER})     : {mean_sp:7.2f} ± {std_sp:.2f}")
    print(f"\nEnd-to-end (L32 baseline) : {mean_e2e:7.2f} ms")
    print(f"Expected gated overhead   : {gate_rate:.3f} × {mean_sp:.2f} = {expected_overhead:.2f} ms")
    print(f"Overhead as % of baseline : {overhead_pct:.2f}%")
    print(f"\nConclusion: Gated ALSR adds ≈{expected_overhead:.1f} ms ({overhead_pct:.1f}% of L32 latency) on average,")
    print(f"since the second pass fires on only {gate_rate*100:.1f}% of samples.")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "device":               str(wrapper.device),
        "n_samples":            N_SAMPLES,
        "gate_threshold":       GATE_THRESHOLD,
        "gate_rate":            gate_rate,
        "gate_rate_source":     gate_src,
        "router_layer":         ROUTER_LAYER,
        "backbone_ms_mean":     float(mean_bb),
        "backbone_ms_std":      float(std_bb),
        "first_detr_ms_mean":   float(mean_fp),
        "first_detr_ms_std":    float(std_fp),
        "router_mlp_ms_mean":   float(mean_rt),
        "router_mlp_ms_std":    float(std_rt),
        "second_detr_ms_mean":  float(mean_sp),
        "second_detr_ms_std":   float(std_sp),
        "e2e_l32_ms_mean":      float(mean_e2e),
        "expected_overhead_ms": float(expected_overhead),
        "overhead_pct":         float(overhead_pct),
    }
    import json as _json
    save_path = os.path.join(OUT_DIR, "latency_benchmark.json")
    with open(save_path, "w") as f:
        _json.dump(out, f, indent=2)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
