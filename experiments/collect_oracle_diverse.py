"""
Diverse Oracle Collection — Step 1 for scalable router training (Reviewer Exp. 3).

Extends collect_oracle_layers.py to collect oracle data from MULTIPLE SA-Co
domains (all sharing the same metaclip-images/ image root).  Training the router
on a diverse mix of concept types (general, attribute-qualified, crowded scenes,
Wikipedia categories) produces a more generalizable routing signal.

Domains collected (all use metaclip-images/):
    metaclip       — general web concepts        (~1500 samples)
    attributes     — attribute-qualified objects  (~1500 samples)
    crowded        — cluttered/crowded scenes     (~1500 samples)
    wiki-food      — Wikipedia food & drink       (~1500 samples)
    wiki-sports    — Wikipedia sports equipment   (~1500 samples)
    wiki-common1k  — Wikipedia common 1K concepts (~1500 samples)

SA-1B is excluded: it uses a different image source (sa_*.jpg files not in
metaclip-images/).

Saves merged arrays to results/router_training_data_diverse/:
    text_embs.npy      [N_total, 1024]
    img_embs.npy       [N_total, 1024]
    cgf1_matrix.npy    [N_total, 16]
    meta.json          with domain labels and 70/20/10 splits

Usage:
    cd capr_clean
    python experiments/collect_oracle_diverse.py

    # Collect only specific domains:
    DOMAINS=metaclip,crowded python experiments/collect_oracle_diverse.py

    # Limit per-domain sample count:
    N_PER_DOMAIN=500 python experiments/collect_oracle_diverse.py
"""
import os, sys, json, random
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam3_wrapper import SAM3Wrapper
from metrics import compute_cgf1, merge_gt_masks

# ── Config ────────────────────────────────────────────────────────────────────
SACO_ROOT  = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/saco_gold_data"
IMAGE_ROOT = "/home/grads/f/fahimehorvatinia/Documents/newpaper_2026/metaclip-images"
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "router_training_data_diverse")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_LAYERS  = list(range(17, 33))   # 16 candidate layers
N_PER_DOMAIN  = int(os.environ.get("N_PER_DOMAIN", "1500"))
SEED          = 42

# Domain → (subfolder, json_prefix)
DOMAIN_CONFIGS = {
    "metaclip":       ("metaclip",          "saco_gold_metaclip_test_1.json"),
    "attributes":     ("attributes",         "saco_gold_attributes_test_1.json"),
    "crowded":        ("crowded",            "saco_gold_crowded_test_1.json"),
    "wiki-food":      ("wiki-food&drink",    "saco_gold_wiki-food&drink_test_1.json"),
    "wiki-sports":    ("wiki-sports_equipment", "saco_gold_wiki-sports_equipment_test_1.json"),
    "wiki-common1k":  ("wiki-common1k",      "saco_gold_wiki-common1k_test_1.json"),
}

_env_domains = os.environ.get("DOMAINS", "")
if _env_domains:
    DOMAINS = [d.strip() for d in _env_domains.split(",") if d.strip()]
else:
    DOMAINS = list(DOMAIN_CONFIGS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_domain_positives(domain: str, n: int):
    """Load up to n positive samples from a domain's test_1 JSON."""
    subdir, fname = DOMAIN_CONFIGS[domain]
    json_path = os.path.join(SACO_ROOT, subdir, fname)
    with open(json_path) as f:
        data = json.load(f)

    anno_map = defaultdict(list)
    for a in data["annotations"]:
        anno_map[a["image_id"]].append(a)

    pool = []
    for img in data["images"]:
        annos = anno_map.get(img["id"], [])
        if not annos:
            continue
        path = os.path.join(IMAGE_ROOT, img["file_name"])
        if not os.path.exists(path):
            continue
        pool.append(dict(
            image_id=img["id"],
            image_path=path,
            prompt=img["text_input"],
            height=img["height"],
            width=img["width"],
            annotations=annos,
            domain=domain,
        ))

    random.seed(SEED)
    random.shuffle(pool)
    selected = pool[:n]
    print(f"  [{domain}] positives available: {len(pool):5d}  →  collecting: {len(selected)}")
    return selected


def resize_mask(pred, gt):
    if pred.shape == gt.shape:
        return pred
    return np.array(Image.fromarray(pred.astype(np.uint8) * 255).resize(
        (gt.shape[1], gt.shape[0]), Image.NEAREST)).astype(bool)


def collect_domain(samples, wrapper):
    """Run oracle layer sweep for a list of samples. Returns arrays and meta."""
    text_embs   = []
    img_embs    = []
    cgf1_matrix = []
    meta        = []
    skipped     = 0

    domain_label = samples[0].get('domain', '?') if samples else '?'
    for sample in tqdm(samples, desc=f"  collecting {domain_label}", leave=False):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            skipped += 1; continue

        try:
            gt = merge_gt_masks(sample["annotations"], sample["height"], sample["width"])
            if gt.shape != (sample["height"], sample["width"]):
                gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize(
                    (sample["width"], sample["height"]), Image.NEAREST)).astype(bool)
        except Exception:
            skipped += 1; continue

        if gt.sum() == 0:
            skipped += 1; continue

        try:
            pv, ids = wrapper.preprocess(img, sample["prompt"])
            hs, backbone_lhs, text_emb_detr, text_emb_router, img_emb_router = \
                wrapper.extract(pv, ids)
        except Exception:
            skipped += 1; continue

        row = []
        for l in TRAIN_LAYERS:
            try:
                out  = wrapper.run(img, hs, backbone_lhs, text_emb_detr, pv, layer_idx=l)
                pred = out["union_mask"]
                if pred.shape != gt.shape:
                    pred = resize_mask(pred, gt)
                row.append(compute_cgf1(pred, gt))
            except Exception:
                row.append(0.0)

        if max(row) < 0.01:
            skipped += 1; continue

        text_embs.append(text_emb_router.cpu().float().numpy())
        img_embs.append(img_emb_router.cpu().float().numpy())
        cgf1_matrix.append(row)
        meta.append(dict(image_id=sample["image_id"],
                         prompt=sample["prompt"],
                         domain=sample["domain"]))

    print(f"    → collected: {len(text_embs)}  skipped: {skipped}")
    return (np.array(text_embs,   dtype=np.float32),
            np.array(img_embs,    dtype=np.float32),
            np.array(cgf1_matrix, dtype=np.float32),
            meta)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Diverse Oracle Collection ===")
    print(f"Domains: {DOMAINS}")
    print(f"N per domain: {N_PER_DOMAIN}")
    print(f"Layers: {TRAIN_LAYERS}\n")

    # Load all samples upfront to report availability before heavy compute
    all_samples = []
    print("Scanning available data:")
    for domain in DOMAINS:
        samples = load_domain_positives(domain, N_PER_DOMAIN)
        all_samples.extend(samples)
    print(f"\nTotal samples to collect: {len(all_samples)}\n")

    wrapper = SAM3Wrapper()

    # Collect per-domain (preserves domain label in meta)
    all_text_embs   = []
    all_img_embs    = []
    all_cgf1        = []
    all_meta        = []

    for domain in DOMAINS:
        domain_samples = [s for s in all_samples if s["domain"] == domain]
        print(f"\nCollecting [{domain}] ({len(domain_samples)} samples)...")
        te, ie, cf, mt = collect_domain(domain_samples, wrapper)
        if len(te) == 0:
            print(f"  WARNING: no samples collected for {domain}, skipping.")
            continue
        all_text_embs.append(te)
        all_img_embs.append(ie)
        all_cgf1.append(cf)
        all_meta.extend(mt)

    # Merge
    text_embs   = np.concatenate(all_text_embs,  axis=0)
    img_embs    = np.concatenate(all_img_embs,   axis=0)
    cgf1_matrix = np.concatenate(all_cgf1,       axis=0)
    N           = len(all_meta)

    print(f"\n{'─'*50}")
    print(f"Total collected: {N} samples across {len(DOMAINS)} domains")

    # Domain breakdown
    from collections import Counter
    domain_counts = Counter(m["domain"] for m in all_meta)
    for d, c in domain_counts.items():
        print(f"  {d:<20s}: {c}")

    # Shuffle combined dataset with fixed seed
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    text_embs   = text_embs[perm]
    img_embs    = img_embs[perm]
    cgf1_matrix = cgf1_matrix[perm]
    all_meta    = [all_meta[i] for i in perm]

    # 70/20/10 split
    n_train = int(N * 0.70)
    n_val   = int(N * 0.20)
    splits  = dict(
        train=list(range(0,              n_train)),
        val  =list(range(n_train,        n_train + n_val)),
        test =list(range(n_train + n_val, N)),
    )
    print(f"\n70/20/10 split: {n_train} train | {n_val} val | {N - n_train - n_val} test")

    # Save
    np.save(os.path.join(OUT_DIR, "text_embs.npy"),   text_embs)
    np.save(os.path.join(OUT_DIR, "img_embs.npy"),    img_embs)
    np.save(os.path.join(OUT_DIR, "cgf1_matrix.npy"), cgf1_matrix)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"layer_list": TRAIN_LAYERS, "samples": all_meta,
                   "splits": splits, "domains": DOMAINS}, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  text_embs.npy    : {text_embs.shape}")
    print(f"  img_embs.npy     : {img_embs.shape}")
    print(f"  cgf1_matrix.npy  : {cgf1_matrix.shape}")

    print(f"\nPer-layer mean cgF1 (across all domains):")
    for i, l in enumerate(TRAIN_LAYERS):
        vals = cgf1_matrix[:, i]
        print(f"  L{l}: mean={vals.mean():.4f}  max={vals.max():.4f}")

    best_layer_counts = np.argmax(cgf1_matrix, axis=1)
    print(f"\nOracle best-layer distribution:")
    for i, l in enumerate(TRAIN_LAYERS):
        cnt = (best_layer_counts == i).sum()
        pct = 100 * cnt / N
        bar = "█" * int(pct / 2)
        print(f"  L{l}: {cnt:4d} ({pct:4.1f}%)  {bar}")

    print(f"\nTo train the router on diverse data:")
    print(f"  DATA_DIR=$(pwd)/results/router_training_data_diverse \\")
    print(f"  EMB_MODE=concat python experiments/train_router.py")


if __name__ == "__main__":
    main()
