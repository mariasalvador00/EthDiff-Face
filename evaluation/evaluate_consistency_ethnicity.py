# evaluate_consistency_ethnicity.py
#
# Usage:
#   python evaluate_consistency_ethnicity.py \
#       --emb-dir samples/embeddings/<model_name>/<contexts_name>/<frm_name> \
#       --model eth_classifier/best_model.pth \
#       [--label-map packed_embeddings.pt|label_map.json] \
#       [--ece-bins 15] [--hist-bins 100]
#
# The emb-dir must contain: embeddings.npy, ethnicities.npy, labels.npy

import os
import argparse
import json
import csv
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def build_mlp_classifier(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )


def normalize_label(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def load_label_map(label_map_path):
    if label_map_path is None:
        return None
    ext = os.path.splitext(label_map_path)[1].lower()
    if ext == ".pt":
        blob = torch.load(label_map_path, map_location="cpu")
        if "label_map" not in blob:
            raise ValueError("No 'label_map' found in PT file.")
        return blob["label_map"]
    elif ext == ".json":
        with open(label_map_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("label_map must be a .pt from packing or a .json")


def _min_bins(b, n_min=100):
    return max(int(b), n_min)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    X = np.load(os.path.join(args.emb_dir, "embeddings.npy"))
    y_true_str = np.load(os.path.join(args.emb_dir, "ethnicities.npy"))
    ids = np.load(os.path.join(args.emb_dir, "labels.npy"))
    train_label_map_raw = load_label_map(args.label_map) 

    if train_label_map_raw is not None:
        
        norm_to_idx = {}
        for k, v in train_label_map_raw.items():
            nk = normalize_label(k)
            if nk in norm_to_idx and norm_to_idx[nk] != v:
                print(f"WARNING: conflicting normalized keys for '{nk}' in training label_map.")
            norm_to_idx[nk] = int(v)
        y_true_norm = [normalize_label(s) for s in y_true_str]
        unknown = sorted(set(y_true_norm) - set(norm_to_idx.keys()))
        if unknown:
            known = ", ".join(sorted(norm_to_idx.keys()))
            raise ValueError(
                "Some ground-truth labels from 'ethnicities.npy' don't exist in the training label_map "
                f"after normalization.\nUnknown (normalized) labels: {unknown}\n"
                f"Known (normalized) training labels: {known}\n"
                "If these are just naming differences (e.g., 'African' vs 'african'), this error should not happen. "
                "If they are true class mismatches (e.g., 'south_asian' vs 'indian'), retrain or align names."
            )
        y_true = np.array([norm_to_idx[n] for n in y_true_norm], dtype=int)
        
        index_to_label = {}
        for k, v in train_label_map_raw.items():
            index_to_label[int(v)] = str(k)
        ordered_names = [index_to_label[i] for i in range(len(index_to_label))]

    else:
        
        print("WARNING: no label_map provided; inferring class set/order from y_true. "
              "Ensure this matches the training classes and order.")
        classes_norm = sorted({normalize_label(s) for s in y_true_str.tolist()})
        norm_to_idx = {c: i for i, c in enumerate(classes_norm)}
        y_true = np.array([norm_to_idx[normalize_label(s)] for s in y_true_str], dtype=int)
        ordered_names = classes_norm  # normalized display names

    input_dim = X.shape[1]
    num_classes = len(ordered_names)
    
    model = build_mlp_classifier(input_dim, num_classes).to(device)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    
    with torch.no_grad():
        Xt = torch.from_numpy(X).float().to(device)
        logits = model(Xt)  # (N, C)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        probs = F.softmax(logits, dim=1).cpu().numpy()


    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    print(f"Image-level accuracy: {acc:.4f} | macro-F1: {macro_f1:.4f} | micro-F1: {micro_f1:.4f}\n")

    print("Classification report:\n")
    print(classification_report(y_true, y_pred, target_names=ordered_names, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("Confusion matrix (rows=true, cols=pred):\n", cm, "\n")

    by_id_true = {}
    by_id_preds = defaultdict(list)
    for yi, pid, pi in zip(y_true, ids, y_pred):
        by_id_true.setdefault(pid, yi)
        by_id_preds[pid].append(pi)

    id_correct = 0
    for pid, preds in by_id_preds.items():
        maj = Counter(preds).most_common(1)[0][0]
        if maj == by_id_true[pid]:
            id_correct += 1
    id_acc = id_correct / len(by_id_preds)
    print(f"Identity-level accuracy (majority vote): {id_acc:.4f}")

    def entropy_bits(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum()) if p.size else 0.0

    id_to_indices = {}
    for i, pid in enumerate(ids):
        id_to_indices.setdefault(pid, []).append(i)

    id_rows = []
    warn_multi_gt = 0

    for pid, idxs in id_to_indices.items():
        idxs = np.array(idxs, dtype=int)
        n = len(idxs)
        
        gt_set = set(y_true[idxs].tolist())
        if len(gt_set) != 1:
            warn_multi_gt += 1
        gt = list(gt_set)[0] if len(gt_set) else None

        # predictions + distribution
        preds = y_pred[idxs]
        counts = np.bincount(preds, minlength=num_classes)
        mode = int(counts.argmax())
        mode_count = int(counts.max())
        mode_frac = mode_count / n

        # pairwise agreement P(same class for two random images), defined for n>=2
        if n >= 2:
            pair_agree = (np.sum(counts * (counts - 1))) / (n * (n - 1))
        else:
            pair_agree = 1.0

        # entropy of prediction distribution (0 = perfectly consistent)
        dist = counts / n
        H = entropy_bits(dist)

        # strict consistency flags
        n_unique_pred = int((counts > 0).sum())
        strictly_consistent = (n_unique_pred == 1)

        # correctness flavors
        majority_correct = (gt is not None and mode == gt)
        strictly_correct = strictly_consistent and (gt is not None) and (mode == gt)
        strictly_wrong = strictly_consistent and (gt is not None) and (mode != gt)

        # confidence summaries (max softmax per image)
        max_conf_per_img = probs[idxs].max(axis=1)
        mean_max_conf = float(max_conf_per_img.mean())
        median_max_conf = float(np.median(max_conf_per_img))

        # breakdown string like "asian:7; caucasian:3" using display names
        breakdown_parts = []
        for c in np.where(counts > 0)[0]:
            breakdown_parts.append(f"{ordered_names[c]}:{counts[c]}")
        breakdown = "; ".join(breakdown_parts)

        id_rows.append({
            "identity": str(pid),
            "n_images": n,
            "gt_label": ordered_names[gt] if gt is not None else "<mixed/unknown>",
            "mode_pred": ordered_names[mode],
            "mode_count": mode_count,
            "mode_frac": round(mode_frac, 4),
            "pairwise_agreement": round(pair_agree, 4),
            "pred_entropy_bits": round(H, 4),
            "n_unique_pred": n_unique_pred,
            "strictly_consistent": strictly_consistent,
            "majority_correct": majority_correct,
            "strictly_correct": strictly_correct,
            "strictly_wrong": strictly_wrong,
            "mean_max_conf": round(mean_max_conf, 4),
            "median_max_conf": round(median_max_conf, 4),
            "pred_breakdown": breakdown,
        })

    num_ids = len(id_rows)
    prop_strict = sum(r["strictly_consistent"] for r in id_rows) / num_ids if num_ids else 0.0
    prop_strict_correct = sum(r["strictly_correct"] for r in id_rows) / num_ids if num_ids else 0.0
    prop_strict_wrong = sum(r["strictly_wrong"] for r in id_rows) / num_ids if num_ids else 0.0
    mean_mode_frac = float(np.mean([r["mode_frac"] for r in id_rows])) if num_ids else 0.0
    mean_pair_agree = float(np.mean([r["pairwise_agreement"] for r in id_rows])) if num_ids else 0.0

    print(f"\nPer-identity consistency:")
    print(f"  identities: {num_ids}")
    print(f"  strictly consistent (all images same prediction): {prop_strict:.3f}")
    print(f"    └─ strictly consistent AND correct:           {prop_strict_correct:.3f}")
    print(f"    └─ strictly consistent BUT wrong:             {prop_strict_wrong:.3f}")
    print(f"  mean mode fraction:                             {mean_mode_frac:.3f}")
    print(f"  mean pairwise agreement:                        {mean_pair_agree:.3f}")
    if warn_multi_gt > 0:
        print(f"  WARNING: {warn_multi_gt} identities have multiple ground-truth labels.")

    # write CSV
    os.makedirs("eth_classifier", exist_ok=True)
    out_csv = os.path.join("eth_classifier", "identity_consistency.csv")
    if num_ids > 0:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(id_rows[0].keys()))
            writer.writeheader()
            writer.writerows(id_rows)
        print(f"\nWrote per-identity metrics to {out_csv}")

    # show a few inconsistent identities to inspect
    inconsistent = [r for r in id_rows if r["n_unique_pred"] > 1]
    inconsistent.sort(key=lambda r: (r["mode_frac"], -r["n_images"]))  # lowest consistency first
    print("\nExamples of inconsistent identities:")
    for r in inconsistent[:10]:
        print(f"  id={r['identity']} | n={r['n_images']} | gt={r['gt_label']} | "
              f"mode={r['mode_pred']} ({r['mode_frac']:.2f}) | breakdown: {r['pred_breakdown']}")

    # ---- PROBABILITY PLOTS & CALIBRATION METRICS ----
    out_dir = os.path.join("eth_classifier", "real_data", "figs")
    os.makedirs(out_dir, exist_ok=True)

    def _ece(conf, correct, n_bins=15):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(conf, bins) - 1
        ece = 0.0
        for b in range(n_bins):
            mask = idx == b
            if not np.any(mask):
                continue
            bin_conf = conf[mask].mean()
            bin_acc = correct[mask].mean()
            ece += (mask.mean()) * abs(bin_acc - bin_conf)
        return float(ece)

    def _nll(logits_tensor, y_true_np):
        lt = logits_tensor.detach().cpu()
        yt = torch.from_numpy(y_true_np).long()
        return float(F.cross_entropy(lt, yt, reduction="mean").item())

    def plot_max_conf_hist(probs_np, path, bins=100):
        conf = probs_np.max(axis=1)
        plt.figure(figsize=(6, 4))
        plt.hist(conf, bins=_min_bins(bins), alpha=0.8)
        plt.xlabel("Max softmax confidence")
        plt.ylabel("Count")
        plt.title("Max-confidence distribution")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def plot_true_prob_hist(probs_np, y_true_np, y_pred_np, path, bins=100):
        p_true = probs_np[np.arange(len(y_true_np)), y_true_np]
        correct_mask = (y_pred_np == y_true_np)
        plt.figure(figsize=(6, 4))
        plt.hist(p_true[correct_mask], bins=_min_bins(bins), alpha=0.7, label="correct")
        plt.hist(p_true[~correct_mask], bins=_min_bins(bins), alpha=0.7, label="incorrect")
        plt.xlabel("P(true class)")
        plt.ylabel("Count")
        plt.title("Probability assigned to the true class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def plot_reliability(conf_np, correct_np, path, n_bins=15):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(conf_np, bins) - 1
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if not np.any(mask):
                continue
            xs.append(conf_np[mask].mean())
            ys.append(correct_np[mask].mean())
        xs = np.array(xs); ys = np.array(ys)
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # y=x reference
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Reliability diagram")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def plot_risk_coverage(conf_np, correct_np, path):
        order = np.argsort(-conf_np)
        corr_sorted = correct_np[order].astype(np.float64)
        cum_correct = np.cumsum(corr_sorted)
        idx = np.arange(1, len(corr_sorted) + 1)
        acc_prefix = cum_correct / idx
        risk = 1.0 - acc_prefix
        coverage = idx / len(corr_sorted)
        plt.figure(figsize=(6, 4))
        plt.plot(coverage, risk)
        plt.xlabel("Coverage (fraction kept by confidence)")
        plt.ylabel("Risk (1 - accuracy)")
        plt.title("Risk–coverage curve")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    # compute arrays we need
    conf_max = probs.max(axis=1)
    correct01 = (y_pred == y_true).astype(np.float64)

    # calibration metrics
    ece = _ece(conf_max, correct01, n_bins=args.ece_bins)
    nll = _nll(logits, y_true)
    print(f"\nCalibration: ECE({args.ece_bins} bins) = {ece:.4f} | NLL = {nll:.4f}")

    # plots with at least 100 bins in histograms
    plot_max_conf_hist(probs, os.path.join(out_dir, "max_conf_hist.png"), bins=args.hist_bins)
    plot_true_prob_hist(probs, y_true, y_pred, os.path.join(out_dir, "true_prob_hist.png"), bins=args.hist_bins)
    plot_reliability(conf_max, correct01, os.path.join(out_dir, "reliability_diagram.png"), n_bins=args.ece_bins)
    plot_risk_coverage(conf_max, correct01, os.path.join(out_dir, "risk_coverage.png"))

    # optional: per-class true-probability histograms (also min 100 bins)
    for c_idx, c_name in enumerate(ordered_names):
        mask = (y_true == c_idx)
        if not np.any(mask):
            continue
        p_true_c = probs[mask, c_idx]
        plt.figure(figsize=(6, 4))
        plt.hist(p_true_c, bins=_min_bins(args.hist_bins), alpha=0.85)
        plt.xlabel(f"P(true={c_name})")
        plt.ylabel("Count")
        plt.title(f"Distribution of P(true={c_name})")
        plt.tight_layout()
        safe_name = str(c_name).replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"true_prob_hist_{safe_name}.png"), dpi=150)
        plt.close()

    print(f"Saved probability plots in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True, help="Directory with embeddings.npy, ethnicities.npy, labels.npy")
    p.add_argument("--model", required=True, help="Path to eth_classifier/best_model.pth")
    p.add_argument("--label-map", default=None, help="Path to packed_embeddings.pt or label_map.json")
    p.add_argument("--ece-bins", type=int, default=15,
                   help="Number of bins for ECE and the reliability diagram.")
    p.add_argument("--hist-bins", type=int, default=100,
                   help="Number of bins for probability histograms (minimum 100).")
    args = p.parse_args()
    main(args)
