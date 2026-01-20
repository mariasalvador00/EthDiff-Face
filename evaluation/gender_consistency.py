import os
import argparse
import csv
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp_classifier(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )

def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

 
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    X = np.load(os.path.join(args.emb_dir, "embeddings.npy"))
    ids = np.load(os.path.join(args.emb_dir, "labels.npy"))

    print(f"Loaded embeddings: {X.shape}")
    print(f"Number of identities: {len(set(ids))}") 
    model = build_mlp_classifier(X.shape[1]).to(device)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).float().to(device)
        logits = model(Xt).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    id_to_indices = defaultdict(list)
    for i, pid in enumerate(ids):
        id_to_indices[pid].append(i)

    id_rows = []

    for pid, idxs in id_to_indices.items():
        idxs = np.array(idxs, dtype=int)
        n = len(idxs)

        preds_id = preds[idxs]
        probs_id = probs[idxs]

        counts = np.bincount(preds_id, minlength=2)
        mode = int(counts.argmax())
        mode_count = int(counts.max())
        mode_frac = mode_count / n
        
        if n >= 2:
            pair_agree = (counts[0] * (counts[0] - 1) +
                          counts[1] * (counts[1] - 1)) / (n * (n - 1))
        else:
            pair_agree = 1.0

        dist = counts / n
        H = entropy_bits(dist)

        strictly_consistent = (counts > 0).sum() == 1

        mean_conf = float(np.mean(np.maximum(probs_id, 1 - probs_id)))
        median_conf = float(np.median(np.maximum(probs_id, 1 - probs_id)))

        breakdown = "; ".join(
            [f"{lbl}:{counts[lbl]}" for lbl in np.where(counts > 0)[0]]
        )

        id_rows.append({
            "identity": str(pid),
            "n_images": n,
            "mode_gender": mode,
            "mode_count": mode_count,
            "majority_fraction": round(mode_frac, 4),
            "pairwise_agreement": round(pair_agree, 4),
            "prediction_entropy_bits": round(H, 4),
            "n_unique_predictions": int((counts > 0).sum()),
            "strictly_consistent": strictly_consistent,
            "mean_max_confidence": round(mean_conf, 4),
            "median_max_confidence": round(median_conf, 4),
            "prediction_breakdown": breakdown,
        })

    num_ids = len(id_rows)

    prop_strict = sum(r["strictly_consistent"] for r in id_rows) / num_ids
    mean_mode_frac = float(np.mean([r["majority_fraction"] for r in id_rows]))
    mean_pair_agree = float(np.mean([r["pairwise_agreement"] for r in id_rows]))
    mean_entropy = float(np.mean([r["prediction_entropy_bits"] for r in id_rows]))

    print("\nGender consistency summary:")
    print(f"  identities: {num_ids}")
    print(f"  strictly consistent identities: {prop_strict:.3f}")
    print(f"  mean majority fraction:         {mean_mode_frac:.3f}")
    print(f"  mean pairwise agreement:        {mean_pair_agree:.3f}")
    print(f"  mean prediction entropy (bits): {mean_entropy:.3f}")

    out_dir = "gender_consistency"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "identity_gender_consistency.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=id_rows[0].keys())
        writer.writeheader()
        writer.writerows(id_rows)

    print(f"\nSaved per-identity metrics to: {out_csv}")

    inconsistent = [r for r in id_rows if not r["strictly_consistent"]]
    inconsistent.sort(key=lambda r: (r["majority_fraction"], -r["n_images"]))

    print("\nMost inconsistent identities:")
    for r in inconsistent[:10]:
        print(
            f"  id={r['identity']} | n={r['n_images']} | "
            f"majority_frac={r['majority_fraction']:.2f} | "
            f"entropy={r['prediction_entropy_bits']:.2f} | "
            f"breakdown={r['prediction_breakdown']}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--emb-dir",
        required=True,
        help="Directory containing embeddings.npy and labels.npy"
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to trained gender classifier (.pth)"
    )
    args = p.parse_args()
    main(args)
