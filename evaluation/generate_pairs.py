import argparse
import os
import itertools
import random
import csv
import numpy as np
from collections import defaultdict

def build_index_by_race(dataset_dir):
    file_paths = np.load(os.path.join(dataset_dir, "file_paths.npy"))
    labels     = np.load(os.path.join(dataset_dir, "labels.npy"))
    ethnicities= np.load(os.path.join(dataset_dir, "ethnicities.npy"))

    data = defaultdict(lambda: defaultdict(list))

    for path, lab, eth in zip(file_paths, labels, ethnicities):
        data[str(eth)][str(lab)].append(path)

    return data



def generate_pairs(samples_dir, neg_ratio=1, seed=42):
    random.seed(seed)
    data = build_index_by_race(samples_dir)

    all_pairs = []

    for eth, ids in data.items():
        positives = []
        for ident, imgs in ids.items():
            for a, b in itertools.combinations(imgs, 2):
                positives.append((a, b, 1))

        negatives = []
        id_list = list(ids.keys())
        neg_needed = int(len(positives) * neg_ratio)

        while len(negatives) < neg_needed:
            id1, id2 = random.sample(id_list, 2)
            i1 = random.choice(ids[id1])
            i2 = random.choice(ids[id2])
            negatives.append((i1, i2, 0))

        print(f"{eth}: {len(positives)} positives, {len(negatives)} negatives")
        all_pairs.extend(positives + negatives)

    random.shuffle(all_pairs)
    return all_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--out", default="pairs.csv")
    parser.add_argument("--neg-ratio", type=float, default=1.0, help="#negatives = neg_ratio * #positives")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = generate_pairs(args.samples_dir, neg_ratio=args.neg_ratio, seed=args.seed)
    print(f"Generated {len(pairs)} pairs ({sum(1 for p in pairs if p[2]==1)} positives, {sum(1 for p in pairs if p[2]==0)} negatives)")
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        for a, b, same in pairs:
            writer.writerow([a, b, int(same)])
    print("Wrote pairs to", args.out)

if __name__ == "__main__":
    main()
