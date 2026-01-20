import argparse
import os
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn

def build_file_index(samples_dir, embeddings_dir):
    file_paths_path = os.path.join(embeddings_dir, "file_paths.npy")

    if os.path.exists(file_paths_path):
        file_paths = np.load(file_paths_path)
        return [p.replace("\\", "/") for p in file_paths]
    else:
        files = []
        for ethnicity in sorted(os.listdir(samples_dir)):
            p1 = os.path.join(samples_dir, ethnicity)
            if not os.path.isdir(p1):
                continue
            p2 = os.path.join(p1, ethnicity)
            if not os.path.isdir(p2):
                continue
            for identity in sorted(os.listdir(p2)):
                id_path = os.path.join(p2, identity)
                if not os.path.isdir(id_path):
                    continue
                for img_file in sorted(os.listdir(id_path)):
                    if img_file.lower().endswith(('.png','.jpg','.jpeg')):
                        files.append(os.path.join(ethnicity, ethnicity, identity, img_file).replace("\\","/"))
        return files




def load_pairs_csv(pairs_csv):
    pairs = []
    with open(pairs_csv, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 3:
                continue
            pairs.append((parts[0], parts[1], int(parts[2])==1))
    return pairs

def map_pairs_to_indices(files_list, pairs):
    # files_list is list of relative paths in the same order as embeddings.npy
    
    file_to_idx = {f.replace("\\", "/"): i for i, f in enumerate(files_list)}

    emb_idx_pairs = []
    unresolved = []
    for i, (p1, p2, same) in enumerate(pairs):
        p1 = p1.replace("\\", "/")
        p2 = p2.replace("\\", "/")
        if p1 in file_to_idx and p2 in file_to_idx:
            emb_idx_pairs.append((file_to_idx[p1], file_to_idx[p2], same))
        else:
            unresolved.append((i, p1, p2))

    if unresolved:
        print(f"Warning: {len(unresolved)} pairs not matched to embeddings. Sample:")
        for u in unresolved[:5]:
            print(u)
    return emb_idx_pairs


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)
    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def save_genuines_impostors(distances, issame, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    genuine_file = save_path + "_genuines.txt"
    impostor_file = save_path + "_impostors.txt"
    genuines = [d for d,s in zip(distances, issame) if s]
    impostors = [d for d,s in zip(distances, issame) if not s]
    np.savetxt(genuine_file, genuines)
    np.savetxt(impostor_file, impostors)

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same) if n_same>0 else 0.0
    far = float(false_accept) / float(n_diff) if n_diff>0 else 0.0
    return val, far

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0, gen_im_path=None):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        embeddings1 = sklearn.preprocessing.normalize(embeddings1)
        embeddings2 = sklearn.preprocessing.normalize(embeddings2)
        dist = 1 - np.sum(embeddings1 * embeddings2, axis=1)

    if gen_im_path is not None:
        save_genuines_impostors(dist, actual_issame, gen_im_path)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            dist = 1 - np.sum(embed1 * embed2, axis=1)

        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    dist = 1 - np.sum(embeddings1 * embeddings2, axis=1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])

        unique_far_train, unique_indices = np.unique(far_train, return_index=True)
        unique_thresholds = thresholds[unique_indices]

        if len(unique_far_train) > 1 and np.max(far_train) >= far_target:
            f = interpolate.interp1d(unique_far_train, unique_thresholds, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0 if len(unique_far_train) == 0 else unique_thresholds[0]

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", default="results/verification.json")
    parser.add_argument("--pca-dims", type=int, default=0)
    parser.add_argument("--nfolds", type=int, default=10)
    args = parser.parse_args()

    emb_path = os.path.join(args.embeddings_dir, "embeddings.npy")
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(emb_path + " not found")
    
    print("SAMPLES_DIR =", args.samples_dir)
    

    embeddings = np.load(emb_path)
    print("Loaded embeddings:", embeddings.shape)
    embeddings = sklearn.preprocessing.normalize(embeddings)

    files_list = build_file_index(args.samples_dir, args.embeddings_dir)
    print("Indexed files:", len(files_list))


    pairs = load_pairs_csv(args.pairs)
    mapped = map_pairs_to_indices(files_list, pairs)
    if len(mapped) == 0:
        raise RuntimeError("No pairs mapped to embeddings. Check samples_dir and pairs.csv paths.")

    emb1 = []
    emb2 = []
    issame = []
    for idx1, idx2, same in mapped:
        emb1.append(embeddings[idx1])
        emb2.append(embeddings[idx2])
        issame.append(same)
    emb1 = np.stack(emb1, axis=0)
    emb2 = np.stack(emb2, axis=0)
    issame = np.array(issame, dtype=bool)

    idx2eth = {i: files_list[i].replace("\\","/").split("/")[0] for i in range(len(files_list))}

    pair_eth = []
    for idx1, idx2, _ in mapped:
        pair_eth.append(idx2eth[idx1])
    pair_eth = np.array(pair_eth)


    thresholds = np.arange(0, 4, 0.01)
    thresholds_val = np.arange(0, 4, 0.001)

    summary = {}
    per_race = {}

    for race in sorted(set(pair_eth)):
        m = pair_eth == race
        if np.sum(m) < args.nfolds:
            print(f"Skipping {race}, not enough pairs")
            continue

        tpr, fpr, acc = calculate_roc(
            thresholds,
            emb1[m], emb2[m], issame[m],
            nrof_folds=args.nfolds,
            pca=args.pca_dims
        )

        val, val_std, far = calculate_val(
            thresholds_val,
            emb1[m], emb2[m], issame[m],
            1e-3, nrof_folds=args.nfolds
        )

        acc_mean = float(np.mean(acc))
        acc_std  = float(np.std(acc))

        per_race[race] = {
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "n_pairs": int(np.sum(m))
        }

        os.makedirs(os.path.dirname(args.out), exist_ok=True)

        np.savez(args.out.replace(".json", f"_{race}_roc.npz"), tpr=tpr, fpr=fpr)

    summary["per_race"] = per_race
    summary["pca_dims"] = int(args.pca_dims)
    summary["nfolds"] = int(args.nfolds)

    tpr_all, fpr_all, acc_all = calculate_roc(
        thresholds,
        emb1, emb2, issame,
        nrof_folds=args.nfolds,
        pca=args.pca_dims
    )

    acc_mean_all = float(np.mean(acc_all))
    acc_std_all  = float(np.std(acc_all))

    summary["overall"] = {
        "acc_mean": acc_mean_all,
        "acc_std": acc_std_all,
        "n_pairs": int(len(issame))
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved verification results to {args.out}")


if __name__ == "__main__":
    main()