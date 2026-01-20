import os
import csv
import numpy as np
from pyeer.eer_info import get_eer_stats


def build_file_index(samples_dir):
    files = []
    for ethnicity in sorted(os.listdir(samples_dir)):
        eth1 = os.path.join(samples_dir, ethnicity)
        if not os.path.isdir(eth1):
            continue

        eth2 = os.path.join(eth1, ethnicity)
        if not os.path.isdir(eth2):
            continue

        for identity in sorted(os.listdir(eth2)):
            id_path = os.path.join(eth2, identity)
            if not os.path.isdir(id_path):
                continue

            for img_file in sorted(os.listdir(id_path)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    files.append(
                        os.path.join(ethnicity, ethnicity, identity, img_file)
                    )
    return files


def main():
    EMBEDDINGS_PATH = "data/ethdiff_face/elasticface-arc/embeddings.npy"
    ETHNICITIES_PATH = "data/ethdiff_face/elasticface-arc/ethnicities.npy"
    PAIRS_PATH = "pairs.csv"

    SAMPLES_DIR = "../IDiff-Face/samples/our_approach/samples_aligned"

    QUANTILE = 0.2
    BY_ETHNICITY = True

    embeddings = np.load(EMBEDDINGS_PATH)
    ethnicities = np.load(ETHNICITIES_PATH)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    files = build_file_index(SAMPLES_DIR)

    assert len(files) == len(embeddings), \
        "Mismatch between file list and embeddings order"

    path_to_idx = {p: i for i, p in enumerate(files)}

    pairs = []
    with open(PAIRS_PATH, "r") as f:
        reader = csv.reader(f)
        for a, b, same in reader:
            i = path_to_idx[a]
            j = path_to_idx[b]
            pairs.append((i, j, int(same)))

    genuine_pairs = []    # (i, j, sim)
    impostor_pairs = []   # (i, j, sim)

    for i, j, same in pairs:
        sim = float(np.dot(embeddings[i], embeddings[j]))
        if same == 1:
            genuine_pairs.append((i, j, sim))
        else:
            impostor_pairs.append((i, j, sim))

    genuine_scores = np.array([s for _, _, s in genuine_pairs])
    impostor_scores = np.array([s for _, _, s in impostor_pairs])

    print(f"[INFO] Genuine pairs:  {len(genuine_scores)}")
    print(f"[INFO] Impostor pairs: {len(impostor_scores)}")


    overall_stats = get_eer_stats(
        genuine_scores.tolist(),
        impostor_scores.tolist()
    )
    print(f"[OVERALL] EER = {overall_stats.eer:.4f}")


    q = QUANTILE

    g_low, g_high = np.quantile(genuine_scores, [q, 1 - q])
    i_low, i_high = np.quantile(impostor_scores, [q, 1 - q])

    easy_genuine_pairs = [(i, j) for i, j, s in genuine_pairs if s >= g_high]
    hard_genuine_pairs = [(i, j) for i, j, s in genuine_pairs if s <= g_low]

    easy_impostor_pairs = [(i, j) for i, j, s in impostor_pairs if s <= i_low]
    hard_impostor_pairs = [(i, j) for i, j, s in impostor_pairs if s >= i_high]

    easy_stats = get_eer_stats(
        [s for _, _, s in genuine_pairs if s >= g_high],
        [s for _, _, s in impostor_pairs if s <= i_low]
    )

    hard_stats = get_eer_stats(
        [s for _, _, s in genuine_pairs if s <= g_low],
        [s for _, _, s in impostor_pairs if s >= i_high]
    )

    print(f"[EASY] EER = {easy_stats.eer:.4f}")
    print(f"[HARD] EER = {hard_stats.eer:.4f}")


    if BY_ETHNICITY:
        for eth in np.unique(ethnicities):
            eth_idxs = set(np.where(ethnicities == eth)[0])

            eth_genuine = []
            eth_impostor = []

            for i, j, same in pairs:
                if i in eth_idxs and j in eth_idxs:
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if same == 1:
                        eth_genuine.append(sim)
                    else:
                        eth_impostor.append(sim)

            if len(eth_genuine) == 0 or len(eth_impostor) == 0:
                continue

            eth_genuine = np.array(eth_genuine)
            eth_impostor = np.array(eth_impostor)

            g_low, g_high = np.quantile(eth_genuine, [q, 1 - q])
            i_low, i_high = np.quantile(eth_impostor, [q, 1 - q])

            eth_easy = get_eer_stats(
                eth_genuine[eth_genuine >= g_high].tolist(),
                eth_impostor[eth_impostor <= i_low].tolist()
            )

            eth_hard = get_eer_stats(
                eth_genuine[eth_genuine <= g_low].tolist(),
                eth_impostor[eth_impostor >= i_high].tolist()
            )

            print(
                f"[{eth.upper()}] "
                f"EER easy = {eth_easy.eer:.4f} | "
                f"EER hard = {eth_hard.eer:.4f}"
            )

    np.save("easy_genuine_pairs.npy", easy_genuine_pairs)
    np.save("hard_genuine_pairs.npy", hard_genuine_pairs)
    np.save("easy_impostor_pairs.npy", easy_impostor_pairs)
    np.save("hard_impostor_pairs.npy", hard_impostor_pairs)


if __name__ == "__main__":
    main()
