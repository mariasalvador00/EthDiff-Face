import os
import torch
import numpy as np

SRC = "data/ethdiff_face/elasticface-arc/embeddings_aligned"
DST = "data/ethdiff_face/elasticface-arc/embeddings_aligned_balanced_filterimgs/keeptop15imgsforeachid"
KEEP_IDS_PATH = "evaluation/ethnicity-score/protocolA_keep_ids_50.pt"
SCORES_PATH = "evaluation/ethnicity-score/ethnicity_scores.pt"
KEEP_IMGS_PATH = "evaluation/ethnicity-score/top15_imgs_per_id.pt"


def main():
    keep_imgs = set(torch.load(KEEP_IMGS_PATH))

    embeddings  = np.load(os.path.join(SRC, "embeddings.npy"))
    labels      = np.load(os.path.join(SRC, "labels.npy"))
    ethnicities = np.load(os.path.join(SRC, "ethnicities.npy"))
    file_paths  = np.load(os.path.join(SRC, "file_paths.npy"))

    mask = np.array([i in keep_imgs for i in range(len(file_paths))], dtype=bool)

    emb_f   = embeddings[mask]
    lab_f   = labels[mask]
    eth_f   = ethnicities[mask]
    files_f = file_paths[mask]

    os.makedirs(DST, exist_ok=True)
    np.save(os.path.join(DST, "embeddings.npy"), emb_f)
    np.save(os.path.join(DST, "labels.npy"), lab_f)
    np.save(os.path.join(DST, "ethnicities.npy"), eth_f)
    np.save(os.path.join(DST, "file_paths.npy"), files_f)

    print(f"Kept {len(files_f)} images out of {len(file_paths)}.")
    print(f"Kept {len(np.unique(lab_f))} identities out of {len(np.unique(labels))}.") 
    # keep_ids = set(torch.load(KEEP_IDS_PATH))
    # score_data = torch.load(SCORES_PATH)

    # score_paths = score_data["file_paths"]
    # score_ids   = score_data["id_labels"]

    # path_to_identity = {p: int(i) for p, i in zip(score_paths, score_ids)}

    # embeddings = np.load(os.path.join(SRC, "embeddings.npy"))
    # labels     = np.load(os.path.join(SRC, "labels.npy"))
    # ethnicities= np.load(os.path.join(SRC, "ethnicities.npy"))

    # files = score_paths
    # files = np.array(files)
    

    # mask = np.array(
    #     [(f in path_to_identity) and (path_to_identity[f] in keep_ids) for f in files],
    #     dtype=bool
    # )

    # assert len(mask) == embeddings.shape[0]

    # emb_f = embeddings[mask]
    # lab_f = labels[mask]
    # eth_f = ethnicities[mask]
    # files_f = files[mask]

    # os.makedirs(DST, exist_ok=True)
    # np.save(os.path.join(DST, "embeddings.npy"), emb_f)
    # np.save(os.path.join(DST, "labels.npy"), lab_f)
    # np.save(os.path.join(DST, "ethnicities.npy"), eth_f)
    # np.save(os.path.join(DST, "file_paths.npy"), files_f)

    # print(f"Kept {len(np.unique(lab_f))} identities out of {len(np.unique(labels))}.")

if __name__ == "__main__":
    main()
