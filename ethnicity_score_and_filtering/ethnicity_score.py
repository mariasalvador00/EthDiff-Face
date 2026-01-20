import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn

def build_mlp_classifier(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

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
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(
                        os.path.join(ethnicity, ethnicity, identity, img_file)
                    )

    return files

def compute_ethnicity_scores(embeddings, prototypes, tau=0.1):
    embeddings = F.normalize(embeddings, dim=1)

    sim = embeddings @ prototypes.T
    scores = torch.softmax(sim / tau, dim=1)

    return scores

def compute_image_representativeness(scores, eth_labels):
    idx = torch.arange(len(eth_labels), device=eth_labels.device)
    return scores[idx, eth_labels]

def compute_id_representativeness(r, id_labels):
    id_values = torch.unique(id_labels)
    r_id = {}

    for idv in id_values:
        idx = (id_labels == idv)
        r_id[int(idv)] = r[idx].mean().item()

    return r_id

def filter_strong_identities(r_id, eth_labels, id_labels, q=0.8):
    mask = torch.zeros(len(id_labels), dtype=torch.bool)

    for k in torch.unique(eth_labels):
        ids_k = torch.unique(id_labels[eth_labels == k])

        r_vals = torch.tensor([r_id[int(i.item())] for i in ids_k])
        thresh = torch.quantile(r_vals, q)

        strong_ids = ids_k[r_vals >= thresh]

        for sid in strong_ids:
            mask |= (id_labels == sid)

    return mask


def main():
    base = "data/rfw/embeddings_aligned"
    out_dir = "evaluation/ethnicity-score/rfw"


    samples_dir = "/nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/rfw/test/data"
    file_paths = np.load(os.path.join(base, "file_paths.npy")) #build_file_index(samples_dir)

    os.makedirs(out_dir,exist_ok=True) 
    embeddings = torch.from_numpy(np.load(os.path.join(base, "embeddings.npy"))).float()
    
    assert len(file_paths) == embeddings.shape[0], \
        f"Mismatch: {len(file_paths)} files vs {embeddings.shape[0]} embeddings"

    eth_raw = np.load(os.path.join(base, "ethnicities.npy"))
    unique_eth = sorted(set(eth_raw.tolist()))
    eth2id = {e: i for i, e in enumerate(unique_eth)}
    eth_labels = torch.tensor([eth2id[e] for e in eth_raw], dtype=torch.long)

    id_raw = np.load(os.path.join(base, "labels.npy"))
    unique_ids = sorted(set(id_raw.tolist()))
    id2int = {v: i for i, v in enumerate(unique_ids)}
    id_labels = torch.tensor([id2int[i] for i in id_raw], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp_classifier(embeddings.shape[1], num_classes=4).to(device)
    model.load_state_dict(torch.load("eth_classifier/best_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(embeddings.to(device))
        scores = torch.softmax(logits, dim=1).cpu()

    r_img = compute_image_representativeness(scores, eth_labels)
    r_id  = compute_id_representativeness(r_img, id_labels)

    mask = filter_strong_identities(r_id, eth_labels, id_labels, q=0.8)

    torch.save({
        "scores": scores.cpu(),
        "r_image": r_img.cpu(),
        "r_identity": r_id,
        "mask": mask.cpu(),
        "eth_labels": eth_labels.cpu(),
        "id_labels": id_labels.cpu(),
        "file_paths": file_paths,
        "ethnicity-mapping": eth2id
    }, os.path.join(out_dir,"ethnicity_scores.pt"))


    for  eth_name , eth_id in eth2id.items():
        vals = r_img[eth_labels == eth_id].numpy()
        print(f"{eth_name:10s}  mean={vals.mean():.3f}  std={vals.std():.3f}  n={len(vals)}")


    fig, ax = plt.subplots(figsize=(8.5, 5))
    all_vals = []
    for k, name in eth2id.items():
        all_vals.append(r_img[eth_labels == name].numpy())
    all_vals = np.concatenate(all_vals)

    bins = np.linspace(all_vals.min(), all_vals.max(), 50)

    colors = {
        "african": "#BCE4FC",
        "asian": "#ADF4AF",
        "caucasian": "#FFF7AB",
        "indian": "#FFB1F9",
    }

    for k, name in eth2id.items():
        vals = r_img[eth_labels == name].numpy()

        ax.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            label=k.capitalize(),
            color=colors.get(name),
        )

    ax.set_title("Representativeness score distribution by ethnicity")
    ax.set_xlabel(r"$r_i$ score")
    ax.set_ylabel("Number of images")

    ax.legend(
        title="Ethnicity",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "r_dist_by_ethnicity.png"), dpi=300)
    plt.close(fig)



if __name__=="__main__":
    main()