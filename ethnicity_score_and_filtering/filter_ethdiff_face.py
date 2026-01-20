import torch
import numpy as np
import os

def keep_top_k_images_per_id(r_img, id_labels, k=10):
    keep_indices = []

    for i in torch.unique(id_labels):
        idx = (id_labels == i).nonzero(as_tuple=True)[0]
        scores = r_img[idx]

        sorted_idx = idx[torch.argsort(scores, descending=True)]

        keep_indices.extend(sorted_idx[:k].tolist())

    return sorted(keep_indices)


def protocol_A_keep_list(r_id, eth_labels, id_labels, Z):
    active_ids = set(r_id.keys())

    id_to_eth = {}
    for i in torch.unique(id_labels):
        idx = (id_labels == i)
        id_to_eth[int(i)] = int(eth_labels[idx][0])

    for _ in range(Z):
        ES = {}
        for y in torch.unique(eth_labels):
            ids_y = [i for i in active_ids if id_to_eth[i] == int(y)]
            ES[int(y)] = np.mean([r_id[i] for i in ids_y]) if len(ids_y) else float("inf")


        y_star = min(ES, key=ES.get)

        ids_y = [i for i in active_ids if id_to_eth[i] == y_star]
        j_star = min(ids_y, key=lambda j: r_id[j])

        active_ids.remove(j_star)

    return sorted(active_ids)

def protocol_A_image_keep_list(r_img, eth_labels, Z):
    active_idx = set(range(len(r_img)))

    for _ in range(Z):
        ES = {}
        for y in torch.unique(eth_labels):
            idx_y = [i for i in active_idx if eth_labels[i] == y]
            ES[int(y)] = np.mean([r_img[i] for i in idx_y]) if len(idx_y) else float("inf")

        y_star = min(ES, key=ES.get)

        idx_y = [i for i in active_idx if eth_labels[i] == y_star]
        i_star = min(idx_y, key=lambda i: r_img[i])

        active_idx.remove(i_star)

    return sorted(active_idx)

def main():
    
    base = "evaluation/ethnicity-score"
    Z = 10000 #number of ids/imgs to exclude

    data = torch.load(os.path.join(base, "ethnicity_scores.pt"))

    # r_id = data["r_identity"]
    # eth_labels = data["eth_labels"]
    # id_labels = data["id_labels"]

    # keep_ids = protocol_A_keep_list(r_id, eth_labels, id_labels, Z)

    # torch.save(keep_ids, os.path.join(base, f"protocolA_keep_ids_{Z}.pt"))
    # print(f"Saved {len(keep_ids)} identities.")

    #r_img = data["r_image"]
    #eth_labels = data["eth_labels"]

    #keep_imgs = protocol_A_image_keep_list(r_img, eth_labels, Z)

    #torch.save(keep_imgs, os.path.join(base, f"protocolA_keep_imgs_{Z}.pt"))
    #print(f"Saved {len(keep_imgs)} images.")

    k = 15 # number of images per identity to keep

    data = torch.load(os.path.join(base, "ethnicity_scores.pt"))

    r_img = data["r_image"]
    id_labels = data["id_labels"]

    keep_imgs = keep_top_k_images_per_id(r_img, id_labels, k)

    torch.save(
        keep_imgs,
        os.path.join(base, f"top{k}_imgs_per_id.pt")
    )

    print(f"Saved {len(keep_imgs)} images ({k} per identity).")



if __name__ == "__main__":
    main()
