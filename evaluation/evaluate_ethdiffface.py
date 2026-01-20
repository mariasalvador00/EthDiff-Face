import os
import csv
import math
from typing import Any

import matplotlib.pyplot as plt
import hydra
import torch
from lightning.fabric import Fabric
from omegaconf import OmegaConf, DictConfig
import numpy as np

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report

from utils.helpers import ensure_path_join

import sys
sys.path.insert(0, 'IDiff-Face/')


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


class EvaluatorLite(Fabric):
    def run(self, cfg) -> Any:

        self.seed_everything(cfg.evaluation.seed)
        eer_stats = {}

        default_n_bins = getattr(cfg.evaluation, "n_bins", 100)

        for model_name in cfg.evaluation.model_names:
            for frm_name in cfg.evaluation.frm_names:

                synthetic_context_short = (
                    cfg.evaluation.synthetic_contexts_name
                    .replace("random_", "")
                    .replace("_5000", "")
                )

                if cfg.evaluation.aligned:
                    eval_dir = ensure_path_join(
                        "evaluation",
                        "eth_diff_face",
                        f"aligned_{synthetic_context_short}",
                        model_name,
                        frm_name,
                    )
                    synthetic_preencoded_data_dir = "data/ethdiff_face/cosface/embeddings_aligned"
                    samples_dir = "../IDiff-Face/samples/our_approach/samples_aligned"
                else:
                    eval_dir = ensure_path_join(
                        "evaluation","ethdiff_face25cpd","balanced_dataset","elasticface-arc"
                    )
                    synthetic_preencoded_data_dir = os.path.join(
                        "samples",
                        "embeddings",
                        model_name,
                        cfg.evaluation.synthetic_contexts_name,
                        frm_name,
                    )
                    samples_dir = "../IDiff-Face/samples/our_approach/samples_aligned"

                os.makedirs(eval_dir, exist_ok=True)

                synthetic_embeddings = torch.from_numpy(
                    np.load(os.path.join(synthetic_preencoded_data_dir, "embeddings.npy"))
                )
                synthetic_ethnicities = np.load(
                    os.path.join(synthetic_preencoded_data_dir, "ethnicities.npy")
                )

                # -------- FILE UNIVERSE --------
                if cfg.evaluation.use_protocolA:
                    files = np.load(os.path.join(synthetic_preencoded_data_dir, "file_paths.npy")).tolist()
                else:
                    files = sorted(build_file_index(samples_dir))

                assert len(files) == synthetic_embeddings.shape[0], \
                    f"FILES {len(files)} != EMB {synthetic_embeddings.shape[0]}"

                # -------- PROTOCOL A FILTER --------
                if cfg.evaluation.use_protocolA:

                    keep_ids = set(torch.load(cfg.evaluation.protocolA_keep_ids))

                    score_data = torch.load(cfg.evaluation.ethnicity_scores)
                    score_paths = score_data["file_paths"]
                    score_ids   = score_data["id_labels"]
                    path_to_identity = {p: int(i) for p, i in zip(score_paths, score_ids)}

                    mask = np.array([path_to_identity[f] in keep_ids for f in files])


                    synthetic_embeddings  = synthetic_embeddings[mask]
                    synthetic_ethnicities = synthetic_ethnicities[mask]
                    files = [f for f, m in zip(files, mask) if m]

                    assert len(files) == synthetic_embeddings.shape[0] == len(synthetic_ethnicities)
                            

                else:
                    assert len(files) == len(synthetic_embeddings), \
                        "Mismatch between files ordering and embeddings.npy"

                path_to_idx = {p: i for i, p in enumerate(files)}

                pairs_path = os.path.join("pairs.csv")
                pairs = []
                with open(pairs_path, "r") as f:
                    reader = csv.reader(f)
                    for a, b, same in reader:
                        if a in path_to_idx and b in path_to_idx:
                            pairs.append((path_to_idx[a], path_to_idx[b], int(same)))


                # ---- GLOBAL SCORES ----
                genuine_scores = []
                imposter_scores = []

                for i, j, same in pairs:
                    cos_sim = float(np.dot(synthetic_embeddings[i], synthetic_embeddings[j]))
                    if same == 1:
                        genuine_scores.append(cos_sim)
                    else:
                        imposter_scores.append(cos_sim)

                print(
                    f"[INFO] {model_name}/{frm_name} – "
                    f"genuine: {len(genuine_scores)}, "
                    f"imposter: {len(imposter_scores)}"
                )

                all_scores = genuine_scores + imposter_scores
                min_score, max_score = float(min(all_scores)), float(max(all_scores))
                if math.isclose(min_score, max_score):
                    min_score -= 1e-6
                    max_score += 1e-6
                bins = np.linspace(min_score, max_score, int(default_n_bins))

                plt.clf()
                plt.hist(genuine_scores, bins=bins, label="genuine", alpha=0.5)
                plt.hist(imposter_scores, bins=bins, label="imposter", alpha=0.5)
                plt.xlabel("Cosine similarity")
                plt.ylabel("Count")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    ensure_path_join(eval_dir, "synthetic_vs_synthetic_distributions.png"),
                    dpi=512,
                )

                global_stats = get_eer_stats(genuine_scores, imposter_scores)
                eer_stats[f"{model_name}_{frm_name}_overall"] = global_stats
                overall_eer = global_stats.eer

                # ---- BY ETHNICITY ----
                if cfg.evaluation.by_ethnicity:
                    for ethnicity in np.unique(synthetic_ethnicities):
                        eth_idxs = set(np.where(synthetic_ethnicities == ethnicity)[0])

                        eth_genuine = []
                        eth_imposter = []

                        for i, j, same in pairs:
                            if i in eth_idxs and j in eth_idxs:
                                cos_sim = float(
                                    np.dot(synthetic_embeddings[i], synthetic_embeddings[j])
                                )
                                if same == 1:
                                    eth_genuine.append(cos_sim)
                                else:
                                    eth_imposter.append(cos_sim)

                        if len(eth_genuine) == 0 or len(eth_imposter) == 0:
                            continue

                        eth_stats = get_eer_stats(eth_genuine, eth_imposter)
                        delta_eer = eth_stats.eer - overall_eer

                        print(
                            f"[DELTA-EER] {model_name}/{frm_name}/{ethnicity} – "
                            f"EER={eth_stats.eer:.4f}, ΔEER={delta_eer:+.4f}"
                        )

                        eer_stats[f"{model_name}_{frm_name}_{ethnicity}"] = {
                            "stats": eth_stats,
                            "delta_eer_overall": delta_eer,
                        }

        report_path = os.path.join(eval_dir, "pyeer_report.html")
        print(f"Saving PyEER report to: {report_path}")
        generate_eer_report(
            [v["stats"] if isinstance(v, dict) else v for v in eer_stats.values()],
            list(eer_stats.keys()),
            report_path,
        )


@hydra.main(config_path="configs", config_name="evaluate_config", version_base=None)
def evaluate(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    evaluator = EvaluatorLite(devices="auto", accelerator="auto")
    evaluator.run(cfg)


if __name__ == "__main__":
    evaluate()
