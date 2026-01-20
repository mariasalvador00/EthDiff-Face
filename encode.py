import os
from typing import Any

import hydra
import torch
import torchvision
from lightning.fabric import Fabric
from torchvision.transforms.functional import resize

from PIL import Image

from omegaconf import OmegaConf, DictConfig
import numpy as np

from utils.helpers import ensure_path_join, normalize_to_neg_one_to_one

import sys

from utils.iresnet import iresnet100, iresnet50
from utils.irse import IR_101
from utils.synface_resnet import LResNet50E_IR
from utils.moco import MoCo

sys.path.insert(0, 'IDiff-Face/')



class EncoderLite(Fabric):
    def run(self, cfg) -> Any:

        face_backbone = None

        for frm_name in cfg.encode.frm_names:
            print(f"Starting Encoding Process for FRM: {frm_name}")
            del face_backbone

            # Load face recognition model
            if frm_name == "elasticface":
                face_backbone = iresnet100(num_features=512)
                ckpt = torch.load(os.path.join("pre_trained_FR_models",
                        "elasticface_resnet_100","arc","295672backbone_arc.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "curricularface":
                face_backbone = IR_101([112, 112])
                ckpt = torch.load(os.path.join("pre_trained_FR_models","CurricularFace_Backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "idiff-face":
                face_backbone = iresnet50(num_features=512)
                ckpt = torch.load(os.path.join("utils", "54684backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "sface":
                face_backbone = iresnet50(num_features=512)
                ckpt = torch.load(os.path.join("utils", "79232backbone.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "usynthface":
                face_backbone = MoCo(base_encoder=iresnet50, dim=512, K=32768)
                ckpt = torch.load(os.path.join("utils", "checkpoint_051.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt, strict=False)
                face_backbone = face_backbone.encoder_q

            elif frm_name == "synface":
                face_backbone = LResNet50E_IR([112, 96])
                ckpt = torch.load(os.path.join("utils", "model_10k_50_idmix_9197.pth"), map_location="cpu")["state_dict"]
                face_backbone.load_state_dict(ckpt)

            elif frm_name == "arcface":


                face_backbone = iresnet100(num_features=512)
                ckpt = torch.load(os.path.join("pre_trained_FR_models",
                        "arcface_r_100","arcface_r100.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt,strict=True)


            elif frm_name == "cosface":
                face_backbone = iresnet100(num_features=512)
                ckpt = torch.load(os.path.join("pre_trained_FR_models",
                        "cosface_r100","cosface_r100.pth"), map_location="cpu")
                face_backbone.load_state_dict(ckpt,strict=True)

            face_backbone = self.setup(face_backbone)
            face_backbone = face_backbone.cpu()
            face_backbone.eval()

            for model_name in cfg.encode.model_names:
                for contexts_name in cfg.encode.contexts_names:

                    if cfg.encode.aligned:
                        samples_dir ="/nas-ctm01/datasets/public/BIOMETRICS/Face_Recognition/rfw/test/data" #"../IDiff-Face/samples/our_approach/samples_aligned"#"../IDiff-Face/face_recognition_training/BF_subset_train_ids/train"#"../IDiff-Face/samples/our_approach/samples_aligned"
                        embeddings_dir = "data/rfw/embeddings_aligned"
                    else:
                        samples_dir = ensure_path_join("samples", model_name, contexts_name)
                        embeddings_dir = ensure_path_join("samples", "embeddings", model_name, contexts_name, frm_name)

                    if not os.path.isdir(samples_dir):
                        print(f"Samples directory {samples_dir} does not exist! Skipping.")
                        continue

                    if os.path.isfile(os.path.join(embeddings_dir, f"embeddings.npy")):
                        print(f"Embeddings already exist in {embeddings_dir}. Skipping {contexts_name}.")
                        continue
                    embeddings, labels, ethnicities = self.encode_images(
                        face_backbone, samples_dir, image_size=112 if cfg.encode.aligned else cfg.encode.image_size
                    )
                    os.makedirs(embeddings_dir, exist_ok=True)
                    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)
                    np.save(os.path.join(embeddings_dir, "labels.npy"), labels)
                    np.save(os.path.join(embeddings_dir, "ethnicities.npy"), ethnicities)


    def encode_images(self, face_backbone, samples_dir, image_size=128):
        embeddings = []
        id_labels = []
        ethnicities = []

        for ethnicity in sorted(os.listdir(samples_dir)):
            ethnicity_path = os.path.join(samples_dir, ethnicity)#, ethnicity)
            if not os.path.isdir(ethnicity_path):
                continue

            for identity in sorted(os.listdir(ethnicity_path)):
                identity_path = os.path.join(ethnicity_path, identity)
                if not os.path.isdir(identity_path):
                    continue

                print("Encoding:", ethnicity, identity)

                id_images = []
                for img_file in sorted(os.listdir(identity_path)):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    img_path = os.path.join(identity_path, img_file)
                    with open(img_path, "rb") as f:
                        img = Image.open(f).convert("RGB")
                    img = torchvision.transforms.functional.to_tensor(img)
                    img = resize(img, [image_size, image_size])
                    id_images.append(img)

                if not id_images:
                    continue

                id_images = torch.stack(id_images)
                id_images = normalize_to_neg_one_to_one(id_images)#.cuda()
                id_images = torchvision.transforms.functional.resize(id_images, [112, 112])

                with torch.no_grad():
                    id_embeds = face_backbone(id_images)
                    id_embeds = torch.nn.functional.normalize(id_embeds)

                for embed in id_embeds.detach().cpu().numpy():
                    embeddings.append(embed)
                    id_labels.append(identity)
                    ethnicities.append(ethnicity)

        return np.array(embeddings), np.array(id_labels), np.array(ethnicities)



@hydra.main(config_path='configs', config_name='encode_config', version_base=None)
def encode(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = EncoderLite(devices=1, accelerator="cpu")

    sampler.run(cfg)


if __name__ == "__main__":
    encode()
