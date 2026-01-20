
import os
from typing import Any
import math

import hydra
import torch

from lightning.fabric import Fabric

import omegaconf
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from torchvision.utils import save_image

from models.autoencoder.vqgan import VQEncoderInterface, VQDecoderInterface
from utils.helpers import ensure_path_join, denormalize_to_zero_to_one

import sys
sys.path.insert(0, 'IDiff-Face/')

class DiffusionSamplerLite(Fabric):

    def run(self, cfg) -> Any:
        train_cfg_path = "outputs/.hydra/config.yaml"
        train_cfg = omegaconf.OmegaConf.load(train_cfg_path)
        self.seed_everything(cfg.sampling.seed * (1 + self.global_rank))
        diffusion_model = instantiate(train_cfg.diffusion)
        diffusion_model = self.setup(diffusion_model)

        if cfg.checkpoint.use_non_ema:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', 'model.ckpt')
        elif cfg.checkpoint.global_step is not None:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'checkpoints', f'ema_averaged_model_{cfg.checkpoint.global_step}.ckpt')
        else:
            checkpoint_path = os.path.join(cfg.checkpoint.path, 'ema_averaged_model.ckpt')

        diffusion_model.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        size = (train_cfg.constants.input_channels, train_cfg.constants.image_size, train_cfg.constants.image_size)

        if train_cfg.latent_diffusion:
            latent_encoder = VQEncoderInterface(
                first_stage_config_path=os.path.join(".", "models", "autoencoder", "first_stage_config.yaml"),
                encoder_state_dict_path=os.path.join(".", "models", "autoencoder", "first_stage_encoder_state_dict.pt")
            )
            size = latent_encoder(torch.ones([1, *size])).shape[-3:]
            del latent_encoder
            latent_decoder = VQDecoderInterface(
                first_stage_config_path=os.path.join(".", "models", "autoencoder", "first_stage_config.yaml"),
                decoder_state_dict_path=os.path.join(".", "models", "autoencoder", "first_stage_decoder_state_dict.pt")
            )
            latent_decoder = self.setup(latent_decoder)
            latent_decoder.eval()
        else:
            latent_decoder = None

        assert cfg.sampling.contexts_file is not None
        contexts = torch.load(cfg.sampling.contexts_file, weights_only=False)
        assert len(contexts) >= cfg.sampling.n_contexts

        if isinstance(contexts, dict):
            input_contexts_name = cfg.sampling.contexts_file.split("/")[-1].split(".")[0]
            model_name = cfg.checkpoint.path.split("/")[-1]
            context_ids = list(contexts.keys())[:cfg.sampling.n_contexts]
        else:
            exit(1)

        if cfg.checkpoint.use_non_ema:
            model_name += "_non_ema"
        elif cfg.checkpoint.global_step is not None:
            model_name += f"_{cfg.checkpoint.global_step}"

        samples_dir = ensure_path_join("samples", model_name, input_contexts_name)

        length_before_filter = len(context_ids)
        context_ids = list(filter(lambda i: not os.path.isfile(os.path.join(samples_dir, f"{i}.png")), context_ids))
        print(f"Skipped {length_before_filter - len(context_ids)} context ids, because for them files already seem to exist!")
        context_ids = self.split_across_devices(context_ids)

        if self.global_rank == 0:
            with open(ensure_path_join(f"{samples_dir}.yaml"), "w+") as f:
                OmegaConf.save(config=cfg, f=f.name)

        try:
            model_device = next(diffusion_model.parameters()).device
        except StopIteration:
            model_device = next(diffusion_model.module.parameters()).device

        for id_name in context_ids:
            prefix = id_name
            ctx_src = contexts[id_name]
            context = ctx_src if isinstance(ctx_src, torch.Tensor) else torch.from_numpy(ctx_src)
            context = context.to(device=model_device, dtype=torch.float32)

            if context.dim() == 1:
                context = context.unsqueeze(0)
            elif context.dim() == 3 and context.size(1) == 1 and context.size(0) == 1:
                context = context.squeeze(1)
            elif context.dim() != 2:
                raise RuntimeError(f"Expected context to be 1D or 2D (or [1,1,D] singleton), got shape {tuple(context.shape)}")

            print(f"[{prefix}] context shape before sample:", tuple(context.shape))

            self.perform_sampling(
                diffusion_model=diffusion_model,
                n_samples=cfg.sampling.n_samples_per_context,
                size=size,
                batch_size=cfg.sampling.batch_size,
                samples_dir=samples_dir,
                prefix=prefix,
                context=context,
                latent_decoder=latent_decoder
            )

    @staticmethod
    def perform_sampling(
        diffusion_model, n_samples, size, batch_size, samples_dir,
        prefix: str = None, context: torch.Tensor = None,
        latent_decoder: torch.nn.Module = None
    ):
        samples_dir = ensure_path_join(samples_dir, prefix)
        os.makedirs(samples_dir, exist_ok=True)
        with torch.no_grad():
            for i in range(n_samples):
                ctx = context[:1]
                batch_samples = diffusion_model.sample(1, size, context=ctx)
                if latent_decoder:
                    batch_samples = latent_decoder(batch_samples).cpu()
                batch_samples = denormalize_to_zero_to_one(batch_samples)
                save_image(batch_samples, ensure_path_join(samples_dir, f"{i}.png"))

    def split_across_devices(self, L):
        if type(L) is int:
            L = list(range(L))
        chunk_size = math.ceil(len(L) / self.world_size)
        L_per_device = [L[idx: idx + chunk_size] for idx in range(0, len(L), chunk_size)]
        while len(L_per_device) < self.world_size:
            L_per_device.append([])
        return L_per_device[self.global_rank]


@hydra.main(config_path='configs', config_name='sample_config', version_base=None)
def sample(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sampler = DiffusionSamplerLite(devices="auto", accelerator="auto")
    sampler.run(cfg)

if __name__ == "__main__":
    sample()
