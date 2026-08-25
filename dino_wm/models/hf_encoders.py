import torch
import torch.nn as nn
from transformers import CLIPVisionModel, SiglipVisionModel, ViTMAEModel


class CLIPEncoder(nn.Module):
    def __init__(self, name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.name = name
        # use_safetensors=True: avoids transformers' torch.load path, which
        # this old torch (2.3.0 < 2.6) is blocked from using for security reasons
        self.base_model = CLIPVisionModel.from_pretrained(name, use_safetensors=True)
        self.emb_dim = self.base_model.config.hidden_size
        self.patch_size = self.base_model.config.patch_size
        self.latent_ndim = 2

    def forward(self, x):
        out = self.base_model(pixel_values=x).last_hidden_state
        return out[:, 1:]  # drop the CLS token, keep patch tokens


class SigLIPEncoder(nn.Module):
    def __init__(self, name="google/siglip-base-patch16-224"):
        super().__init__()
        self.name = name
        self.base_model = SiglipVisionModel.from_pretrained(name, use_safetensors=True)
        self.emb_dim = self.base_model.config.hidden_size
        self.patch_size = self.base_model.config.patch_size
        self.latent_ndim = 2

    def forward(self, x):
        # SigLIP has no CLS token: every token in last_hidden_state is a patch token
        return self.base_model(pixel_values=x).last_hidden_state


class MAEEncoder(nn.Module):
    def __init__(self, name="facebook/vit-mae-base"):
        super().__init__()
        self.name = name
        # mask_ratio=0.0 disables MAE's random patch masking so the encoder
        # returns the full, deterministic patch grid instead of a random subset
        self.base_model = ViTMAEModel.from_pretrained(name, mask_ratio=0.0, use_safetensors=True)
        self.emb_dim = self.base_model.config.hidden_size
        self.patch_size = self.base_model.config.patch_size
        self.latent_ndim = 2

    def forward(self, x):
        # Even with mask_ratio=0.0 (no patches dropped), HF's random_masking still
        # shuffles patch order using fresh random noise on every call unless we pin
        # it ourselves. Without this, the world model would see patches in a
        # different spatial order every forward pass.
        b = x.shape[0]
        num_patches = (self.base_model.config.image_size // self.base_model.config.patch_size) ** 2
        noise = torch.arange(num_patches, device=x.device, dtype=torch.float32).unsqueeze(0).expand(b, -1)
        out = self.base_model(pixel_values=x, noise=noise).last_hidden_state
        return out[:, 1:]  # drop the CLS token, keep patch tokens
