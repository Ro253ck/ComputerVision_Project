import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens": # mi ritorna una numero di patchs che rappresentano il frame nello spazio latente -> [Grandezza_Batch,Numero_patchs,Grandezza_token]
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken": #invece che patchs ho solo un token,  il CLS token. -> [Grandezza_Batch,Grandezza_token]
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim : [Grandezza_Batch,Grandezza_token] -> [Grandezza_Batch,1,Grandezza_token]
        return emb