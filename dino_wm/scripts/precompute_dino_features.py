"""
Precalcola una volta sola le feature dell'encoder DINO (congelato) per ogni
episodio, e le salva su disco. Va lanciato UNA volta prima del training, con la
stessa identica configurazione (encoder, dataset, img_size) del training, cosi'
le feature coincidono con quelle che il modello produrrebbe live.

Uso (puntando ai dati PERSISTENTI, non a TMPDIR):
    DATASET_DIR=/work/cvcs2026/LubiMoRe/dino_wm/data \
    python precompute_dino_features.py env=pusht frameskip=5 num_hist=3

Scrive le feature in:  <data_path>/<split>/dino_feats/episode_XXX.pt
"""
import hydra
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms


@hydra.main(config_path="../conf", config_name="train")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) encoder identico a quello del training, congelato e in eval
    encoder = hydra.utils.instantiate(cfg.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 2) encoder_transform identico a quello costruito nel modello
    #    (Resize a (img_size // 16) * patch_size)
    num_side = cfg.img_size // 16
    enc_img_size = num_side * encoder.patch_size
    encoder_transform = transforms.Compose([transforms.Resize(enc_img_size)])

    # 3) dataset GREZZO (senza cache): riusa lo stesso loader del training,
    #    quindi applica la stessa `transform` in get_frames
    _, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )

    chunk_size = 64  # quanti frame encodare per volta; abbassa se vai in OOM

    for split in ["train", "valid"]:
        dset = traj_dsets[split]                       # PushTDataset grezzo
        cache_dir = Path(dset.data_path) / "dino_feats"
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{split}] scrivo {len(dset)} episodi in {cache_dir}")

        for idx in tqdm(range(len(dset)), desc=f"encode {split}"):
            out_file = cache_dir / f"episode_{idx:03d}.pt"
            if out_file.exists():
                continue  # gia' fatto: riprendibile se il job viene killato

            obs, _, _, _ = dset[idx]                   # obs["visual"]: (T, 3, H, W)
            visual = obs["visual"].to(device)

            feats = []
            with torch.no_grad():
                for s in range(0, visual.shape[0], chunk_size):
                    chunk = encoder_transform(visual[s:s + chunk_size])
                    emb = encoder.forward(chunk)        # (n, num_patches, emb_dim)
                    feats.append(emb.half().cpu())      # fp32 per risparmiare spazio, se vuoi ripsarmiare spazio metti emb.half().cpu()
            feats = torch.cat(feats, dim=0)             # (T, num_patches, emb_dim)
            torch.save(feats, out_file)

    print("Precalcolo completato.")


if __name__ == "__main__":
    main()