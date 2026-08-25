import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path("/work/cvcs2026/LubiMoRe/dino_wm/data/pusht_noise")

for split in ["train", "val"]:
    cache_dir = BASE_DIR / split / "dino_feats"
    if not cache_dir.exists():
        print(f"[{split}] ATTENZIONE: {cache_dir} non esiste, salto.")
        continue

    files = sorted(cache_dir.glob("episode_*.pt"))
    print(f"[{split}] trovati {len(files)} file in {cache_dir}")

    for pt_file in files:
        feats = torch.load(pt_file)              # shape (T, N, D)
        feats_np = feats.numpy().astype(np.float16)
        out_path = cache_dir / (pt_file.stem + ".dat")
        feats_np.tofile(out_path)                 # binario grezzo
        print(pt_file.stem, feats_np.shape)