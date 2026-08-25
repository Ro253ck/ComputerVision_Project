"""
Calcola ACTION_MEAN/STD e PROPRIO_MEAN/STD reali per PushCube, da incollare
in datasets/pushcube_dset.py al posto dei placeholder (mean=0, std=1).

Uso:
    python scripts/compute_pushcube_stats.py \
        --data-path /work/cvcs2026/LubiMoRe/dino_wm/data/pushcube_noise/train
"""
import argparse
import pickle
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                         help="cartella train/ del dataset pushcube_noise")
    args = parser.parse_args()

    states = torch.load(f"{args.data_path}/states.pth", weights_only=True).float()
    actions = torch.load(f"{args.data_path}/abs_actions.pth", weights_only=True).float()
    velocities = torch.load(f"{args.data_path}/velocities.pth", weights_only=True).float()
    with open(f"{args.data_path}/seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)

    proprio = torch.cat([states[..., :3], velocities], dim=-1)  # tcp_pos + tcp_vel

    # media/std solo sugli step VALIDI di ciascun episodio (fino a
    # seq_lengths[i]), non sull'intero tensore - altrimenti, con lunghezza
    # variabile, si mischierebbero dentro anche gli step della "coda
    # congelata" oltre la fine reale di ogni episodio
    action_rows, proprio_rows = [], []
    for i, T in enumerate(seq_lengths):
        action_rows.append(actions[i, :T])
        proprio_rows.append(proprio[i, :T])
    actions_valid = torch.cat(action_rows, dim=0)
    proprio_valid = torch.cat(proprio_rows, dim=0)

    action_mean = actions_valid.mean(dim=0)
    action_std = actions_valid.std(dim=0)
    proprio_mean = proprio_valid.mean(dim=0)
    proprio_std = proprio_valid.std(dim=0)

    def fmt(t):
        return "torch.tensor([" + ", ".join(f"{v:.4f}" for v in t.tolist()) + "])"

    print(f"ACTION_MEAN = {fmt(action_mean)}")
    print(f"ACTION_STD = {fmt(action_std)}")
    print(f"PROPRIO_MEAN = {fmt(proprio_mean)}")
    print(f"PROPRIO_STD = {fmt(proprio_std)}")


if __name__ == "__main__":
    main()
