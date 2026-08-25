"""
Corregge RETROATTIVAMENTE seq_lengths.pkl di un dataset pushcube_noise gia'
generato, senza ri-simulare nulla: legge states.pth (che contiene gia'
posizione di cubo e goal ad ogni step), trova per ciascun episodio il primo
step in cui il cubo e' entrato nel raggio di successo, e accorcia la
lunghezza registrata a quel punto (+ un piccolo margine). Le finestre di
training/planning oltre quel punto (la "coda congelata" post-successo) non
verranno piu' usate, perche' TrajSlicerDataset si basa su get_seq_length()
per decidere quante finestre generare per episodio - senza bisogno di
toccare video o tensori.

Uso:
    python scripts/trim_pushcube_seq_lengths.py \
        --data-path /work/cvcs2026/LubiMoRe/dino_wm/data/pushcube_noise/train
    python scripts/trim_pushcube_seq_lengths.py \
        --data-path /work/cvcs2026/LubiMoRe/dino_wm/data/pushcube_noise/val

Va lanciato su ENTRAMBE le cartelle (train e val). Salva un backup
(seq_lengths.pkl.bak) prima di sovrascrivere.
"""
import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

GOAL_RADIUS = 0.1  # env.unwrapped.goal_radius reale di PushCube-v1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                         help="cartella train/ o val/ di data/pushcube_noise")
    parser.add_argument("--success-buffer", type=int, default=10,
                         help="quanti step extra tenere dopo il primo successo "
                              "(qualche esempio di 'stato di goal raggiunto', senza esagerare)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    states = torch.load(data_path / "states.pth", weights_only=True)  # (N, T, 9)
    with open(data_path / "seq_lengths.pkl", "rb") as f:
        old_seq_lengths = pickle.load(f)

    cube_pos = states[:, :, 3:6]
    goal_pos = states[:, :, 6:9]
    dist = (cube_pos - goal_pos).norm(dim=-1)  # (N, T)

    new_seq_lengths = []
    n_never_succeeded = 0
    for i in range(states.shape[0]):
        max_T = old_seq_lengths[i]
        success_steps = (dist[i, :max_T] < GOAL_RADIUS).nonzero()
        if len(success_steps) == 0:
            # non ha mai raggiunto il successo: teniamo la lunghezza piena
            # (nessuna coda congelata "vera" da tagliare, l'episodio non si e' mai risolto)
            new_seq_lengths.append(max_T)
            n_never_succeeded += 1
        else:
            first_success = success_steps[0].item()
            new_len = min(first_success + args.success_buffer + 1, max_T)
            new_seq_lengths.append(new_len)

    backup_path = data_path / "seq_lengths.pkl.bak"
    shutil.copy(data_path / "seq_lengths.pkl", backup_path)
    with open(data_path / "seq_lengths.pkl", "wb") as f:
        pickle.dump(new_seq_lengths, f)

    old_arr = np.array(old_seq_lengths)
    new_arr = np.array(new_seq_lengths)
    print(f"Episodi totali: {len(new_seq_lengths)}")
    print(f"Mai andati a successo (lunghezza invariata): {n_never_succeeded} "
          f"({100 * n_never_succeeded / len(new_seq_lengths):.1f}%)")
    print(f"Lunghezza media PRIMA: {old_arr.mean():.1f}")
    print(f"Lunghezza media DOPO: {new_arr.mean():.1f}")
    print(f"Backup del vecchio seq_lengths.pkl salvato in: {backup_path}")


if __name__ == "__main__":
    main()
