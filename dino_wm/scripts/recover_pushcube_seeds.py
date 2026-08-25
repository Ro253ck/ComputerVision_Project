"""
Ricalcola e salva il seed ManiSkill di ogni episodio gia' generato in
data/pushcube_noise, SENZA ri-simulare nulla e SENZA toccare i file esistenti
usati dal training (states.pth, abs_actions.pth, velocities.pth, i video).

Funziona perche' la scelta del seed per ogni episodio e' deterministica in
generate_pushcube_dataset.py: episodio i usa sempre
demos[i % len(demos)]["seed"], quindi basta rileggere il file delle demo e
rifare lo stesso calcolo - nessuna fisica coinvolta, e' istantaneo.

Serve per il planning: env.rollout() dovra' poter tornare esattamente alla
scena iniziale di un episodio del dataset (env.reset(seed=...) riproduce lo
stato iniziale byte per byte, verificato durante il debug della generazione),
cosa che i soli states.pth/abs_actions.pth non permettono di fare per un
ambiente ManiSkill (a differenza di PushT/Wall, il nostro stato compatto a 9
valori non contiene abbastanza informazione per ricostruire la scena fisica
completa: mancano gli angoli dei giunti del robot, i quaternioni, ecc).

Uso:
    python scripts/recover_pushcube_seeds.py \
        --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.none.pd_ee_delta_pos.physx_cpu.h5 \
        --out-dir /work/cvcs2026/LubiMoRe/dino_wm/data/pushcube_noise \
        --num-train-rollouts 10000 --num-val-rollouts 50

I valori di --num-train-rollouts/--num-val-rollouts DEVONO combaciare con
quelli usati per generare il dataset esistente (controllabili con
`ls data/pushcube_noise/{train,val}/obses | wc -l`).
"""
import argparse
import pickle
from pathlib import Path

from generate_pushcube_dataset import load_base_demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-train-rollouts", type=int, required=True)
    parser.add_argument("--num-val-rollouts", type=int, required=True)
    args = parser.parse_args()

    demos = load_base_demos(args.demo_path)
    print(f"Caricate {len(demos)} demo di base da {args.demo_path}")

    n_train = args.num_train_rollouts
    n_val = args.num_val_rollouts
    split_ranges = {
        "train": range(0, n_train),
        "val": range(n_train, n_train + n_val),
    }

    out_dir = Path(args.out_dir)
    for split, idx_range in split_ranges.items():
        seeds = [int(demos[i % len(demos)]["seed"]) for i in idx_range]
        split_dir = out_dir / split
        assert split_dir.exists(), f"{split_dir} non esiste - controlla --out-dir"
        with open(split_dir / "seeds.pkl", "wb") as f:
            pickle.dump(seeds, f)
        print(f"{split}: salvati {len(seeds)} seed in {split_dir / 'seeds.pkl'}")


if __name__ == "__main__":
    main()
