"""
Genera un dataset di rollout per PushCube-v1 (ManiSkill3) nel formato atteso
da dino_wm (stesso layout di datasets/pusht_dset.py):

    <out_dir>/{train,val}/
        obses/episode_000.mp4, episode_001.mp4, ...
        states.pth        # (N, T, state_dim) float32
        abs_actions.pth   # (N, T, action_dim) float32
        velocities.pth    # (N, T, vel_dim) float32  (velocita' lineare del TCP)
        seq_lengths.pkl   # list[int] lunghe N, qui sempre = max_steps (lunghezza fissa)

STEP A - una tantum, PRIMA di questo script (tooling ufficiale ManiSkill,
non reimplementato qui perche' piu' affidabile del nostro codice):

    python -m mani_skill.utils.download_demo PushCube-v1

    python -m mani_skill.trajectory.replay_trajectory \\
        --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \\
        --target-control-mode pd_ee_delta_pos \\
        --obs-mode none \\
        --save-traj \\
        --num-envs 8

    Questo produce trajectory.none.pd_ee_delta_pos.physx_cpu.h5 (+ .json con
    i metadati, incluso il seed di reset di ogni episodio) con azioni gia'
    nel control mode che usiamo per il world model. Il nome file include
    anche obs-mode e sim-backend usati, non solo il control mode. Sono le
    traiettorie "esperte" di base: questo script le RE-ESEGUE nel simulatore
    aggiungendo rumore gaussiano alle azioni, cosi' da ottenere rollout
    diversi tra loro pur restando vicini a una traiettoria che risolve il
    task (stessa filosofia di data/pusht_noise).

STEP B - questo script:

    python generate_pushcube_dataset.py \\
        --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.none.pd_ee_delta_pos.physx_cpu.h5 \\
        --out-dir /work/cvcs2026/LubiMoRe/dino_wm/data/pushcube_noise \\
        --num-train-rollouts 10000 \\
        --num-val-rollouts 50 \\
        --max-steps 50 \\
        --action-noise-std 0.05

    train e val sono due pool INDIPENDENTI (stessa convenzione di
    data/pusht_noise: train=18685, val=21 episodi fissi, non uno split
    percentuale di un unico pool) - cosi' env.dataset.n_rollout=10000 in
    training trova per intero i 10000 episodi di train che si aspetta.

ATTENZIONE: script scritto senza poter eseguire ManiSkill in locale (non e'
installato in questo sandbox) - alcuni nomi di chiavi in obs/info (es.
"success", "tcp_pose") vanno verificati sul cluster. Lanciare PRIMA con
--num-rollouts 20 e controllare a occhio un paio di episode_XXX.mp4 e le
shape dei tensori salvati, prima di lanciare il job da 10000.
"""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import imageio
import numpy as np
import torch
from tqdm import tqdm

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registra PushCube-v1 in gym)


def load_base_demos(demo_path):
    """Carica le traiettorie esperte convertite (STEP A) come lista di dict
    {seed, actions (T, action_dim), init_state}.

    init_state e' lo stato COMPLETO della simulazione al tempo 0 (posa di
    cubo/goal/tavolo + stato del braccio Panda), preso da env_states nel
    file .h5. Serve per forzare la condizione iniziale esatta con
    env.set_state_dict() invece di affidarsi a env.reset(seed=...): il seed
    da solo non garantisce di riprodurre esattamente la stessa configurazione
    per cui le azioni della demo erano state calcolate (es. rumore di
    inizializzazione del robot non deterministico rispetto al seed)."""
    demo_path = Path(demo_path)
    with open(demo_path.with_suffix(".json")) as f:
        meta = json.load(f)

    demos = []
    with h5py.File(demo_path, "r") as f:
        for ep in meta["episodes"]:
            traj_id = f"traj_{ep['episode_id']}"
            g = f[traj_id]
            actions = np.array(g["actions"])  # (T, action_dim)
            seed = ep["reset_kwargs"].get("seed", ep["episode_seed"])

            init_state = {
                "actors": {
                    name: np.array(g["env_states"]["actors"][name][0])
                    for name in g["env_states"]["actors"].keys()
                },
                "articulations": {
                    name: np.array(g["env_states"]["articulations"][name][0])
                    for name in g["env_states"]["articulations"].keys()
                },
            }
            demos.append({"seed": seed, "actions": actions, "init_state": init_state})
    return demos


def _attempt_episode(env, base_actions, seed, max_steps, action_noise_std, rng):
    """Un singolo tentativo di ri-eseguire la traiettoria esperta con rumore,
    a lunghezza fissa max_steps. Una volta raggiunto il successo, congela
    l'azione a zero (gripper escluso) per il resto dell'episodio invece di
    continuare a spingere il cubo.

    Ritorna anche se il successo e' stato raggiunto: la dinamica di contatto
    rigido e' caotica, quindi anche con azioni identiche e seed identico due
    esecuzioni possono divergere leggermente nel momento esatto del contatto
    (verificato: le prime ~60 step combaciano byte per byte, poi una
    differenza di ~1mm al contatto puo' far fallire la spinta) - per questo
    chi chiama questa funzione ritenta se non riesce, vedi rollout_one_episode."""
    obs, info = env.reset(seed=int(seed))

    frames = []
    tcp_positions = []
    cube_positions = []
    goal_positions = []
    actions_taken = []

    action_dim = env.action_space.shape[-1]
    success_reached = False
    first_success_step = None
    # ultima azione del gripper comandata dalla demo (l'ultima colonna
    # dell'action space, tipicamente ~-1 = chiuso per tutta la traiettoria
    # in PushCube): da mantenere anche a movimento congelato, altrimenti
    # azzerarla di colpo (da -1 a 0) da' uno scossone al gripper che
    # disturba il cubo subito dopo il successo, vanificandolo
    last_gripper_action = base_actions[min(len(base_actions), 1) - 1, -1] if len(base_actions) > 0 else 0.0

    for t in range(max_steps):
        if success_reached:
            action = np.zeros(action_dim, dtype=np.float32)
            action[-1] = last_gripper_action
        elif t < len(base_actions):
            noise = rng.normal(0, action_noise_std, size=action_dim).astype(np.float32)
            action = np.clip(base_actions[t] + noise, -1.0, 1.0)
            last_gripper_action = action[-1]
        else:
            # traiettoria esperta piu' corta di max_steps: resta fermo, ma
            # mantieni comunque il gripper com'era, non azzerarlo di colpo
            action = np.zeros(action_dim, dtype=np.float32)
            action[-1] = last_gripper_action

        obs, rew, terminated, truncated, info = env.step(action)
        if bool(info.get("success", False)) and not success_reached:
            success_reached = True
            first_success_step = t

        frame = env.render()  # rgb_array mode; ManiSkill3 e' vettorizzato, quindi
        if hasattr(frame, "cpu"):  # puo' essere un tensore torch su GPU con una
            frame = frame.cpu().numpy()  # dimensione di batch anche con un solo env
        frame = np.asarray(frame)
        if frame.ndim == 4:  # (1, H, W, 3) -> (H, W, 3)
            frame = frame[0]
        frames.append(frame)

        # stato "privilegiato": posa TCP + posizione cubo + posizione goal
        # NB: le chiavi esatte dipendono dalla versione di ManiSkill, verificare
        # con `print(env.unwrapped.get_state_dict().keys())` sul cluster.
        unwrapped = env.unwrapped
        tcp_pos = unwrapped.agent.tcp.pose.p.cpu().numpy().reshape(-1)
        cube_pos = unwrapped.obj.pose.p.cpu().numpy().reshape(-1)
        goal_pos = unwrapped.goal_region.pose.p.cpu().numpy().reshape(-1)

        tcp_positions.append(tcp_pos)
        cube_positions.append(cube_pos)
        goal_positions.append(goal_pos)
        actions_taken.append(action)

    frames = np.stack(frames)  # (T, H, W, 3)
    tcp_positions = np.stack(tcp_positions)  # (T, 3)
    cube_positions = np.stack(cube_positions)  # (T, 3)
    goal_positions = np.stack(goal_positions)  # (T, 3)
    actions_taken = np.stack(actions_taken)  # (T, action_dim)

    # velocita' del TCP come derivata finita delle posizioni (coerente con
    # come pusht_dset ricava le "velocities" per il proprio)
    velocities = np.zeros_like(tcp_positions)
    velocities[1:] = tcp_positions[1:] - tcp_positions[:-1]

    # stato = [tcp_pos(3), cube_pos(3), goal_pos(3)]  -> state_dim = 9
    state = np.concatenate([tcp_positions, cube_positions, goal_positions], axis=-1)

    return frames, state, actions_taken, velocities, success_reached, first_success_step


def rollout_one_episode(env, base_actions, seed, init_state, max_steps, action_noise_std, rng,
                         max_retry=10, success_dist_threshold=0.1):
    # success_dist_threshold=0.1 = env.unwrapped.goal_radius reale di
    # PushCube-v1 (verificato sul cluster), non un valore a caso
    """Chiama _attempt_episode fino a max_retry volte, tenendo il primo
    tentativo che raggiunge il successo (o quello con la distanza finale
    cubo-goal minore, se nessuno riesce entro max_retry tentativi)."""
    best = None
    best_dist = np.inf
    for attempt in range(max_retry):
        frames, state, actions_taken, velocities, success, first_success_step = _attempt_episode(
            env, base_actions, seed, max_steps, action_noise_std, rng,
        )
        final_dist = np.linalg.norm(state[-1, 3:6] - state[-1, 6:9])
        if success or final_dist < success_dist_threshold:
            return frames, state, actions_taken, velocities, first_success_step
        if final_dist < best_dist:
            best_dist = final_dist
            best = (frames, state, actions_taken, velocities, first_success_step)
    # nessun tentativo riuscito entro max_retry: teniamo comunque il migliore,
    # invece di scartare l'episodio (in produzione andrebbe loggato quanti
    # episodi falliscono anche dopo i retry, per capire se max_retry va alzato)
    return best


def write_episode_video(path, frames, fps=10):
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_split_tensors(split_dir, states, actions, velocities, seq_lengths):
    split_dir = Path(split_dir)
    if len(states) == 0:
        return
    torch.save(torch.tensor(np.stack(states), dtype=torch.float32), split_dir / "states.pth")
    torch.save(torch.tensor(np.stack(actions), dtype=torch.float32), split_dir / "abs_actions.pth")
    torch.save(torch.tensor(np.stack(velocities), dtype=torch.float32), split_dir / "velocities.pth")
    with open(split_dir / "seq_lengths.pkl", "wb") as f:
        pickle.dump(list(seq_lengths), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", type=str, required=True,
                         help="trajectory.pd_ee_delta_pos.h5 prodotto dallo STEP A")
    parser.add_argument("--out-dir", type=str, required=True)
    # train e val sono due pool INDIPENDENTI, non uno split percentuale di un
    # unico pool: e' la stessa convenzione di data/pusht_noise (train=18685,
    # val=21 episodi fissi). Se poi in training usi env.dataset.n_rollout=10000,
    # il train pool deve contenerne almeno 10000 per intero.
    parser.add_argument("--num-train-rollouts", type=int, default=10000)
    parser.add_argument("--num-val-rollouts", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50,
                         help="lunghezza fissa di ogni rollout")
    parser.add_argument("--action-noise-std", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--success-buffer", type=int, default=10,
                         help="quanti step extra tenere (in seq_lengths) dopo il primo "
                              "successo, invece di continuare fino a max_steps")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    demos = load_base_demos(args.demo_path)
    print(f"Caricate {len(demos)} traiettorie esperte di base da {args.demo_path}")

    env = gym.make(
        "PushCube-v1",
        obs_mode="none",           # non ci serve l'obs del simulatore: renderizziamo noi i frame
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="cpu",         # un env alla volta: niente bisogno del backend gpu vettorizzato
        max_episode_steps=args.max_steps,
        # risoluzione della camera di rendering "human"/rgb_array, coerente
        # col resto della pipeline (default ManiSkill era 512x512)
        human_render_camera_configs=dict(width=args.img_size, height=args.img_size),
    )

    # split train/val deciso PRIMA di generare, cosi' possiamo scrivere ogni
    # video su disco subito e non tenere mai in RAM piu' di un rollout di
    # frame per volta (10000 rollout x 50 step x 224x224x3 sarebbero ~75GB
    # se accumulati tutti insieme prima di salvare)
    # train e val sono due pool indipendenti: il primo blocco di indici va
    # in train, il secondo (che continua a ciclare sulle demo/seed) in val.
    n_train = args.num_train_rollouts
    n_val = args.num_val_rollouts
    split_ranges = [("train", range(0, n_train)), ("val", range(n_train, n_train + n_val))]

    out_dir = Path(args.out_dir)
    split_states = {"train": [], "val": []}
    split_actions = {"train": [], "val": []}
    split_vel = {"train": [], "val": []}
    split_seq_lengths = {"train": [], "val": []}
    n_never_succeeded = {"train": 0, "val": 0}
    for split in ("train", "val"):
        (out_dir / split / "obses").mkdir(parents=True, exist_ok=True)

    for split, idx_range in split_ranges:
        for ep_idx, i in enumerate(tqdm(idx_range, desc=f"rollout ({split})")):
            base = demos[i % len(demos)]
            # IMPORTANTE: il seed deve essere quello ORIGINALE della demo, non
            # uno diverso per ogni rollout - il seed determina la posizione
            # random di cubo/goal nella scena, e le azioni registrate in
            # base["actions"] sono state pianificate apposta per QUELLA
            # configurazione iniziale. Cambiare seed = azioni "cieche" su una
            # scena diversa da quella per cui erano state calcolate -> quasi
            # mai successo. La diversita' tra i rollout viene solo dal rumore
            # sulle azioni, non dal seed.
            frames, state, actions, vel, first_success_step = rollout_one_episode(
                env, base["actions"], seed=base["seed"], init_state=base["init_state"],
                max_steps=args.max_steps, action_noise_std=args.action_noise_std, rng=rng,
            )

            write_episode_video(out_dir / split / "obses" / f"episode_{ep_idx:03d}.mp4", frames)
            del frames  # libera subito la parte pesante in RAM

            # lunghezza EFFETTIVA salvata in seq_lengths: fino a poco dopo il
            # primo successo, non fino a max_steps - cosi' le finestre di
            # training/planning (TrajSlicerDataset, che usa get_seq_length())
            # non pescano mai dalla "coda congelata" post-successo, che
            # diluisce/degrada il segnale utile (vedi discussione). Se
            # l'episodio non ha mai avuto successo (capita con i casi piu'
            # duri anche dopo i retry), teniamo la lunghezza piena: non c'e'
            # una vera coda congelata da tagliare in quel caso.
            if first_success_step is None:
                seq_len = args.max_steps
                n_never_succeeded[split] += 1
            else:
                seq_len = min(first_success_step + args.success_buffer + 1, args.max_steps)

            split_states[split].append(state)
            split_actions[split].append(actions)
            split_vel[split].append(vel)
            split_seq_lengths[split].append(seq_len)
    env.close()

    for split in ("train", "val"):
        save_split_tensors(
            out_dir / split, split_states[split], split_actions[split], split_vel[split],
            seq_lengths=split_seq_lengths[split],
        )
        avg_len = sum(split_seq_lengths[split]) / len(split_seq_lengths[split])
        print(f"{split}: {len(split_states[split])} rollout salvati in {out_dir / split}"
              f" (lunghezza media {avg_len:.1f}/{args.max_steps},"
              f" mai andati a successo: {n_never_succeeded[split]})")


if __name__ == "__main__":
    main()
