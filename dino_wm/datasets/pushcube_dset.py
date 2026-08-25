import torch
import decord
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
decord.bridge.set_bridge("torch")

# statistiche precalcolate su data/pushcube_noise/train (10000 rollout,
# seq_lengths corretti con scripts/trim_pushcube_seq_lengths.py - lunghezza
# variabile fino a poco dopo il successo, non piu' 150 fissi), vedi
# scripts/compute_pushcube_stats.py
ACTION_MEAN = torch.tensor([0.0152, 0.0005, -0.0513, -0.9799])
ACTION_STD = torch.tensor([0.0870, 0.0540, 0.0872, 0.0298])
PROPRIO_MEAN = torch.tensor([0.0231, -0.0003, 0.0543, 0.0008, -0.0000, -0.0016])
PROPRIO_STD = torch.tensor([0.0696, 0.0499, 0.0578, 0.0036, 0.0017, 0.0032])


class PushCubeDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pushcube_noise",
        normalize_action: bool = True,
        with_velocity: bool = True,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        self.states = torch.load(self.data_path / "states.pth").float()  # (N, T, 9) tcp+cube+goal
        self.actions = torch.load(self.data_path / "abs_actions.pth").float()  # (N, T, 4) xyz+gripper

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        # seed ManiSkill di ogni episodio (scripts/recover_pushcube_seeds.py):
        # serve solo per il planning (PushCubeEnvWrapper.rollout/prepare), che
        # deve poter tornare alla scena fisica esatta di un episodio - lo
        # stato compatto (9 valori) non basta a ricostruirla per ManiSkill,
        # a differenza di PushT/Wall. Opzionale: se il file non esiste
        # (dataset generato prima di questa modifica, o serve solo per il
        # training) get_frames restituisce semplicemente {} come per prima.
        seeds_path = self.data_path / "seeds.pkl"
        if seeds_path.exists():
            with open(seeds_path, "rb") as f:
                self.seeds = pickle.load(f)
        else:
            self.seeds = None

        self.n_rollout = n_rollout
        n = self.n_rollout if self.n_rollout else len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        if self.seeds is not None:
            self.seeds = self.seeds[:n]

        # proprio = solo il TCP (posizione del braccio), MAI cubo/goal: quelli
        # il modello deve dedurli dal canale visivo, non riceverli come input
        # numerico diretto (altrimenti il confronto tra encoder perde senso)
        self.proprios = self.states[..., :3].clone()
        self.with_velocity = with_velocity
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")[:n].float()
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.proprio_mean = PROPRIO_MEAN[: self.proprio_dim]
            self.proprio_std = PROPRIO_STD[: self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        # state (tcp+cubo+goal, unita' grezze) non viene normalizzato nel
        # training (solo proprio/azioni lo sono, come per PushT) - mean=0/std=1
        # e' un no-op, usato da plan.py solo per metriche diagnostiche
        # (Preprocessor.normalize_states), non per l'obiettivo di planning
        self.state_mean = torch.zeros(self.state_dim)
        self.state_std = torch.ones(self.state_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]

        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)

        image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        info = {"seed": self.seeds[idx]} if self.seeds is not None else {}
        return obs, act, state, info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_pushcube_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/pushcube_noise",
    normalize_action=True,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
):
    train_dset = PushCubeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )
    val_dset = PushCubeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
