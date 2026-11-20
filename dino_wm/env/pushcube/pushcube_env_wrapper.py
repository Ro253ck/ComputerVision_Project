import numpy as np
import gymnasium
import mani_skill.envs  # noqa: F401  registers PushCube-v1 with gymnasium

# ManiSkill uses gymnasium, but DINO-WM registers and creates envs with legacy gym (env/__init__.py, plan.py), so the wrapper is a legacy gym.Env like PushT and Wall
import gym as legacy_gym
from gym import spaces

from utils import aggregate_dct

ENV_ACTION_DIM = 4  # xyz end-effector delta + gripper
GOAL_RADIUS = 0.1  # PushCube-v1 goal_radius


class PushCubeEnvWrapper(legacy_gym.Env):
    """PushCube-v1 (ManiSkill3) behind the same interface as env/pusht and env/wall, so plan.py can use it unchanged.

    State is 9-dim: tcp xyz, cube xyz, goal xyz. Unlike PushT/Wall the scene cannot be rebuilt from it (no joint angles or orientations), so episodes are reset from their simulator seed (see PushCubeDataset.seeds)."""

    def __init__(self, img_size=224, **kwargs):
        self._env = gymnasium.make(
            "PushCube-v1",
            obs_mode="none",
            control_mode="pd_ee_delta_pos",
            render_mode="rgb_array",
            sim_backend="cpu",
            human_render_camera_configs=dict(width=img_size, height=img_size),
        )
        self.action_dim = self._env.action_space.shape[-1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self._episode_seed = None
        self._prev_tcp = None

    def update_env(self, env_info):
        """Receives the episode's {'seed': int} from PushCubeDataset, used by prepare() to reset to that scene."""
        if "seed" in env_info:
            self._episode_seed = int(env_info["seed"])

    def eval_state(self, goal_state, cur_state):
        """Success = 3D distance between cube and goal centres below GOAL_RADIUS.

        Slightly stricter than ManiSkill's own check, which uses the xy distance only: the cube centre sits ~2 cm above the goal disc, so this accepts up to ~9.8 cm in xy instead of 10 cm."""
        cube_pos = cur_state[3:6]
        goal_pos = cur_state[6:9]
        dist_to_goal = np.linalg.norm(cube_pos - goal_pos)
        success = bool(dist_to_goal < GOAL_RADIUS)
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def _read_state(self):
        u = self._env.unwrapped
        tcp = u.agent.tcp.pose.p.cpu().numpy().reshape(-1)
        cube = u.obj.pose.p.cpu().numpy().reshape(-1)
        goal = u.goal_region.pose.p.cpu().numpy().reshape(-1)
        return np.concatenate([tcp, cube, goal]).astype(np.float32)

    def _render(self):
        frame = self._env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:  # ManiSkill3 is vectorized: (1, H, W, 3) -> (H, W, 3)
            frame = frame[0]
        return frame

    def prepare(self, seed, init_state):
        """Resets to the episode's scene from its seed; init_state is unused (see class docstring)."""
        reset_seed = self._episode_seed if self._episode_seed is not None else int(seed)
        self._env.reset(seed=reset_seed)
        state = self._read_state()
        self._prev_tcp = state[:3].copy()
        proprio = np.concatenate([state[:3], np.zeros(3, dtype=np.float32)])  # zero initial velocity
        obs = {"visual": self._render(), "proprio": proprio}
        return obs, state

    def step_multiple(self, actions):
        obses = []
        states = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            _, rew, terminated, truncated, info = self._env.step(np.asarray(action, dtype=np.float32))
            state = self._read_state()
            tcp = state[:3]
            vel = tcp - self._prev_tcp
            self._prev_tcp = tcp.copy()
            proprio = np.concatenate([tcp, vel])
            obs = {"visual": self._render(), "proprio": proprio}

            obses.append(obs)
            states.append(state)
            rewards.append(float(rew))
            dones.append(bool(terminated) or bool(truncated))
            infos.append({"state": state, "success": bool(info.get("success", False))})

        obses = aggregate_dct(obses)
        states = np.stack(states)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        infos["state"] = states  # overwrite with the already-stacked array, matching pusht/wall
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """seed: int; init_state: unused, see prepare(); actions: (T, action_dim); returns obses (dict of (T+1, ...) arrays) and states (T+1, state_dim)."""
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
