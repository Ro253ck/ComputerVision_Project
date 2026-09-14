import numpy as np
import gymnasium
import mani_skill.envs  # noqa: F401  registers PushCube-v1 with gymnasium

# legacy gym (used by env/__init__.py's registration and plan.py's gym.make("pushcube", ...)) is a different package from gymnasium (ManiSkill's); PushTEnv/DotWall both inherit from gym.Env for the same reason, so we do too here
import gym as legacy_gym
from gym import spaces

from utils import aggregate_dct

ENV_ACTION_DIM = 4  # xyz end-effector delta + gripper
GOAL_RADIUS = 0.1  # PushCube-v1's real env.unwrapped.goal_radius, verified on the cluster


class PushCubeEnvWrapper(legacy_gym.Env):
    """Wraps PushCube-v1 (ManiSkill3) with the same interface as env/pusht and env/wall, so plan.py needs no changes to use it.

    Unlike PushT/Wall, ManiSkill's scene can't be rebuilt from the compact 9-dim state (it lacks joint angles/orientation quaternions), so we reset with the episode's own seed (see PushCubeDataset.seeds) instead."""

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
        """Receives {'seed': int} from PushCubeDataset.get_frames, already routed per-trajectory."""
        if "seed" in env_info:
            self._episode_seed = int(env_info["seed"])

    def eval_state(self, goal_state, cur_state):
        """Success = cube within GOAL_RADIUS of the goal, matching ManiSkill's own PushCube-v1 criterion."""
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
        if frame.ndim == 4:  # (1, H, W, 3) -> (H, W, 3), ManiSkill3 is vectorized
            frame = frame[0]
        return frame

    def prepare(self, seed, init_state):
        """Resets to the episode's exact scene; init_state is unused (see class docstring), we reset by seed instead."""
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
