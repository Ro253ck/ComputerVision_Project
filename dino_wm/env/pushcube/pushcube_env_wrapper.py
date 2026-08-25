"""
Wrapper per PushCube-v1 (ManiSkill3) con la stessa interfaccia di
env/pusht/pusht_wrapper.py e env/wall/wall_env_wrapper.py, cosi' da poter
essere usato da plan.py senza modifiche al resto della pipeline.

Differenza importante rispetto a PushT/Wall: quei simulatori sono a bassa
dimensionalita' e il loro stato compatto basta a ricostruire l'intera scena
(env.set_init_state(state) funziona). ManiSkill no: il nostro state a 9
valori (tcp+cubo+goal) non contiene gli angoli dei giunti del robot ne' i
quaternioni di orientamento, quindi non e' invertibile in una configurazione
fisica esatta. Usiamo invece il SEED dell'episodio (salvato a parte in
data/pushcube_noise/{train,val}/seeds.pkl, vedi scripts/recover_pushcube_seeds.py)
con env.reset(seed=...), che riproduce la scena iniziale byte per byte
(verificato durante il debug della generazione dataset).

update_env() riceve il seed dal dataset (via PushCubeDataset che lo espone
nel 4o elemento restituito da __getitem__) e lo salva; rollout()/prepare()
lo usano al posto dello stato compatto per il reset.
"""
import numpy as np
import gymnasium
import mani_skill.envs  # noqa: F401  (registra PushCube-v1 in gymnasium)

# gym "vecchio" (quello usato dal sistema di registrazione in env/__init__.py
# e da plan.py per gym.make("pushcube", ...)) e' un pacchetto DIVERSO da
# gymnasium (quello di ManiSkill) - PushTEnv/DotWall ereditano entrambi da
# gym.Env per soddisfare l'interfaccia che il vecchio gym.make() si aspetta
# (es. l'attributo .unwrapped); facciamo lo stesso qui.
import gym as legacy_gym
from gym import spaces

from utils import aggregate_dct

ENV_ACTION_DIM = 4  # xyz end-effector delta + gripper
GOAL_RADIUS = 0.1   # env.unwrapped.goal_radius reale di PushCube-v1, verificato


class PushCubeEnvWrapper(legacy_gym.Env):
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
        """env_info arriva da PushCubeDataset.get_frames come {'seed': int},
        gia' smistato per-traiettoria dal livello di vettorizzazione."""
        if "seed" in env_info:
            self._episode_seed = int(env_info["seed"])

    def eval_state(self, goal_state, cur_state):
        """successo = cubo entro GOAL_RADIUS dal goal, come definito
        internamente da ManiSkill per PushCube-v1 (env.unwrapped.goal_radius)."""
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
        if frame.ndim == 4:  # (1, H, W, 3) -> (H, W, 3), ManiSkill3 e' vettorizzato
            frame = frame[0]
        return frame

    def prepare(self, seed, init_state):
        """Reset alla scena esatta dell'episodio. init_state non e' usabile
        direttamente per ManiSkill (vedi docstring del modulo) - usiamo il
        seed salvato da update_env se disponibile, altrimenti seed passato."""
        reset_seed = self._episode_seed if self._episode_seed is not None else int(seed)
        self._env.reset(seed=reset_seed)
        state = self._read_state()
        self._prev_tcp = state[:3].copy()
        proprio = np.concatenate([state[:3], np.zeros(3, dtype=np.float32)])  # velocita' iniziale 0
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
        infos["state"] = states  # sovrascrive con l'array gia' impilato, coerente con pusht/wall
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        seed: int
        init_state: (state_dim,) - non usato direttamente, vedi note sopra
        actions: (T, action_dim)
        obses: dict con array (T+1, ...)
        states: (T+1, state_dim)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
