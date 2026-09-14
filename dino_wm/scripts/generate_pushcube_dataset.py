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
import mani_skill.envs  # noqa: F401  registers PushCube-v1 with gym


def load_base_demos(demo_path):
    """Loads the expert demo trajectories as a list of {seed, actions} dicts."""
    demo_path = Path(demo_path)
    with open(demo_path.with_suffix(".json")) as f:
        meta = json.load(f)

    demos = []
    with h5py.File(demo_path, "r") as f:
        for ep in meta["episodes"]:
            traj_id = f"traj_{ep['episode_id']}"
            actions = np.array(f[traj_id]["actions"])  # (T, action_dim)
            seed = ep["reset_kwargs"].get("seed", ep["episode_seed"])
            demos.append({"seed": seed, "actions": actions})
    return demos


def _read_state(env):
    """Reads TCP pose, cube position, and goal position from the unwrapped ManiSkill env."""
    u = env.unwrapped
    tcp_pos = u.agent.tcp.pose.p.cpu().numpy().reshape(-1)
    cube_pos = u.obj.pose.p.cpu().numpy().reshape(-1)
    goal_pos = u.goal_region.pose.p.cpu().numpy().reshape(-1)
    return tcp_pos, cube_pos, goal_pos


def _render_frame(env):
    """Renders one RGB frame, squeezing ManiSkill's batch dimension."""
    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:  # (1, H, W, 3) -> (H, W, 3)
        frame = frame[0]
    return frame


def _stack_episode(frames, tcp_positions, cube_positions, goal_positions, actions_taken):
    """Stacks per-step lists into arrays and builds the 9-dim state (tcp, cube, goal)."""
    frames = np.stack(frames)
    tcp_positions = np.stack(tcp_positions)
    cube_positions = np.stack(cube_positions)
    goal_positions = np.stack(goal_positions)
    actions_taken = np.stack(actions_taken)
    velocities = np.zeros_like(tcp_positions)
    velocities[1:] = tcp_positions[1:] - tcp_positions[:-1]  # finite-difference TCP velocity
    state = np.concatenate([tcp_positions, cube_positions, goal_positions], axis=-1)
    return frames, state, actions_taken, velocities


def _attempt_episode_train(env, base_actions, seed, max_steps, action_noise_std, rng):
    """One attempt at replaying a noisy expert trajectory for the full max_steps; freezes to a zero action (gripper held) after first success instead of continuing to push."""
    env.reset(seed=int(seed))
    frames, tcp_positions, cube_positions, goal_positions, actions_taken = [], [], [], [], []
    action_dim = env.action_space.shape[-1]
    success_reached = False
    first_success_step = None
    last_gripper_action = base_actions[min(len(base_actions), 1) - 1, -1] if len(base_actions) > 0 else 0.0  # held steady once frozen, so it doesn't shock the cube

    for t in range(max_steps):
        if success_reached or t >= len(base_actions):
            action = np.zeros(action_dim, dtype=np.float32)
            action[-1] = last_gripper_action
        else:
            noise = rng.normal(0, action_noise_std, size=action_dim).astype(np.float32)
            action = np.clip(base_actions[t] + noise, -1.0, 1.0)
            last_gripper_action = action[-1]

        _, _, _, _, info = env.step(action)
        if bool(info.get("success", False)) and not success_reached:
            success_reached = True
            first_success_step = t

        frames.append(_render_frame(env))
        tcp_pos, cube_pos, goal_pos = _read_state(env)
        tcp_positions.append(tcp_pos)
        cube_positions.append(cube_pos)
        goal_positions.append(goal_pos)
        actions_taken.append(action)

    frames, state, actions_taken, velocities = _stack_episode(frames, tcp_positions, cube_positions, goal_positions, actions_taken)
    return frames, state, actions_taken, velocities, success_reached, first_success_step


def rollout_one_episode_train(env, base_actions, seed, max_steps, action_noise_std, rng, max_retry, success_dist_threshold=0.1):
    """Retries the train attempt up to max_retry times, keeping the closest-to-success one if none actually succeeds."""
    best, best_dist = None, np.inf
    for _ in range(max_retry):
        frames, state, actions_taken, velocities, success, first_success_step = _attempt_episode_train(
            env, base_actions, seed, max_steps, action_noise_std, rng,
        )
        final_dist = np.linalg.norm(state[-1, 3:6] - state[-1, 6:9])
        if success or final_dist < success_dist_threshold:
            return frames, state, actions_taken, velocities, first_success_step
        if final_dist < best_dist:
            best_dist = final_dist
            best = (frames, state, actions_taken, velocities, first_success_step)
    return best  # kept even though it never succeeded, training still benefits from the variation


def _attempt_episode_val(env, base_actions, seed, max_steps, action_noise_std, rng):
    """Runs one attempt only up to the first success, then stops simulating and pads by duplicating the last frame/state up to max_steps; returns None if success is never reached."""
    env.reset(seed=int(seed))
    frames, tcp_positions, cube_positions, goal_positions, actions_taken = [], [], [], [], []
    action_dim = env.action_space.shape[-1]
    success_reached = False
    first_success_step = None

    for t in range(min(max_steps, len(base_actions))):
        noise = rng.normal(0, action_noise_std, size=action_dim).astype(np.float32)
        action = np.clip(base_actions[t] + noise, -1.0, 1.0)
        _, _, _, _, info = env.step(action)

        frames.append(_render_frame(env))
        tcp_pos, cube_pos, goal_pos = _read_state(env)
        tcp_positions.append(tcp_pos)
        cube_positions.append(cube_pos)
        goal_positions.append(goal_pos)
        actions_taken.append(action)

        if bool(info.get("success", False)):
            success_reached = True
            first_success_step = t
            break  # stop simulating as soon as the task is solved

    if not success_reached:
        return None

    last_frame, last_tcp, last_cube, last_goal = frames[-1], tcp_positions[-1], cube_positions[-1], goal_positions[-1]
    last_gripper = actions_taken[-1][-1]
    while len(frames) < max_steps:  # pad the rest of the episode by duplicating the successful frame, never by simulating further
        frames.append(last_frame.copy())
        tcp_positions.append(last_tcp.copy())
        cube_positions.append(last_cube.copy())
        goal_positions.append(last_goal.copy())
        pad_action = np.zeros(action_dim, dtype=np.float32)
        pad_action[-1] = last_gripper
        actions_taken.append(pad_action)

    frames, state, actions_taken, velocities = _stack_episode(frames, tcp_positions, cube_positions, goal_positions, actions_taken)
    return frames, state, actions_taken, velocities, first_success_step


def rollout_one_episode_val(env, base_actions, seed, max_steps, action_noise_std, rng, max_retry):
    """Retries until an attempt actually succeeds; unlike train, a val episode that never succeeds is not usable."""
    for attempt in range(max_retry):
        result = _attempt_episode_val(env, base_actions, seed, max_steps, action_noise_std, rng)
        if result is not None:
            return (*result, attempt + 1)
    return None


def write_episode_video(path, frames, fps=10):
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_split_tensors(split_dir, states, actions, velocities, seq_lengths, seeds):
    split_dir = Path(split_dir)
    if len(states) == 0:
        return
    torch.save(torch.tensor(np.stack(states), dtype=torch.float32), split_dir / "states.pth")
    torch.save(torch.tensor(np.stack(actions), dtype=torch.float32), split_dir / "abs_actions.pth")
    torch.save(torch.tensor(np.stack(velocities), dtype=torch.float32), split_dir / "velocities.pth")
    with open(split_dir / "seq_lengths.pkl", "wb") as f:
        pickle.dump(list(seq_lengths), f)
    with open(split_dir / "seeds.pkl", "wb") as f:
        pickle.dump(list(seeds), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", type=str, required=True, help="trajectory .h5 produced by ManiSkill's motion-planning + replay step")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-train-rollouts", type=int, default=10000, help="train and val are independent pools, not a split of one")
    parser.add_argument("--num-val-rollouts", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=150, help="fixed length of every saved rollout")
    parser.add_argument("--action-noise-std", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--success-buffer", type=int, default=10, help="extra steps kept in train seq_lengths after first success, instead of the full max_steps")
    parser.add_argument("--max-retry-train", type=int, default=10, help="attempts per train episode before keeping the closest-to-success one anyway")
    parser.add_argument("--max-retry-val", type=int, default=50, help="attempts per val episode before giving up (val requires an actual success)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    demos = load_base_demos(args.demo_path)
    print(f"Loaded {len(demos)} base expert trajectories from {args.demo_path}")

    env = gym.make(
        "PushCube-v1",
        obs_mode="none",  # we render frames ourselves, no simulator obs needed
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        sim_backend="cpu",  # one env at a time, no need for the GPU-vectorized backend
        max_episode_steps=args.max_steps,
        human_render_camera_configs=dict(width=args.img_size, height=args.img_size),
    )

    out_dir = Path(args.out_dir)
    for split in ("train", "val"):
        (out_dir / split / "obses").mkdir(parents=True, exist_ok=True)

    # train: original behaviour, an episode may keep pushing without ever succeeding
    train_states, train_actions, train_vel, train_seq_lengths, train_seeds = [], [], [], [], []
    n_never_succeeded = 0
    for ep_idx in tqdm(range(args.num_train_rollouts), desc="rollout (train)"):
        base = demos[ep_idx % len(demos)]  # seed must be the demo's own: it fixes the cube/goal layout the recorded actions were planned for
        frames, state, actions, vel, first_success_step = rollout_one_episode_train(
            env, base["actions"], seed=base["seed"], max_steps=args.max_steps,
            action_noise_std=args.action_noise_std, rng=rng, max_retry=args.max_retry_train,
        )
        write_episode_video(out_dir / "train" / "obses" / f"episode_{ep_idx:03d}.mp4", frames)
        del frames

        if first_success_step is None:
            seq_len = args.max_steps
            n_never_succeeded += 1
        else:
            seq_len = min(first_success_step + args.success_buffer + 1, args.max_steps)  # trims the frozen post-success tail out of training windows

        train_states.append(state)
        train_actions.append(actions)
        train_vel.append(vel)
        train_seq_lengths.append(seq_len)
        train_seeds.append(int(base["seed"]))

    save_split_tensors(out_dir / "train", train_states, train_actions, train_vel, train_seq_lengths, train_seeds)
    avg_len = sum(train_seq_lengths) / len(train_seq_lengths)
    print(f"train: {len(train_states)} rollouts saved to {out_dir / 'train'} (avg length {avg_len:.1f}/{args.max_steps}, never succeeded: {n_never_succeeded})")

    # val: every episode must succeed and hold the goal for the full length
    val_states, val_actions, val_vel, val_seq_lengths, val_seeds = [], [], [], [], []
    retries_used = []
    for ep_idx in tqdm(range(args.num_val_rollouts), desc="rollout (val)"):
        base = demos[ep_idx % len(demos)]
        result = rollout_one_episode_val(
            env, base["actions"], seed=base["seed"], max_steps=args.max_steps,
            action_noise_std=args.action_noise_std, rng=rng, max_retry=args.max_retry_val,
        )
        if result is None:
            raise RuntimeError(f"val episode {ep_idx} never succeeded within {args.max_retry_val} attempts; raise --max-retry-val")
        frames, state, actions, vel, first_success_step, n_attempts = result
        write_episode_video(out_dir / "val" / "obses" / f"episode_{ep_idx:03d}.mp4", frames)
        del frames

        val_states.append(state)
        val_actions.append(actions)
        val_vel.append(vel)
        val_seq_lengths.append(args.max_steps)  # always the full length: the tail is a genuine held success, safe to use as a goal at any horizon
        val_seeds.append(int(base["seed"]))
        retries_used.append(n_attempts)

    env.close()

    save_split_tensors(out_dir / "val", val_states, val_actions, val_vel, val_seq_lengths, val_seeds)
    print(f"val: {len(val_states)} rollouts saved to {out_dir / 'val'} (attempts per episode: min={min(retries_used)}, mean={sum(retries_used) / len(retries_used):.1f}, max={max(retries_used)})")


if __name__ == "__main__":
    main()
