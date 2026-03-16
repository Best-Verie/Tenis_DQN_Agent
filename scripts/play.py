#!/usr/bin/env python
"""
play.py — Load the best trained DQN model and run it on ALE/Tennis-v5.

Usage (local):
    python scripts/play.py --model models/best_dqn_tennis_cnn.zip

Usage (Colab / headless — saves a video):
    python scripts/play.py --model /content/drive/MyDrive/DQN_Tennis/student/models/best_dqn_tennis_cnn.zip \
                           --headless --video-dir videos/
"""

import argparse
import os
import sys
import time

import ale_py  # noqa: F401 — registers ALE namespace
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def parse_args():
    p = argparse.ArgumentParser(description="Play Atari with a saved DQN model.")
    p.add_argument("--model",      type=str, required=True,
                   help="Path to the saved .zip model (e.g. models/best_dqn_tennis_cnn.zip)")
    p.add_argument("--env-id",     type=str, default="ALE/Tennis-v5",
                   help="Gymnasium environment ID (default: ALE/Tennis-v5)")
    p.add_argument("--episodes",   type=int, default=3,
                   help="Number of episodes to play (default: 3)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--headless",   action="store_true",
                   help="Headless mode: record video instead of opening a window (use for Colab/servers)")
    p.add_argument("--video-dir",  type=str, default="videos/best_play",
                   help="Directory to save recorded videos (headless mode only)")
    p.add_argument("--fps",        type=float, default=60.0,
                   help="Playback speed in frames per second (local mode only)")
    return p.parse_args()


def play_local(model_path: str, env_id: str, n_episodes: int, seed: int, fps: float):
    """
    Render directly to a GUI window.
    Uses GreedyQPolicy: deterministic=True makes model.predict select argmax Q(s,a).
    """
    # CNN env requires VecFrameStack (4-frame stack) to match training preprocessing
    env = make_atari_env(env_id, n_envs=1, seed=seed, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load(
        model_path,
        env=env,
        custom_objects={"buffer_size": 200, "learning_starts": 0},
    )

    print(f"\nPlaying {n_episodes} episode(s) — close the window to exit.\n")
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # deterministic=True → greedy Q-value policy (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])
            done = bool(dones[0])
            time.sleep(1.0 / fps)
        print(f"  Episode {ep + 1:2d}  return: {ep_reward:.2f}")

    env.close()


def play_headless(model_path: str, env_id: str, n_episodes: int, seed: int, video_dir: str):
    """
    Headless mode: record episodes with RecordVideo, no GUI window.
    Suitable for Colab or any display-less environment.
    """
    os.makedirs(video_dir, exist_ok=True)

    # Use the raw gym env (not VecEnv) so RecordVideo can wrap it properly
    raw_env = gym.make(env_id, render_mode="rgb_array")
    raw_env.reset(seed=seed)
    record_env = RecordVideo(
        raw_env,
        video_folder=video_dir,
        episode_trigger=lambda ep_id: True,
        name_prefix="play",
    )

    model = DQN.load(
        model_path,
        custom_objects={"buffer_size": 200, "learning_starts": 0},
    )

    print(f"\nRecording {n_episodes} episode(s) to: {video_dir}\n")
    for ep in range(n_episodes):
        obs, _ = record_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # Greedy Q-policy: always pick the action with the highest Q-value
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = record_env.step(int(action))
            ep_reward += float(reward)
            done = terminated or truncated
        print(f"  Episode {ep + 1:2d}  return: {ep_reward:.2f}")

    record_env.close()

    # List saved videos
    videos = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    if videos:
        print("\nSaved videos:")
        for v in videos:
            print(f"  {os.path.join(video_dir, v)}")
    else:
        print("No .mp4 files found — check that 'moviepy' is installed.")


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Model   : {args.model}")
    print(f"Env     : {args.env_id}")
    print(f"Episodes: {args.episodes}")
    print(f"Mode    : {'headless (video)' if args.headless else 'GUI'}")

    if args.headless:
        play_headless(
            model_path=args.model,
            env_id=args.env_id,
            n_episodes=args.episodes,
            seed=args.seed,
            video_dir=args.video_dir,
        )
    else:
        play_local(
            model_path=args.model,
            env_id=args.env_id,
            n_episodes=args.episodes,
            seed=args.seed,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
