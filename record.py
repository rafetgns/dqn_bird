import os
import gymnasium as gym
import torch
import imageio
import numpy as np
from src.dqn import DQN
import flappy_bird_gymnasium

def record_model(model_path, filename, num_episodes=10, fps=60):
    """
    records num_episodes of gameplay and saves them as a single mp4 in assets/.
    fps=60 speeds up playback (original game runs at ~30fps).
    """
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

    num_states  = env.observation_space.shape[0]
    num_actions = env.action_space.n
    policy = DQN(num_states, num_actions, 256, 256)  # match your YAML
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    all_frames   = []
    all_rewards  = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated  = False
        episode_reward = 0
        episode_frames = []

        while not (terminated or truncated):
            episode_frames.append(env.render())
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                action = policy(state_tensor).squeeze().argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        all_rewards.append(episode_reward)
        all_frames.extend(episode_frames)
        print(f"  Episode {episode+1:>2}/{num_episodes} — reward: {episode_reward:.1f} — frames: {len(episode_frames)}")

    env.close()

    
    os.makedirs("assets", exist_ok=True)
    
    
    mp4_filename = filename.replace(".gif", ".mp4")
    
    
    save_path = os.path.join("assets", mp4_filename)

   
    imageio.mimwrite(save_path, all_frames, fps=fps, codec="libx264")

    print(f"\n✓ Saved '{save_path}'")
    print(f"  Episodes : {num_episodes}")
    print(f"  Avg reward: {np.mean(all_rewards):.1f}")
    print(f"  Best reward: {max(all_rewards):.1f}")
    print(f"  Total frames: {len(all_frames)} @ {fps}fps\n")


print("=== Stage 1: Initial model (untrained) ===")
record_model("runs/flappybird1_init.pt", "stage1_init.mp4", num_episodes=10, fps=60)

print("=== Stage 2: Mid-training model ===")
record_model("runs/flappybird1_mid.pt", "stage2_mid.mp4", num_episodes=10, fps=60)

print("=== Stage 3: Best model ===")
record_model("runs/flappybird1.pt",     "stage3_best.mp4", num_episodes=10, fps=60)