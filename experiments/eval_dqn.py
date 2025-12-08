import os
import sys
import numpy as np
import torch
import time

# --- Make sure we can import from project root ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from env.villain_env import VillainAkinatorEnv
from agent.dqn import DQN


def evaluate_dqn(
    csv_path="data/disney_villains_akinator_Dataset.csv",
    model_path="villain_dqn_shaped.pth",
    max_questions=10,
    episodes=50,
    use_main_villains_only=True,
):
    start_time = time.time()

    # Create environment
    env = VillainAkinatorEnv(
        csv_path=csv_path,
        max_questions=max_questions,
        use_main_villains_only=use_main_villains_only,
        min_questions_before_guess=3
    )

    state_dim = env.state_dim
    num_actions = env.action_dim

    print("Env info:")
    print(f"  Num villains:   {env.num_villains}")
    print(f"  Num questions:  {env.num_questions}")
    print(f"  Num actions:    {num_actions}")
    print(f"  Max questions per episode: {max_questions}")
    print(f"  Episodes to evaluate:      {episodes}")
    print("-" * 40)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DQN model and load trained weights
    policy_net = DQN(state_dim, num_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    total_return = 0.0
    total_correct = 0
    total_questions = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_return = 0.0
        ep_questions = 0
        step_count = 0
        max_steps_per_episode = 50  # generous upper bound

        while not done and step_count < max_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done, info = env.step(action)

            ep_return += reward
            if action < env.num_questions:
                ep_questions += 1

            state = next_state
            step_count += 1

        # if it hit the step cap without done=True, we can mark it as a failure episode
        if not done:
            info["correct_guess"] = False

        total_return += ep_return
        total_questions += ep_questions

        if info.get("correct_guess", False):
            total_correct += 1
            #Print a special celebration when it gets one right
            print("\n🎉 CORRECT GUESS! 🎉")
            print(f"  Episode: {ep}")
            print(f"  Secret villain:  {info.get('secret_villain')}")
            print(f"  Guessed villain: {info.get('guessed_villain')}")
            print(f"  Questions asked: {ep_questions}")
            print(f"  Episode return:  {ep_return:.2f}\n")

        if ep % 10 == 0 or ep == 1:
            print(
                f"Episode {ep}/{episodes} done: "
                f"return={ep_return:.2f}, "
                f"questions={ep_questions}, "
                f"correct={info.get('correct_guess', False)}"
            )


    avg_return = total_return / episodes
    success_rate = total_correct / episodes
    avg_questions = total_questions / episodes
    elapsed = time.time() - start_time

    print("\n=== Evaluation Results ===")
    print(f"Episodes:              {episodes}")
    print(f"Average return:        {avg_return:.2f}")
    print(f"Success rate:          {success_rate * 100:.2f}%")
    print(f"Average # questions:   {avg_questions:.2f}")
    print(f"Total eval time:       {elapsed:.2f} seconds")


if __name__ == "__main__":
    evaluate_dqn()