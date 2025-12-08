import numpy as np
import torch
import os
import sys
import json

# Make sure we can import from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from agent.dqn_agent import DQNAgent
from env.villain_env import VillainAkinatorEnv


def train_dqn(
    csv_path="data/disney_villains_akinator_Dataset.csv",
    max_questions=10,
    episodes=3000,
    start_eps=1.0,
    end_eps=0.05,
    decay_episodes=300,
    target_update_freq=500,
    use_reward_shaping=True,
    tag="shaped",
):
    """
    Train a DQN agent on the Disney Villain Akinator environment.

    use_reward_shaping:
        True  -> uses candidate-reduction bonus
        False -> plain DQN with only base rewards

    tag:
        Used to distinguish output files, e.g. "baseline", "shaped"
    """

    env = VillainAkinatorEnv(
        csv_path=csv_path,
        max_questions=max_questions,
        use_main_villains_only=True,
        min_questions_before_guess=3,
        use_reward_shaping=use_reward_shaping,
    )

    print("TRAINING ENV INFO:")
    print("  Num villains:", env.num_villains)
    print("  Num questions:", env.num_questions)
    print("  Num actions:", env.num_actions)

    state_dim = 2 * env.num_questions
    num_actions = env.num_actions

    agent = DQNAgent(state_dim, num_actions, lr=1e-3)

    # For logging training performance
    returns = []

    def epsilon(ep):
        # Linear decay from start_eps to end_eps over 'decay_episodes'
        if ep >= decay_episodes:
            return end_eps
        return start_eps - (ep / decay_episodes) * (start_eps - end_eps)

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            eps = epsilon(ep)
            action = agent.select_action(state, eps)

            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if agent.train_steps % target_update_freq == 0:
                agent.update_target_network()

        returns.append(total_reward)

        if ep % 100 == 0:
            avg = np.mean(returns[-100:])
            print(
                f"[{tag}] Episode {ep}/{episodes}, "
                f"avg return (last 100) = {avg:.2f}"
            )

    model_path = f"villain_dqn_{tag}.pth"
    torch.save(agent.q_net.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    returns_path = f"training_returns_{tag}.json"
    with open(returns_path, "w") as f:
        json.dump(returns, f)
    print(f"Saved training returns to {returns_path}")


if __name__ == "__main__":
    train_dqn(use_reward_shaping=True, tag="shaped")

    train_dqn(use_reward_shaping=False, tag="baseline")

