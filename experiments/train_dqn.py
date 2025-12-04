import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from env.villain_env import VillainAkinatorEnv  # adjust this if your env file has a different name


def train_dqn(
    csv_path="Data/villains.csv",
    max_questions=10,
    episodes=3000,
    start_eps=1.0,
    end_eps=0.05,
    decay_episodes=1500,
    target_update_freq=500,
):
    env = VillainAkinatorEnv(csv_path, max_questions)

    state_dim = 2 * env.num_questions
    num_actions = env.num_actions

    agent = DQNAgent(state_dim, num_actions, lr=1e-3)

    def epsilon(ep):
        if ep >= decay_episodes:
            return end_eps
        return start_eps - (ep / decay_episodes) * (start_eps - end_eps)

    returns = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

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
            print(f"Episode {ep}/{episodes}, avg return {avg:.2f}")

    torch.save(agent.q_net.state_dict(), "villain_dqn.pth")
    print("Saved model to villain_dqn.pth")


if __name__ == "__main__":
    train_dqn()
