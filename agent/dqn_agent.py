import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        hidden_sizes=(128, 128),
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_net = DQN(state_dim, num_actions, hidden_sizes).to(self.device)
        self.target_q_net = DQN(state_dim, num_actions, hidden_sizes).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(state_dim, buffer_capacity)

        self.train_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train_step(self):
        if not self.replay_buffer.can_sample(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1

        return float(loss.item())
