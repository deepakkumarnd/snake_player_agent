import os.path
import random
import numpy as np
import torch
from torch import nn, optim
import json


class SnakeBoardStates:
    def __init__(self):
        self.states = []

    def add(self, state, action, reward, next_state, game_over):
        self.states.append((state, action, reward, next_state, game_over))

    def sample(self, sample_size: int):
        return random.sample(self.states, sample_size)

    def clear(self):
        self.states = []

    def write_to_disk(self):
        lines = []

        for (state, action, reward, next_state, done) in self.states:
            obj = {
                "state": state.numpy().tolist(),
                "action": action,
                "reward": reward,
                "next_state": next_state.numpy().tolist(),
                "done": done
            }

            lines.append(json.dumps(obj) + "\n")

        with open("history.txt", "a+") as f:
            f.writelines(lines)

    def __len__(self):
        return len(self.states)


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 5e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift


class SnakePlayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_layer = nn.Linear(64, 512)
        hidden_layer = nn.Linear(512, 512)
        output_layer = nn.Linear(512, 4)
        self.network = nn.Sequential(
            input_layer,
            nn.ReLU(),
            hidden_layer,
            LayerNormalization(512),
            nn.ReLU(),
            output_layer
        )

    def forward(self, x):
        return self.network(x)


class Agent:
    def __init__(self):
        print("Agent initialized")
        self.batch_size = 16
        self.model = SnakePlayerModel()
        self.model.eval()
        self.policy_model = SnakePlayerModel()
        self.policy_model.train()
        self.history = SnakeBoardStates()
        self.gamma = 0.9
        self.loss_f = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=0.1)
        self.games = 0
        self.loss_avg = None
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.01
        self.moves = 0

        if os.path.isfile("snake.pt"):
            print("Loading model")
            checkpoint = torch.load("snake.pt")
            self.policy_model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.games = checkpoint.get("games", 0)
            self.moves = checkpoint.get("moves", 0)
            self.loss_avg = checkpoint.get("loss_avg", 100000.0)

    def predict(self, grid) -> int:
        explore = random.random()

        if explore < self.epsilon:
            print("Exploring")
            action = random.randint(0, 3)
        else:
            print("Predicting")
            x = self._grid_to_vector(grid)
            output = self.model(x).squeeze(1)
            action = torch.argmax(output).item()

        self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self.epsilon_decay))
        return action

    def feedback(self, grid, action, reward, next_grid, game_over):
        print("Move =", action, "Reward =", reward, "Game over=", game_over)
        current_state = self._grid_to_vector(grid)
        next_state = self._grid_to_vector(next_grid)
        self.history.add(current_state, action, reward, next_state, game_over)
        self.moves = self.moves + 1

        if game_over:
            self.games = self.games + 1

        if len(self.history) > 1000:
            print("Writing history to disk")
            self.history.write_to_disk()
            self.history.clear()
        elif (len(self.history) % self.batch_size) == 0:
            print("Start training history len=", len(self.history))
            self.train()

    def train(self):
        losses = []
        for _ in range(1000):
            states, actions, rewards, next_states, game_over = zip(*self.history.sample(self.batch_size))
            states = torch.cat(states)
            next_states = torch.cat(next_states)
            dones = (torch.tensor(game_over) * 1).unsqueeze(1)
            outputs = self.policy_model(states)
            # predicted Q values for the actions
            q_values = torch.gather(outputs, dim=1, index=torch.tensor(actions).unsqueeze(1))
            # predicted Q values for the next_states
            values, _indices = self.policy_model(next_states).max(dim=1)
            max_q_values = values.unsqueeze(1)
            fut_q_values = torch.tensor(rewards).unsqueeze(1) + self.gamma * max_q_values * (1 - dones)
            loss = self.loss_f(q_values.squeeze(1), fut_q_values.squeeze(1))
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        self.loss_avg = sum(losses) / len(losses)
        self.model.load_state_dict(self.policy_model.state_dict())
        print("Saving model")
        torch.save({
            "games": self.games,
            "moves": self.moves,
            "loss_avg": self.loss_avg,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, 'snake.pt')

    def _grid_to_vector(self, grid):
        flat_array = np.array(grid).flatten()
        return torch.tensor(flat_array).float().unsqueeze(dim=0)
