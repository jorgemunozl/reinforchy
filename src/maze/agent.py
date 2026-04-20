"""
Agent for the maze game.

We are going to make an agent move in a straigh line.
It should move left or right.

The goal is that is stays in the middle of the line.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass


def action_meaning(action: int) -> str:
    """
    Get the meaning of an action.
    """
    return ACTION_MEANING[action]



@dataclass
class MazeAgentConfig:
    state_size: int = 1
    action_size: int = 2
    hidden_size: int = 128
    learning_rate: float = 0.001
    batch_size: int = 2
    gamma: float = 0.99
    epsilon: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


ACTION_MEANING = {
    0: "left",
    1: "rigth",
    2: "left",
    3: "right",
}


class MazeAgent(nn.Module):
    def __init__(self, config: MazeAgentConfig):
        super(MazeAgent, self).__init__()
        self.fc1 = nn.Linear(config.state_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.action_size)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

config = MazeAgentConfig()
agent = MazeAgent(config).to(config.device)

state = 0

# Here you can directly backpropagate the error in the state
optimizer = optim.SGD(agent.parameters(), lr=0.01)
for _ in range(100):
    optimizer.zero_grad()
    action_probs = agent(torch.tensor([[state]], device=config.device, dtype=torch.float32))
    loss = F.mse_loss(action_probs.argmax(dim=-1), torch.tensor([0.]))  
    loss.backward()
    optimizer.step()

    print("loss:", loss.item())

    print(action_probs, end=" ")

    discrete_action = action_probs.argmax(dim=-1).item()
    action_for_state = discrete_action * 2 - 1

    state += action_for_state

    print(action_meaning(discrete_action), end=" ")
    print(f"state: {state}")
    time.sleep(0.01)


