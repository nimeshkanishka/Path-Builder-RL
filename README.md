# Path Builder: A Custom Gymnasium Environment for Reinforcement Learning

## Installation

1. Clone the repository

```bash
git clone https://github.com/nimeshkanishka/Path-Builder-RL.git
```

2. Install the package

```bash
cd Path-Builder-RL
pip install .
```

## Usage

The example below demonstrates how to create, interact with, and visualize the environment.

```python
import gymnasium as gym
import path_builder

env = gym.make("PathBuilder-v0", render_mode="human")

observation, info = env.reset()
total_steps = 0
total_reward = 0
done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_steps += 1
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished in {total_steps} steps with total reward {total_reward}")

env.close()
```

The example below demonstrates how to train an agent using Stable Baselines3's DQN algorithm.

```python
import gymnasium as gym
import path_builder
from stable_baselines3 import DQN

env = gym.make("PathBuilder-v0")

model = DQN(
    policy="MlpPolicy",
    env=env,
    buffer_size=250_000,
    exploration_fraction=0.5,
    exploration_final_eps=0.02
)

model.learn(total_timesteps=1_000_000)
model.save("path_builder_agent")
```

## Environment Details

### Description

The goal of the agent is to build a path and reach the target position (randomly generated) from the starting position (also randomly generated) in the minimum possible number of timesteps in a grid.

# Observation Space

The observation is a ```ndarray``` with shape ```(8,)``` consisting of the x and y coordinates of the agent's and target's positions and the status (```-1```: out of bounds, ```0```: not built, ```1```: built) of the 4 neighboring cells (up, down, left, right).

# Action Space

There are 4 discrete actions:

0: Move up

1: Move down

2: Move left

3: Move right

# Rewards

A reward of ```+1``` is given for moving a cell closer to the target, and a reward of ```-1``` is given for moving a cell further from the target. Additionally, penalties are given for moving to already built cells (```-0.5```) and trying to move out of bounds (```-2```). A reward of ```+25``` is given for reaching the target.

# Episode End

An episode is terminated when the agent reaches the target position.

An episode is truncated when the episode length reaches the maximum number of timesteps calculated at the start of the episode. ```episode_steps = int(distance_to_target * (1 + extra_timesteps_fraction))```

# Arguments

```python
env = gym.make("PathBuilder-v0", render_mode=None, grid_size=16, extra_timesteps_fraction=0.5)
```

```render_mode: str | None = None``` determines the render mode of the environment. ```render_mode="human"``` visualizes the environment using pygame.

```grid_size: int | tuple[int, int] = 16``` determines the size of the grid. If a ```int``` is provided, grid size will be ```(grid_size, grid_size)```. If a ```tuple``` is provided, it should be ```(grid_width, grid_height)```.

```extra_timesteps_fraction: float = 0.5``` determines the fraction of extra timesteps available to the agent. Must be non-negative.

# Version History

v0: Initial version