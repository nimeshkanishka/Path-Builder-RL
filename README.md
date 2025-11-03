# Path Builder: A Custom Gymnasium Environment for Reinforcement Learning

The goal of the agent is to build a path and reach the target position (randomly generated) from the starting position (also randomly generated) in the minimum possible number of timesteps in a grid world. The agent is rewarded for getting closer to the target.

# Observation Space

The observation is a ndarray with shape (8,) consisting of the x and y coordinates of the agent's and target's positions and the status of 4 neighboring cells (up, down, left, right).

# Action Space

There are 4 discrete actions:

0: Move up

1: Move down

2: Move left

3: Move right

# Episode End

Termination: When the agent reaches the target position.

Truncation: When the episode length reaches the maximum number of timesteps calculated at the start of the episode. (This is distance_to_target * (1 + extra_timesteps_fraction))

# Arguments

grid_size: int | tuple[int, int] = 16 determines the size of the grid.

extra_timesteps_fraction: float = 0.5 determines the fraction of extra timesteps available to the agent.

# Version History

v0: Initial version