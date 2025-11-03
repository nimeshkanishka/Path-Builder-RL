import gymnasium as gym
import numpy as np
import pygame

class PathBuilderEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        render_mode: str | None = None,
        grid_size: int | tuple[int, int] = 16,
        extra_timesteps_fraction: float = 0.5
    ):
        self.render_mode = render_mode

        # Grid size
        # A positive integer (square grid) or a tuple of two positive integers (rectangular grid)
        # Shape: (Width, Height)
        if isinstance(grid_size, int):
            if grid_size <= 0:
                raise ValueError("grid_size must be positive.")
            self.grid_size = (grid_size, grid_size)
        elif isinstance(grid_size, tuple):
            if len(grid_size) != 2:
                raise ValueError("grid_size must be an int or a tuple of two ints.")
            if any(not isinstance(x, int) for x in grid_size):
                raise TypeError("grid_size values must be integers.")
            if any(x <= 0 for x in grid_size):
                raise ValueError("grid_size values must be positive.")
            self.grid_size = grid_size
        else:
            raise TypeError("grid_size must be an int or a tuple of two ints.")
        
        w, h = self.grid_size

        if w + h < 4:
            raise ValueError("Grid is too small for training. Please use a larger grid_size.")                
        
        # Minimum distance required between start and target positions in each episode
        # This is half of the maximum possible Manhattan distance in the grid
        self.min_distance = ((w - 1) + (h - 1)) // 2
        
        # Fraction of extra timesteps available to the agent (>=0)
        try:
            extra_timesteps_fraction = float(extra_timesteps_fraction)
        except:
            raise TypeError("extra_timesteps_fraction must be of numeric type.")
        if extra_timesteps_fraction < 0:
            raise ValueError("extra_timesteps_fraction must be non-negative.")
        self.extra_timesteps_fraction = extra_timesteps_fraction

        # Pygame variables
        self.screen = None
        self.clock = None
        self.cell_size = None
        self.screen_size = None
        
        # Observation space
        # Contains x and y coordinates of agent and target positions and
        # status of 4 neighboring cells (up, down, left, right)
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=max(w, h) - 1,
            shape=(8,),
            dtype=np.int8
        )

        # Action space
        # Contains 4 discrete actions: move up, down, left, right
        self.action_space = gym.spaces.Discrete(4)

        # Numpy random generator
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)        

        # Initialize grid with all zeros (no built paths)
        # Shape: (Width, Height)
        self.grid = np.zeros(shape=self.grid_size, dtype=np.uint8)
        w, h = self.grid_size

        # Random starting position for the agent
        self.agent_position = (
            self.np_random.integers(0, w),
            self.np_random.integers(0, h)
        )
        # Mark starting position as built
        self.grid[self.agent_position] = 1
        
        # Find all valid target positions
        valid_pos = [
            (x, y)
            for x in range(w)
            for y in range(h)
            if self._calculate_manhattan_distance(self.agent_position, (x, y)) >= self.min_distance
        ]
        # Randomly select a target position from the valid positions
        self.target_position = valid_pos[self.np_random.integers(0, len(valid_pos))]
        
        # Distance to target from agent's starting position
        distance_to_target = self._calculate_manhattan_distance(self.agent_position, self.target_position)
        # Total timesteps in the episode
        self.episode_steps = int(distance_to_target * (1 + self.extra_timesteps_fraction))

        self.render()
        
        return self._get_observation(), {}

    def step(self, action):
        if action < 0 or action > 3:
            raise ValueError(f"Invalid action {action}. Action must be an integer in [0, 3].")
        
        reward = 0.0
        terminated = False
        truncated = False

        # Calculate distance to target before agent moves
        distance_before = self._calculate_manhattan_distance(self.agent_position, self.target_position)

        # Offsets of neighboring cells (up, down, left, right)
        neighbor_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # Calculate new position (x and y coordinates) of the agent
        new_position = (
            self.agent_position[0] + neighbor_offsets[action][0],
            self.agent_position[1] + neighbor_offsets[action][1]
        )

        # If new position is within grid bounds, update agent position
        if (0 <= new_position[0] < self.grid_size[0]) and (0 <= new_position[1] < self.grid_size[1]):
            self.agent_position = new_position

            # If agent has moved to an unbuilt cell, mark cell as built
            if self.grid[self.agent_position] == 0:
                self.grid[self.agent_position] = 1

            # Penalize for moving to already built cell
            else:
                reward -= 0.5

        # Penalize for trying to move out of bounds
        else:
            reward -= 2

        # Calculate distance to target after agent moves
        distance_after = self._calculate_manhattan_distance(self.agent_position, self.target_position)

        # Reward for reducing distance to target
        # +1 for moving a cell closer to target
        # 0 for no change of distance to target
        # -1 for moving a cell further from target
        reward += distance_before - distance_after

        # Episode termniates when agent reaches target position
        if self.agent_position == self.target_position:
            # Large reward for reaching the target
            reward += 25
            terminated = True

        # Episode truncates when maximum timesteps are used up
        self.episode_steps -= 1
        if self.episode_steps <= 0:
            truncated = True
            
        self.render()

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return
        
        self._render_human()

    def _calculate_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        # Agent's x and y coordinates
        ax, ay = self.agent_position
        # Target's x and y coordinates
        tx, ty = self.target_position
        # Width and height of grid
        w, h = self.grid_size

        obs = [ax, ay, tx, ty]

        # Offsets of neighboring cells (up, down, left, right)
        neighbor_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # Status of neighboring cells: 0 (not built), 1 (built), -1 (out of bounds)
        for dx, dy in neighbor_offsets:
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < w and 0 <= ny < h:
                obs.append(self.grid[(nx, ny)])
            else:
                obs.append(-1)

        return np.array(obs, dtype=np.int8)

    def _render_human(self):
        # Initialize pygame once
        if self.screen is None:
            pygame.init()

            # Determine cell size based on grid size
            if max(self.grid_size) <= 8:
                self.cell_size = 64
            elif max(self.grid_size) <= 16:
                self.cell_size = 32
            elif max(self.grid_size) <= 32:
                self.cell_size = 16
            elif max(self.grid_size) <= 64:
                self.cell_size = 8
            # If grid is too large, disable rendering
            else:
                print(f"Grid size of {self.grid_size} is too large for visualization. Disabling rendering...")
                self.render_mode = None
                return
            
            # Calculate screen size
            self.screen_size = (
                self.grid_size[0] * self.cell_size,
                self.grid_size[1] * self.cell_size
            )

            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Path Builder")

            self.clock = pygame.time.Clock()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Black background (unbuilt cells)
        self.screen.fill((0, 0, 0))

        # Draw built cells in white
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.grid[(x, y)] == 1:
                    pygame.draw.rect(
                        self.screen,
                        (255, 255, 255),
                        pygame.Rect(
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        )
                    )

        # Draw grid lines in gray
        screen_x, screen_y = self.screen_size
        color = (225, 225, 225)
        # Vertical lines
        for x in range(0, screen_x + 1, self.cell_size):
            pygame.draw.line(self.screen, color, (x, 0), (x, screen_y))
        # Horizontal lines
        for y in range(0, screen_y + 1, self.cell_size):
            pygame.draw.line(self.screen, color, (0, y), (screen_x, y))
        
        # Draw target position as gold circle
        target_x, target_y = self.target_position
        center = (
            target_x * self.cell_size + self.cell_size // 2,
            target_y * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(
            self.screen,
            (255, 215, 0),
            center,
            self.cell_size // 4
        )

        # Draw agent as red circle        
        agent_x, agent_y = self.agent_position
        center = (
            agent_x * self.cell_size + self.cell_size // 2,
            agent_y * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            center,
            self.cell_size // 3
        )

        pygame.display.flip()
        self.clock.tick(self.metadata.get("render_fps", 5))