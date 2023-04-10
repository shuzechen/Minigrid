from __future__ import annotations

import math

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces


class LocalOPTEnv(MiniGridEnv):

    """
    ## Description

    Maps for LocalOPT test

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-LocalOPT-v0`

    """

    def __init__(
        self,
        width=11,
        height=7,
        agent_start_pos=(3, 1),
        agent_start_dir=1,
        max_steps: int | None = None,
        map_size = "small",
        reward_type = "sparse",
        **kwargs,
    ):
        self.size = map_size
        self.reward_type = reward_type
        if self.size == "small": # OPT = 4.19
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
            self.goal_pos = (width - 4, 1)
        elif self.size == "medium":
            width = 18
            height = 12
            self.agent_start_pos = (5, 2)
            self.agent_start_dir = 1
            self.goal_pos = (width - 6, 2)
        elif self.size == "large":
            width = 25
            height = 16
            self.agent_start_pos = (6, 3)
            self.agent_start_dir = 1
            self.goal_pos = (width - 7, 3)
        elif self.size == "mini":
            width = 5
            height = 5
            self.agent_start_pos = (1, 1)
            self.agent_start_dir = 1
            self.goal_pos = (3, 1)
        # OPT = 2.8229751097999998

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * width * height

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(width, height),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Box(low=-1, high=5, shape=()),
                "position": spaces.Box(low=-1.0, high=100.0, shape=(2,))
                # "mission": mission_space,
            }
        )
        self._grid_type = np.zeros((11,), dtype=np.int8)
        self._grid_type[1] = 1
        self._grid_type[2] = 2
        self._grid_type[8] = 3
        self._grid_type[10] = 4

    def gen_obs(self):
        """
        Generate the agent's view (fully observable, low-resolution encoding)
        """
         
        # Get the fully observable states
        grid = self.grid

        # Encode the fully observable view into a numpy array
        image = grid.encode(None)
        image[self.agent_pos[0], self.agent_pos[1]] = 10 # Agent idx
        image = self._grid_type[image[..., 0]]

        direction = np.array(self.agent_dir)
        position = np.array(self.agent_pos)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": direction, "position": position} # , "mission": self.mission}

        return obs

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        if self.size == "small":
            # Place the wall
            for i in range(width-6):
                self.grid.set(i+3, 3, Wall())
            for i in range(height-5):
                self.grid.set(width//2, i+1, Wall())
        elif self.size == "medium":
            # Place the wall
            mid = width // 2 # 9
            #mid+1, mid-2, mid-5, mid+4
            for i in range(4):
                self.grid.set(mid+1, i+1, Wall())
                self.grid.set(mid-2, i+1, Wall())
                self.grid.set(mid-5, i+4, Wall())
                self.grid.set(mid+4, i+4, Wall())
            for i in range(2):
                self.grid.set(mid-4+i, 4, Wall())
                self.grid.set(mid+2+i, 4, Wall())
            for i in range(8):
                self.grid.set(mid-4+i, 7, Wall())
        elif self.size == "large":
            # Place the wall
            mid = width // 2 # 12
            #mid+1, mid-2, mid-5, mid+4
            for i in range(6):
                self.grid.set(mid+3, i+1, Wall())
                self.grid.set(mid-3, i+1, Wall())
            for i in range(3):
                self.grid.set(mid-8, i+7, Wall())
                self.grid.set(mid+8, i+7, Wall())
            for i in range(5):
                self.grid.set(mid-8+i, 6, Wall())
                self.grid.set(mid+4+i, 6, Wall())
            for i in range(17):
                self.grid.set(mid-8+i, 10, Wall())
        elif self.size == "mini":
            # Place the wall
            self.grid.set(2, 1, Wall())
            self.grid.set(2, 2, Wall())


            # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"
    
    def _reward(self) -> float:
        return 1
    
    def step(self, action):
        if self.reward_type == "sparse":
            obs, reward, done, info = super().step(action)
        elif self.reward_type == "dense":
            goal_posx, goal_posy = self.goal_pos
            cur_x, cur_y = self.agent_pos
            cur_dis = abs(cur_x - goal_posx) + abs(cur_y - goal_posy)
            obs, reward, done, info = super().step(action)
            nxt_x, nxt_y = self.agent_pos
            nxt_dis = abs(nxt_x - goal_posx) + abs(nxt_y - goal_posy)
            reward += cur_dis - nxt_dis
        elif self.reward_type == "dense-L2":
            goal_posx, goal_posy = self.goal_pos
            cur_x, cur_y = self.agent_pos
            cur_dis = math.sqrt((cur_x - goal_posx)**2 + (cur_y - goal_posy)**2)
            obs, reward, done, info = super().step(action)
            nxt_x, nxt_y = self.agent_pos
            nxt_dis = math.sqrt((nxt_x - goal_posx)**2 + (nxt_y - goal_posy)**2)
            reward += cur_dis - nxt_dis

        return obs, reward, done, info               
