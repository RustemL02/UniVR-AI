from gym.envs.registration import register

from envs.cliff import *
from envs.lava  import *
from envs.maze  import *


# Constants for the environment names.
SMALL_MAZE          = "SmallMaze-v0"
GRID_MAZE           = "GridMaze-v0"
BLOCKED_MAZE        = "BlockedMaze-v0"
LAVA_FLOOR          = "LavaFloor-v0"
VERY_BAD_LAVA_FLOOR = "VeryBadLavaFloor-v0"
NICE_LAVA_FLOOR     = "NiceLavaFloor-v0"
BIGGER_LAVA_FLOOR   = "BiggerLavaFloor-v0"
HUGE_LAVA_FLOOR     = "HugeLavaFloor-v0"
CLIFF               = "Cliff-v0"


# Maze environments.
register(
    id          = SMALL_MAZE,
    entry_point = "envs:SmallMazeEnv"
)
register(
    id          = GRID_MAZE,
    entry_point = "envs:GridMazeEnv"
)
register(
    id          = BLOCKED_MAZE,
    entry_point = "envs:BlockedMazeEnv"
)


# Lava environments.
register(
    id          = LAVA_FLOOR,
    entry_point = "envs:LavaFloorEnv"
)
register(
    id          = VERY_BAD_LAVA_FLOOR,
    entry_point = "envs:VeryBadLavaFloorEnv"
)
register(
    id          = NICE_LAVA_FLOOR,
    entry_point = "envs:NiceLavaFloorEnv"
)
register(
    id          = BIGGER_LAVA_FLOOR,
    entry_point = "envs:BiggerLavaFloorEnv"
)
register(
    id          = HUGE_LAVA_FLOOR,
    entry_point = "envs:HugeLavaFloorEnv"
)


# Cliff environments.
register(
    id          = CLIFF,
    entry_point = "envs:CliffEnv"
)
