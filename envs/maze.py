from envs.collections.grid import GridEnv


class SmallMazeEnv(GridEnv):

    def __init__(self):
        grid = [
            ["C", "C", "S", "C"],
            ["C", "C", "W", "C"],
            ["C", "C", "C", "C"],
            ["C", "W", "W", "W"],
            ["C", "C", "C", "G"]
        ]

        rewards = {
            "C": 0,
            "S": 0,
            "G": 1
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {0: 1.0},
            1: {1: 1.0},
            2: {2: 1.0},
            3: {3: 1.0}
        }

        terminals = [
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)


class GridMazeEnv(GridEnv):

    def __init__(self):
        grid = [
            ["C", "C", "C", "S"],
            ["C", "C", "W", "C"],
            ["C", "C", "C", "C"],
            ["C", "W", "W", "W"],
            ["C", "C", "C", "G"]
        ]

        rewards = {
            "C": 0,
            "S": 0,
            "G": 1
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {0: 1.0},
            1: {1: 1.0},
            2: {2: 1.0},
            3: {3: 1.0}
        }

        terminals = [
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)


class BlockedMazeEnv(GridEnv):

    def __init__(self):
        grid = [
            ["C", "C", "S", "C"],
            ["C", "C", "W", "C"],
            ["C", "C", "C", "C"],
            ["C", "C", "W", "W"],
            ["C", "C", "W", "G"]
        ]

        rewards = {
            "C": 0,
            "S": 0,
            "G": 1
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {0: 1.0},
            1: {1: 1.0},
            2: {2: 1.0},
            3: {3: 1.0}
        }

        terminals = [
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)
