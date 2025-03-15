from envs.collections.grid import GridEnv


class LavaFloorEnv(GridEnv):

    def __init__(self):
        grid = [
            ["S", "L", "L", "L"],
            ["L", "W", "L", "P"],
            ["L", "L", "L", "G"]
        ]

        rewards = {
            "L": -0.04,
            "S": -0.04,
            "P": -5.0,
            "G":  1.0
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {
                0: 0.8,
                1: 0.0,
                2: 0.1,
                3: 0.1
            },
            1: {
                0: 0.0,
                1: 0.8,
                2: 0.1,
                3: 0.1
            },
            2: {
                0: 0.1,
                1: 0.1,
                2: 0.8,
                3: 0.0
            },
            3: {
                0: 0.1,
                1: 0.1,
                2: 0.0,
                3: 0.8
            }
        }

        terminals = [
            "P",
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)


class VeryBadLavaFloorEnv(GridEnv):

    def __init__(self):
        grid = [
            ["S", "L", "L", "L"],
            ["L", "W", "L", "P"],
            ["L", "L", "L", "G"]
        ]

        rewards = {
            "L": -5.0,
            "S": -5.0,
            "P": -5.0,
            "G":  1.0
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {
                0: 0.8,
                1: 0.0,
                2: 0.1,
                3: 0.1
            },
            1: {
                0: 0.0,
                1: 0.8,
                2: 0.1,
                3: 0.1
            },
            2: {
                0: 0.1,
                1: 0.1,
                2: 0.8,
                3: 0.0
            },
            3: {
                0: 0.1,
                1: 0.1,
                2: 0.0,
                3: 0.8
            }
        }

        terminals = [
            "P",
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)


class NiceLavaFloorEnv(GridEnv):

    def __init__(self):
        grid = [
            ["S", "L", "L", "L"],
            ["L", "W", "L", "P"],
            ["L", "L", "L", "G"]
        ]

        rewards = {
            "L":  50.0,
            "S":  50.0,
            "P": -50.0,
            "G":  5.0
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {
                0: 0.8,
                1: 0.0,
                2: 0.1,
                3: 0.1
            },
            1: {
                0: 0.0,
                1: 0.8,
                2: 0.1,
                3: 0.1
            },
            2: {
                0: 0.1,
                1: 0.1,
                2: 0.8,
                3: 0.0
            },
            3: {
                0: 0.1,
                1: 0.1,
                2: 0.0,
                3: 0.8
            }
        }

        terminals = [
            "P",
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)


class BiggerLavaFloorEnv(GridEnv):

    def __init__(self):
        grid = [
            ["S", "L", "L", "L", "L", "L"],
            ["L", "L", "W", "L", "L", "P"],
            ["L", "P", "W", "L", "L", "W"],
            ["L", "L", "L", "L", "L", "L"],
            ["P", "L", "L", "L", "L", "G"]
        ]

        rewards = {
            "L": -0.04,
            "S": -0.04,
            "P": -10.0,
            "G":  10.0
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {
                0: 0.8,
                1: 0.0,
                2: 0.1,
                3: 0.1
            },
            1: {
                0: 0.0,
                1: 0.8,
                2: 0.1,
                3: 0.1
            },
            2: {
                0: 0.1,
                1: 0.1,
                2: 0.8,
                3: 0.0
            },
            3: {
                0: 0.1,
                1: 0.1,
                2: 0.0,
                3: 0.8
            }
        }

        terminals = [
            "P",
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)


class HugeLavaFloorEnv(GridEnv):

    def __init__(self):
        grid = [
            ["S", "L", "L", "L", "L", "L", "L", "L", "L", "L"],
            ["L", "L", "L", "L", "L", "P", "L", "L", "L", "L"],
            ["L", "L", "L", "W", "L", "L", "W", "L", "L", "L"],
            ["L", "L", "P", "W", "L", "L", "W", "L", "P", "L"],
            ["L", "L", "L", "W", "L", "L", "W", "L", "L", "L"],
            ["L", "L", "L", "W", "W", "W", "W", "L", "L", "L"],
            ["L", "L", "P", "L", "L", "L", "L", "L", "L", "P"],
            ["L", "L", "L", "L", "L", "P", "L", "L", "L", "L"],
            ["L", "L", "L", "L", "L", "L", "L", "L", "L", "L"],
            ["P", "L", "L", "L", "L", "L", "L", "L", "L", "G"]
        ]

        rewards = {
            "L": -0.04,
            "S": -0.04,
            "P": -10.0,
            "G":  10.0
        }

        actions = {
            0: "L",
            1: "R",
            2: "U",
            3: "D"
        }

        dynamics = {
            0: {
                0: 0.8,
                1: 0.0,
                2: 0.1,
                3: 0.1
            },
            1: {
                0: 0.0,
                1: 0.8,
                2: 0.1,
                3: 0.1
            },
            2: {
                0: 0.1,
                1: 0.1,
                2: 0.8,
                3: 0.0
            },
            3: {
                0: 0.1,
                1: 0.1,
                2: 0.0,
                3: 0.8
            }
        }

        terminals = [
            "P",
            "G"
        ]

        super().__init__(grid, rewards, actions, dynamics, terminals)
