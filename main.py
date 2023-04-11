import numpy as np
from typing import List, Tuple, Dict, Callable


def read_world(filename: str) -> List[List]:
    result = []
    with open(filename) as f:
        for line in f.readlines():
            if len(line) > 0:
                result.append(list(line.strip()))
    return result


def value_iteration(world: List[List], costs: Dict, goal: Tuple, rewards: int, actions: List, gamma: float) -> Dict:
    v_s = np.zeros(goal)
    t = 0
    q_left = []
    q_right = []
    q_up = []
    q_down = []
    while epsilon > 0.01:
        for state in world:
            for act in actions:

    pass


if __name__ == "__main__":
    small_world = read_world('small.txt')
    large_world = read_world('large.txt')
    costs = {'.': -1, '*': -3, '^': -5, '~': -7}
    gamma = 0.9
    cardinal_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    reward = 100
    goal = (len(small_world[0]) - 1, len(small_world) - 1)
