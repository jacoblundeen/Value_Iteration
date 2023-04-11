import numpy as np
from typing import List, Tuple, Dict, Callable
from copy import deepcopy


def read_world(filename: str) -> List[List]:
    result = []
    with open(filename) as f:
        for line in f.readlines():
            if len(line) > 0:
                result.append(list(line.strip()))
    return result


def update_q(v_s: List, a: int, x: int, y: int, q: List, reward: int, gamma: float) -> List:
    if a == 0:
        q[a, x, y] = reward + gamma * (
                0.7 * v_s[x][y - 1] + 0.1 * v_s[(x + 1)][y] + 0.1 * v_s[x][(y + 1)] + 0.1 * v_s[(x - 1)][y])
    elif a == 1:
        q[a, x, y] = reward + gamma * (
                0.7 * v_s[x + 1][y] + 0.1 * v_s[x][y + 1] + 0.1 * v_s[x - 1][y] + 0.1 * v_s[x][y - 1])
    elif a == 2:
        q[a, x, y] = reward + gamma * (
                0.7 * v_s[x][y + 1] + 0.1 * v_s[x - 1][y] + 0.1 * v_s[x][y - 1] + 0.1 * v_s[x + 1][y])
    else:
        q[a, x, y] = reward + gamma * (
                0.7 * v_s[x - 1][y] + 0.1 * v_s[x][y - 1] + 0.1 * v_s[x + 1][y] + 0.1 * v_s[x][y + 1])
    return q


def update_policy(q: List, policy: Dict, x: int, y: int, actions: List) -> Dict:
    direction = q[:, x, y].argmax(0)
    if direction == 0:
        policy.update({(x, y): actions[0]})
    elif direction == 1:
        policy.update({(x, y): actions[1]})
    elif direction == 2:
        policy.update({(x, y): actions[2]})
    else:
        policy.update({(x, y): actions[3]})
    return policy


def value_iteration(world: List[List], costs: Dict, goal: Tuple, rewards: int, actions: List, gamma: float) -> Dict:
    rows = len(world)
    cols = len(world[0])
    v_s = np.zeros((rows + 1, cols + 1))
    q = np.zeros((len(actions), rows, cols))
    policy = {}
    epoch = True
    while epoch:
        v_last = deepcopy(v_s)
        for x in range(rows):
            for y in range(cols):
                if (x, y) == goal:
                    reward = rewards
                elif world[x][y] not in costs.keys():
                    policy.update({(x, y): (0, 0)})
                    continue
                else:
                    reward = costs[world[x][y]]
                for a in range(len(actions)):
                    q = update_q(v_s, a, x, y, q, reward, gamma)
                policy = update_policy(q, policy, x, y, actions)
                v_s[x, y] = q[:, x, y].max(0)
        if np.all(abs(np.subtract(v_s, v_last)) < 0.01): epoch = False
    return policy


def pretty_print_policy(cols: int, rows: int, policy: Dict, goal: Tuple):
    row = []
    for key, values in policy.items():
        if key == goal:
            row.append('G')
        elif values == (0, -1):
            row.append('<')
        elif values == (1, 0):
            row.append('V')
        elif values == (0, 1):
            row.append('>')
        elif values == (0, 0):
            row.append('X')
        else:
            row.append('^')
        if key[1] == cols - 1:
            print(''.join(row))
            row = []


if __name__ == "__main__":
    small_world = read_world('small.txt')
    large_world = read_world('large.txt')
    costs = {'.': -1, '*': -3, '^': -5, '~': -7}
    gamma = 0.9
    cardinal_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    reward = 100
    goal = (len(small_world) - 1, len(small_world[0]) - 1)

    cols = len(small_world[0])
    rows = len(small_world)
    small_policy = value_iteration(small_world, costs, goal, reward, cardinal_moves, gamma)
    pretty_print_policy(cols, rows, small_policy, goal)

    cols = len(large_world[0])
    rows = len(large_world)
    goal = (len(large_world) - 1, len(large_world[0]) - 1)
    large_policy = value_iteration(large_world, costs, goal, reward, cardinal_moves, gamma)
    pretty_print_policy(cols, rows, large_policy, goal)
