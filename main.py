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


def value_iteration(world: List[List], costs: Dict, goal: Tuple, rewards: int, actions: List, gamma: float) -> Dict:
    rows = len(world)
    cols = len(world[0])
    v_s = np.zeros((rows, cols))
    t = 0
    q = np.zeros((len(actions), rows, cols))
    epsilon = 1
    # q_left = np.zeros(len(goal[0]-1), len(goal))
    # q_right = np.zeros(len(goal[0]-1), len(goal))
    # q_up = np.zeros(len(goal[0]-1), len(goal))
    # q_down = np.zeros(len(goal[0]-1), len(goal))
    policy = {}
    while epsilon > 0.01:
        v_last = deepcopy(v_s)
        t += 1
        for x in range(rows):
            for y in range(cols):
                if (x, y) == goal: reward = 100
                else: reward = 0
                for a in range(len(actions)):
                    if a == 0:
                        temp = costs[world[x][y]]
                        q[a,x,y] = reward + gamma * (0.7 * costs[world[x][(y-1)]] + 0.1*costs[world[(x+1)][y]] + 0.1*costs[world[x][(y+1)]] + 0.1*costs[world[(x-1)][y]])
                    elif a == 1:
                        q[a,x,y] = reward + gamma * (0.7 * costs[world[x+1][y]] + 0.1*costs[world[x][y+1]] + 0.1*costs[world[x-1][y]] + 0.1*costs[world[x][y-1]])
                    elif a == 2:
                        q[a,x,y] = reward + gamma * (0.7 * costs[world[x][y+1]] + 0.1*costs[world[x-1][y]] + 0.1*costs[world[x][y-1]] + 0.1*costs[world[x+1][y]])
                    else:
                        q[a,x,y] = reward + gamma * (0.7 * costs[world[x-1][y]] + 0.1*costs[world[x][y-1]] + 0.1*costs[world[x+1][y]] + 0.1*costs[world[x][y+1]])
                direction = q.argmax(2)
                policy.update({(x,y): '<'})
                v_s[x,y] = q[:][x][y].max()
        epsilon = abs(v_s - v_last)
    return policy


if __name__ == "__main__":
    small_world = read_world('small.txt')
    large_world = read_world('large.txt')
    costs = {'.': -1, '*': -3, '^': -5, '~': -7}
    gamma = 0.9
    cardinal_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    reward = 100
    goal = (len(small_world[0]) - 1, len(small_world) - 1)

    small_policy = value_iteration(small_world, costs, goal, reward, cardinal_moves, gamma)
    print(small_policy)