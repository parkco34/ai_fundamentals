#!/usr/bin/env python
import heapq
from typing import List, Tuple

class PuzzleNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.g = 0 if parent is None else parent.g + 1
        self.h = self.calculate_heuristic()
        self.f = self.g + self.h

    def calculate_heuristic(self):
        goal = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
        return sum(abs(i - goal[x][y]) + abs(j - x) + abs(k - y)
                   for i, row in enumerate(self.state)
                   for j, val in enumerate(row)
                   for x, goal_row in enumerate(goal)
                   for y, goal_val in enumerate(goal_row)
                   if val != 0 and val == goal_val)

    def __lt__(self, other):
        return self.f < other.f

def get_neighbors(state: Tuple[Tuple[int, ...], ...]) -> List[Tuple[Tuple[Tuple[int, ...], ...], str]]:
    neighbors = []
    moves = [('UP', -1, 0), ('DOWN', 1, 0), ('LEFT', 0, -1), ('RIGHT', 0, 1)]
    zero_pos = next((i, j) for i, row in enumerate(state) for j, val in enumerate(row) if val == 0)

    for move, dx, dy in moves:
        new_x, new_y = zero_pos[0] + dx, zero_pos[1] + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = list(map(list, state))
            new_state[zero_pos[0]][zero_pos[1]], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[zero_pos[0]][zero_pos[1]]
            neighbors.append((tuple(map(tuple, new_state)), move))

    return neighbors

def solve_puzzle(start_state: Tuple[Tuple[int, ...], ...], goal_state: Tuple[Tuple[int, ...], ...]):
    start_node = PuzzleNode(start_state)
    frontier = [(start_node.f, start_node)]
    explored = set()

    while frontier:
        _, current_node = heapq.heappop(frontier)

        if current_node.state == goal_state:
            path = []
            while current_node.parent:
                path.append(current_node.move)
                current_node = current_node.parent
            return path[::-1]

        explored.add(current_node.state)

        for neighbor_state, move in get_neighbors(current_node.state):
            if neighbor_state not in explored:
                neighbor_node = PuzzleNode(neighbor_state, current_node, move)
                heapq.heappush(frontier, (neighbor_node.f, neighbor_node))

    return None

def main():
    start_state = ((7, 2, 4), (5, 0, 6), (8, 3, 1))
    goal_state = ((1, 2, 3), (4, 5, 0), (6, 7, 8))

    solution = solve_puzzle(start_state, goal_state)

    if solution:
        print(f"Solution found in {len(solution)} moves:")
        print(", ".join(solution))
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()

