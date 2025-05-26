import numpy as np
from collections import deque, defaultdict
import heapq
import time

# Face indices
(U, F, R, B, L, D) = (0, 1, 2, 3, 4, 5)


class Cube2x2:
    def __init__(self):
        self.faces = {
            U: np.full((2, 2), 'W'),  # White
            F: np.full((2, 2), 'G'),  # Green
            R: np.full((2, 2), 'R'),  # Red
            B: np.full((2, 2), 'B'),  # Blue
            L: np.full((2, 2), 'O'),  # Orange
            D: np.full((2, 2), 'Y')  # Yellow
        }

    def is_solved(self):
        return all(np.all(face == face[0, 0]) for face in self.faces.values())

    def rotate_face(self, face, clockwise=True):
        self.faces[face] = np.rot90(self.faces[face], 3 if clockwise else 1)

    def move_U(self, clockwise=True):
        self.rotate_face(U, clockwise)
        temp = np.copy(self.faces[F][0])
        if clockwise:
            self.faces[F][0] = self.faces[R][0]
            self.faces[R][0] = self.faces[B][0]
            self.faces[B][0] = self.faces[L][0]
            self.faces[L][0] = temp
        else:
            self.faces[F][0] = self.faces[L][0]
            self.faces[L][0] = self.faces[B][0]
            self.faces[B][0] = self.faces[R][0]
            self.faces[R][0] = temp

    def move_F(self, clockwise=True):
        self.rotate_face(F, clockwise)
        temp = np.copy(self.faces[U][1])
        if clockwise:
            self.faces[U][1] = np.flip(self.faces[L][:, 1])
            self.faces[L][:, 1] = self.faces[D][0]
            self.faces[D][0] = np.flip(self.faces[R][:, 0])
            self.faces[R][:, 0] = temp
        else:
            self.faces[U][1] = self.faces[R][:, 0]
            self.faces[R][:, 0] = np.flip(self.faces[D][0])
            self.faces[D][0] = self.faces[L][:, 1]
            self.faces[L][:, 1] = np.flip(temp)

    def move_R(self, clockwise=True):
        self.rotate_face(R, clockwise)
        temp = np.copy(self.faces[U][:, 1])
        if clockwise:
            self.faces[U][:, 1] = self.faces[F][:, 1]
            self.faces[F][:, 1] = self.faces[D][:, 1]
            self.faces[D][:, 1] = np.flip(self.faces[B][:, 0])
            self.faces[B][:, 0] = np.flip(temp)
        else:
            self.faces[U][:, 1] = np.flip(self.faces[B][:, 0])
            self.faces[B][:, 0] = np.flip(self.faces[D][:, 1])
            self.faces[D][:, 1] = self.faces[F][:, 1]
            self.faces[F][:, 1] = temp

    def apply_move(self, move):
        if move == "U":
            self.move_U(True)
        elif move == "U'":
            self.move_U(False)
        elif move == "F":
            self.move_F(True)
        elif move == "F'":
            self.move_F(False)
        elif move == "R":
            self.move_R(True)
        elif move == "R'":
            self.move_R(False)

    def scramble(self, moves):
        for move in moves.split():
            self.apply_move(move)

    def copy(self):
        new_cube = Cube2x2()
        for face in self.faces:
            new_cube.faces[face] = np.copy(self.faces[face])
        return new_cube

    def __hash__(self):
        return hash(tuple(tuple(row) for face in [U, F, R, B, L, D] for row in self.faces[face]))

    def __eq__(self, other):
        return all(np.array_equal(self.faces[face], other.faces[face]) for face in self.faces)


# Search Algorithms ===========================================================
def dfs_solve(cube, max_depth=10):
    move_set = ["U", "U'", "F", "F'", "R", "R'"]
    stack = deque([(cube.copy(), [])])
    visited = set()
    visited.add(hash(cube))
    start = time.time()
    nodes = 0

    while stack:
        current, moves = stack.pop()
        nodes += 1

        if current.is_solved():
            return moves, time.time() - start, nodes

        if len(moves) >= max_depth:
            continue

        for move in move_set:
            new_cube = current.copy()
            new_cube.apply_move(move)
            h = hash(new_cube)
            if h not in visited:
                visited.add(h)
                stack.append((new_cube, moves + [move]))

    return None, time.time() - start, nodes


def bfs_solve(cube, max_moves=10):
    move_set = ["U", "U'", "F", "F'", "R", "R'"]
    queue = deque([(cube.copy(), [])])
    visited = set()
    visited.add(hash(cube))
    start = time.time()
    nodes = 0

    while queue:
        current, moves = queue.popleft()
        nodes += 1

        if current.is_solved():
            return moves, time.time() - start, nodes

        if len(moves) >= max_moves:
            continue

        for move in move_set:
            new_cube = current.copy()
            new_cube.apply_move(move)
            h = hash(new_cube)
            if h not in visited:
                visited.add(h)
                queue.append((new_cube, moves + [move]))

    return None, time.time() - start, nodes


def heuristic(cube):
    """Admissible heuristic for A* (misplaced facelets / 4)"""
    errors = sum(np.sum(face != face[0, 0]) for face in cube.faces.values())
    return errors // 4


def a_star_solve(cube, max_moves=15):
    move_set = ["U", "U'", "F", "F'", "R", "R'"]
    heap = []
    counter = 0  # Unique sequence counter to avoid comparing cubes
    initial_h = heuristic(cube)
    heapq.heappush(heap, (initial_h, counter, 0, initial_h, cube.copy(), []))
    visited = dict()
    visited[hash(cube)] = 0
    start = time.time()
    nodes = 0

    while heap:
        _, _, g, h, current, moves = heapq.heappop(heap)
        nodes += 1

        if current.is_solved():
            return moves, time.time() - start, nodes

        if len(moves) >= max_moves:
            continue

        for move in move_set:
            new_cube = current.copy()
            new_cube.apply_move(move)
            new_g = g + 1
            new_h = heuristic(new_cube)
            cube_hash = hash(new_cube)

            if cube_hash not in visited or new_g < visited[cube_hash]:
                counter += 1
                visited[cube_hash] = new_g
                heapq.heappush(heap, (new_g + new_h, counter, new_g, new_h, new_cube, moves + [move]))

    return None, time.time() - start, nodes


# Main Execution ==============================================================
if __name__ == "__main__":
    # Configuration
    SCRAMBLE = "U R F'"  # Test scramble
    ALGORITHM = "a_star"  # Choose: dfs, bfs, a_star
    MAX_DEPTH = 12

    # Initialize cube
    cube = Cube2x2()
    cube.scramble(SCRAMBLE)

    # Solve with selected algorithm
    print(f"Solving scramble: {SCRAMBLE} using {ALGORITHM.upper()}")
    start_total = time.time()

    if ALGORITHM == "dfs":
        solution, search_time, nodes = dfs_solve(cube, MAX_DEPTH)
    elif ALGORITHM == "bfs":
        solution, search_time, nodes = bfs_solve(cube, MAX_DEPTH)
    elif ALGORITHM == "a_star":
        solution, search_time, nodes = a_star_solve(cube, MAX_DEPTH)
    else:
        raise ValueError("Invalid algorithm choice")

    total_time = time.time() - start_total

    # Results
    if solution:
        print(f"\nSolution ({len(solution)} moves): {' '.join(solution)}")
        print(f"Search time: {search_time:.2f}s | Total time: {total_time:.2f}s")
        print(f"Nodes expanded: {nodes}")

        # Verification
        test_cube = Cube2x2()
        test_cube.scramble(SCRAMBLE)
        for move in solution:
            test_cube.apply_move(move)
        print("\nVerification:", "Solved!" if test_cube.is_solved() else "Failed!")
    else:
        print("\nNo solution found. Try increasing MAX_DEPTH.")