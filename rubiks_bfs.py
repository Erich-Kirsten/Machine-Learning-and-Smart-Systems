import numpy as np
from collections import deque
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


def bfs_solve(cube, max_moves=40):
    move_set = ["U", "U'", "F", "F'", "R", "R'"]
    queue = deque()
    queue.append((cube.copy(), []))
    visited = set()
    visited.add(hash(cube))

    start_time = time.time()
    nodes_expanded = 0

    while queue:
        current_cube, moves = queue.popleft()
        nodes_expanded += 1

        if current_cube.is_solved():
            return moves, time.time() - start_time, nodes_expanded

        if len(moves) >= max_moves:
            continue

        for move in move_set:
            new_cube = current_cube.copy()
            new_cube.apply_move(move)
            new_hash = hash(new_cube)

            if new_hash not in visited:
                visited.add(new_hash)
                queue.append((new_cube, moves + [move]))

    return None, time.time() - start_time, nodes_expanded


# Example usage
if __name__ == "__main__":
    # Create and scramble cube
    cube = Cube2x2()
    scramble = "U' R F' U' F R U' F' UU"  # 3-move scramble
    cube.scramble(scramble)

    print(f"Solving cube scrambled with: {scramble}")
    start_time = time.time()
    solution, solve_time, nodes = bfs_solve(cube)
    total_time = time.time() - start_time

    if solution:
        print(f"\nSolution found in {len(solution)} moves: {' '.join(solution)}")
        print(f"BFS search time: {solve_time:.4f} seconds")
        print(f"Total execution time: {total_time:.4f} seconds")
        print(f"Nodes expanded: {nodes}")

        # Verify solution
        test_cube = Cube2x2()
        test_cube.scramble(scramble)
        for move in solution:
            test_cube.apply_move(move)
        print("\nVerification:", "Solved!" if test_cube.is_solved() else "Failed!")
    else:
        print("\nNo solution found within the move limit.")