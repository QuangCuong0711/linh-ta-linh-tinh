from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import math
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool

class AIResponse(BaseModel):
    move: int

def bottom(width, height):
    mask = np.uint64(0)
    for w in range(width):
        mask |= np.uint64(1) << np.uint64((height + 1) * w)
    return mask

class MoveSorter:
    def __init__(self):
        self.size = 0
        self.entries = np.zeros(Position.WIDTH, dtype=[('move', 'u8'), ('score', 'i4')])

    def add(self, move: int, score: int) -> None:
        if self.size >= len(self.entries):
            return

        pos = self.size
        while pos > 0 and self.entries[pos-1]['score'] > score:
            if pos < len(self.entries):
                self.entries[pos] = self.entries[pos-1]
            pos -= 1

        if pos < len(self.entries):
            self.entries[pos]['move'] = move
            self.entries[pos]['score'] = score
        self.size += 1

    def get_next(self) -> int:
        if self.size > 0:
            self.size -= 1
            return int(self.entries[self.size]['move'])
        return 0

    def reset(self) -> None:
        self.size = 0

    def __len__(self) -> int:
        return self.size

class Position:
    WIDTH = 7
    HEIGHT = 6
    MIN_SCORE = -(WIDTH*HEIGHT)//2 + 3
    MAX_SCORE = (WIDTH*HEIGHT+1)//2 - 3
    bottom_mask = bottom(WIDTH, HEIGHT)
    board_mask = bottom_mask * ((np.uint64(1) << np.uint64(HEIGHT)) - np.uint64(1))

    def __init__(self, other=None):
        if other:
            self.current_position = np.uint64(other.current_position)
            self.mask = np.uint64(other.mask)
            self.moves = other.moves
        else:
            self.current_position = np.uint64(0)
            self.mask = np.uint64(0)
            self.moves = 0

    def play(self, move):
        self.current_position ^= self.mask
        self.mask |= np.uint64(move)
        self.moves += 1

    def play_sequence(self, seq):
        for i, c in enumerate(seq):
            col = int(c) - 1
            if col < 0 or col >= self.WIDTH or not self.can_play(col) or self.is_winning_move(col):
                return i
            self.playCol(col)
        return len(seq)

    def canWinNext(self):
        return (self.winning_position() & self.possible()) != np.uint64(0)

    def nb_moves(self):
        return self.moves

    def key(self):
        return np.uint64(self.current_position + self.mask)

    def switch_player(self):
        self.current_position ^= self.mask

    def print_board(self):
        board = [['.' for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]

        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                pos = x * (self.HEIGHT + 1) + y
                if (self.mask >> np.uint64(pos)) & np.uint64(1):
                    if (self.current_position >> np.uint64(pos)) & np.uint64(1):
                        board[self.HEIGHT - 1 - y][x] = 'X'
                    else:
                        board[self.HEIGHT - 1 - y][x] = 'O'

        print("  " + " ".join(str(i+1) for i in range(self.WIDTH)))
        for row in board:
            print("|" + " ".join(row) + "|")
        print("+" + "-"*(2*self.WIDTH-1) + "+")

    def possible_Non_Losing_Moves(self):
        possible_mask = self.possible()
        opponent_win = self.opponent_winning_position()
        forced_moves = possible_mask & opponent_win
        if forced_moves != np.uint64(0):
            if forced_moves & (forced_moves - np.uint64(1)) != np.uint64(0):
                return np.uint64(0)
            else:
                possible_mask = forced_moves
        return possible_mask & ~(opponent_win >> np.uint64(1))

    def moveScore(self, move):
        return self.popcount(self.compute_winning_position(self.current_position | np.uint64(move), self.mask))

    def can_play(self, col):
        return (self.mask & self.top_mask_col(col)) == np.uint64(0)

    def playCol(self, col):
        move = (self.mask + self.bottom_mask_col(col)) & self.column_mask(col)
        return self.play(np.uint64(move))

    def is_winning_move(self, col):
        return (self.winning_position() & self.possible() & self.column_mask(col)) != np.uint64(0)

    def winning_position(self):
        return self.compute_winning_position(self.current_position, self.mask)

    def opponent_winning_position(self):
        return self.compute_winning_position(self.current_position ^ self.mask, self.mask)

    def possible(self):
        return np.uint64(self.mask + self.bottom_mask) & self.board_mask

    @staticmethod
    def popcount(x):
        return bin(int(x)).count('1')

    @staticmethod
    def compute_winning_position(position, mask):
        position = np.uint64(position)
        mask = np.uint64(mask)

        # Vertical
        r = (position << np.uint64(1)) & (position << np.uint64(2)) & (position << np.uint64(3))

        # Horizontal
        p = (position << np.uint64(Position.HEIGHT+1)) & (position << np.uint64(2*(Position.HEIGHT+1)))
        r |= p & (position << np.uint64(3*(Position.HEIGHT+1)))
        r |= p & (position >> np.uint64(Position.HEIGHT+1))
        p = (position >> np.uint64(Position.HEIGHT+1)) & (position >> np.uint64(2*(Position.HEIGHT+1)))
        r |= p & (position << np.uint64(Position.HEIGHT+1))
        r |= p & (position >> np.uint64(3*(Position.HEIGHT+1)))

        # Diagonal 1
        p = (position << np.uint64(Position.HEIGHT)) & (position << np.uint64(2*Position.HEIGHT))
        r |= p & (position << np.uint64(3*Position.HEIGHT))
        r |= p & (position >> np.uint64(Position.HEIGHT))
        p = (position >> np.uint64(Position.HEIGHT)) & (position >> np.uint64(2*Position.HEIGHT))
        r |= p & (position << np.uint64(Position.HEIGHT))
        r |= p & (position >> np.uint64(3*Position.HEIGHT))

        # Diagonal 2
        p = (position << np.uint64(Position.HEIGHT+2)) & (position << np.uint64(2*(Position.HEIGHT+2)))
        r |= p & (position << np.uint64(3*(Position.HEIGHT+2)))
        r |= p & (position >> np.uint64(Position.HEIGHT+2))
        p = (position >> np.uint64(Position.HEIGHT+2)) & (position >> np.uint64(2*(Position.HEIGHT+2)))
        r |= p & (position << np.uint64(Position.HEIGHT+2))
        r |= p & (position >> np.uint64(3*(Position.HEIGHT+2)))

        return r & (Position.board_mask ^ mask)

    @staticmethod
    def top_mask_col(col):
        return np.uint64(1) << np.uint64(Position.HEIGHT - 1) << np.uint64(col * (Position.HEIGHT + 1))

    @staticmethod
    def bottom_mask_col(col):
        return np.uint64(1) << np.uint64(col * (Position.HEIGHT + 1))

    @staticmethod
    def column_mask(col):
        if col is None:
            raise ValueError("col is None in column_mask()")

        height = np.uint64(Position.HEIGHT)
        col = np.uint64(col)

        return ((np.uint64(1) << height) - np.uint64(1)) << (col * (height + np.uint64(1)))

class Solver:
    def __init__(self, max_depth=14):
        self.node_count = 0
        self.max_depth = max_depth
        self.column_order = [3, 2, 4, 1, 5, 0, 6]
        self.transposition_table = TranspositionTable(Position.WIDTH*(Position.HEIGHT + 1),
                                                     self.log2(Position.MAX_SCORE - Position.MIN_SCORE + 1) + 1, 23)

    def log2(self, n):
        if n <= 1:
            return 0
        return int(np.log2(n/2) + 1)

    def evaluate(self, position):
        if position.is_winning_move(0):
            for col in range(Position.WIDTH):
                if position.is_winning_move(col):
                    return float('inf') if position.nb_moves() % 2 == 1 else float('-inf')

        def count_potential_fours(pos, mask):
            count = 0
            for y in range(Position.HEIGHT):
                for x in range(Position.WIDTH - 3):
                    window = 0
                    for i in range(4):
                        bit_pos = (x + i) * (Position.HEIGHT + 1) + y
                        if not (mask & (np.uint64(1) << np.uint64(bit_pos))):
                            window = -1
                            break
                    if window != -1:
                        count += 1

            for x in range(Position.WIDTH):
                for y in range(Position.HEIGHT - 3):
                    window = 0
                    for i in range(4):
                        bit_pos = x * (Position.HEIGHT + 1) + y + i
                        if not (mask & (np.uint64(1) << np.uint64(bit_pos))):
                            window = -1
                            break
                    if window != -1:
                        count += 1

            for x in range(Position.WIDTH - 3):
                for y in range(Position.HEIGHT - 3):
                    window = 0
                    for i in range(4):
                        bit_pos = (x + i) * (Position.HEIGHT + 1) + y + i
                        if not (mask & (np.uint64(1) << np.uint64(bit_pos))):
                            window = -1
                            break
                    if window != -1:
                        count += 1

            for x in range(Position.WIDTH - 3):
                for y in range(3, Position.HEIGHT):
                    window = 0
                    for i in range(4):
                        bit_pos = (x + i) * (Position.HEIGHT + 1) + y - i
                        if not (mask & (np.uint64(1) << np.uint64(bit_pos))):
                            window = -1
                            break
                    if window != -1:
                        count += 1

            return count

        current_player = position.current_position ^ position.mask
        opponent = position.current_position

        current_potential = count_potential_fours(current_player, position.mask)
        opponent_potential = count_potential_fours(opponent, position.mask)

        return current_potential - opponent_potential

    def negamax(self, P, alpha, beta, depth=0):
        assert alpha < beta
        self.node_count += 1

        if self.max_depth is not None and depth >= self.max_depth:
            return self.evaluate(P)

        possible = P.possible_Non_Losing_Moves()
        if possible == np.uint64(0):
            return -(Position.WIDTH * Position.HEIGHT - P.nb_moves()) // 2
        if P.nb_moves() == Position.WIDTH * Position.HEIGHT - 2:
            return 0

        min_score = -(Position.WIDTH*Position.HEIGHT-2 - P.nb_moves())//2
        if alpha < min_score:
            alpha = min_score
            if alpha >= beta:
                return alpha

        max_score = (Position.WIDTH * Position.HEIGHT - 1 - P.nb_moves()) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta

        key = P.key()
        val = self.transposition_table.get(key)
        if val:
            if val > Position.MAX_SCORE - Position.MIN_SCORE + 1:
                min_val = val + 2*Position.MIN_SCORE - Position.MAX_SCORE - 2
                if alpha < min_val:
                    alpha = min_val
                    if alpha >= beta:
                        return alpha
            else:
                max_val = val + Position.MIN_SCORE - 1
                if beta > max_val:
                    beta = max_val
                    if alpha >= beta:
                        return beta

        moves = MoveSorter()
        for i in reversed(range(Position.WIDTH)):
            move = possible & Position.column_mask(self.column_order[i])
            if move != np.uint64(0):
                moves.add(move, P.moveScore(move))

        best_score = -float('inf')
        while True:
            next_move = moves.get_next()
            if next_move == 0:
                break

            P2 = Position(P)
            P2.play(np.uint64(next_move))
            score = -self.negamax(P2, -beta, -alpha, depth + 1)

            if score >= beta:
                value_to_store = score + Position.MAX_SCORE - 2*Position.MIN_SCORE + 2
                max_allowed = (1 << self.transposition_table.value_size) - 1
                if value_to_store > max_allowed:
                    value_to_store = max_allowed
                elif value_to_store < 0:
                    value_to_store = 0
                self.transposition_table.put(key, value_to_store)
                return score

            if score > best_score:
                best_score = score
                if score > alpha:
                    alpha = score

        value_to_store = alpha - Position.MIN_SCORE + 1
        max_allowed = (1 << self.transposition_table.value_size) - 1
        if value_to_store > max_allowed:
            value_to_store = max_allowed
        elif value_to_store < 0:
            value_to_store = 0
        self.transposition_table.put(key, value_to_store)

        return best_score

    def solve(self, P):
        if P.canWinNext():
            return (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        min_score = -(Position.WIDTH * Position.HEIGHT - P.nb_moves()) // 2
        max_score = (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        while min_score < max_score:
            med = min_score + (max_score - min_score) // 2
            if med <= 0 and min_score // 2 < med:
                med = min_score // 2
            elif med >= 0 and max_score // 2 > med:
                med = max_score // 2

            score = self.negamax(P, med, med + 1)

            if score <= med:
                max_score = score
            else:
                min_score = score

        return min_score

    def set_max_depth(self, depth):
        self.max_depth = depth

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def next_prime(n: int) -> int:
    while not is_prime(n):
        n += 1
    return n

def log2(n: int) -> int:
    return 0 if n <= 1 else log2(n // 2) + 1

class TranspositionTable:
    def __init__(self, key_size: int = 64, value_size: int = 16, log_size: int = 20):
        assert key_size <= 64
        assert value_size <= 64
        assert log_size <= 64

        self.key_size = key_size
        self.value_size = value_size
        self.log_size = log_size
        self.size = next_prime(1 << log_size)

        self.key_t = self._get_uint_type(key_size - log_size)
        self.value_t = self._get_uint_type(value_size)

        self.K = np.zeros(self.size, dtype=self.key_t)
        self.V = np.zeros(self.size, dtype=self.value_t)

    def _get_uint_type(self, bits: int) -> type:
        if bits <= 8:
            return np.uint8
        elif bits <= 16:
            return np.uint16
        elif bits <= 32:
            return np.uint32
        return np.uint64

    def index(self, key: int) -> int:
        return key % self.size

    def reset(self) -> None:
        self.K.fill(0)
        self.V.fill(0)

    def put(self, key: int, value: int) -> None:
        assert key >> self.key_size == 0
        assert value >> self.value_size == 0

        pos = self.index(key)
        self.K[pos] = key
        self.V[pos] = value

    def get(self, key: int) -> int:
        assert key >> self.key_size == 0

        pos = self.index(key)
        return int(self.V[pos]) if self.K[pos] == (self.key_t)(key) else 0

    def __del__(self):
        if hasattr(self, 'K'):
            del self.K
        if hasattr(self, 'V'):
            del self.V

def convert_to_bitboard(board: List[List[int]], current_player: int):
    WIDTH, HEIGHT = 7, 6
    position = np.uint64(0)
    mask = np.uint64(0)
    moves = 0

    for row in reversed(range(HEIGHT)):
        for col in range(WIDTH):
            if board[row][col] != 0:
                bit = col * (HEIGHT + 1) + (HEIGHT - 1 - row)
                mask |= np.uint64(1) << np.uint64(bit)
                if board[row][col] == current_player:
                    position |= np.uint64(1) << np.uint64(bit)
                moves += 1

    return position, mask, moves

def best_move(position: Position, valid_moves: List[int], solver: Solver):
    best_col = None
    best_score = -float('inf')

    for col in valid_moves:
        if position.can_play(col) and position.is_winning_move(col):
            return col

    for x in range(Position.WIDTH):
        col = solver.column_order[x]
        if position.can_play(col):
            P2 = Position(position)
            P2.playCol(col)

            opponent_can_win = False
            for y in range(Position.WIDTH):
                if P2.can_play(y) and P2.is_winning_move(y):
                    opponent_can_win = True
                    break

            if opponent_can_win:
                continue

            score = -solver.solve(P2)
            if score > best_score or best_col is None:
                best_col = col
                best_score = score

    if best_col is None:
        for col in range(Position.WIDTH):
            if position.can_play(col):
                best_col = col
                break
    return best_col

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    position = Position()
    solver = Solver()
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")

        position.mask, position.current_position, position.moves = convert_to_bitboard(
            game_state.board, game_state.current_player
        )
        print(f"Bảng hiện tại:\n{game_state.board}")
        print(f"Người chơi hiện tại: {game_state.current_player}")
        position.print_board()

        selected_move = best_move(position, game_state.valid_moves, solver)
        print(f"Nước đi được chọn: {selected_move + 1}") # +1 để hiển thị theo cột 1-7

        return AIResponse(move=selected_move)
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        if game_state.valid_moves:
            return AIResponse(move=random.choice(game_state.valid_moves))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)