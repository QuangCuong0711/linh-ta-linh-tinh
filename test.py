import numpy as np
from typing import List

def bottom(width, height):
    mask = np.uint64(0)
    for w in range(width):
        mask |= np.uint64(1) << (height + 1) * w
    return mask


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
        print(f"column_mask called with col = {col} (type={type(col)})")
        return ((np.uint64(1) << np.uint64(Position.HEIGHT)) - np.uint64(1)) << np.uint64(col * (Position.HEIGHT + 1))

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

def main():
    position = Position()
    empty_board = [
        [1, 0, 0, 0, 2, 0, 0],
        [0, 1, 0, 0, 2, 0, 0],
        [0, 0, 1, 0, 2, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 2, 0, 0],
        [1, 0, 1, 2, 2, 1, 1]
    ]
    position.current_position, position.mask, position.moves = convert_to_bitboard(empty_board,2)
    position.print_board()

if __name__ == "__main__":
    main()