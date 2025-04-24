import numpy as np
from typing import List

def bottom(width, height):
    mask = np.uint64(0)
    for w in range(width):
        mask |= np.uint64(1) << (height + 1) * w
    return mask
class Position:
    WIDTH = 7  # width of the board5
    HEIGHT = 6  # height of the board
    MIN_SCORE = -(WIDTH*HEIGHT)//2 + 3
    MAX_SCORE = (WIDTH*HEIGHT+1)//2 - 3
    bottom_mask = bottom(WIDTH, HEIGHT)  # bottom mask for the board
    board_mask = bottom_mask * ((np.uint64(1) << HEIGHT)-1)  # board mask for the board

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
        self.mask |= move
        self.moves += 1

    def play_sequence(self, seq):
        for i, c in enumerate(seq):
            col = int(c) - 1
            if col < 0 or col >= self.WIDTH or not self.can_play(col) or self.is_winning_move(col):
                return i
            self.playCol(col)
        return len(seq)

    def canWinNext(self):
        return self.winning_position() & self.possible() != 0

    def nb_moves(self):
        return self.moves

    def key(self):
        return self.current_position + self.mask

    def switch_player(self):
        self.current_position ^= self.mask

    def print_board(self):
        """Hiển thị bàn cờ dạng ASCII"""
        board = [['.' for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]
        
        # Lấy các quân cờ từ bitboard
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                pos = x * (self.HEIGHT + 1) + y
                if (self.mask >> pos) & 1:
                    if (self.current_position >> pos) & 1:
                        board[self.HEIGHT - 1 - y][x] = 'X'  # Người chơi hiện tại
                    else:
                        board[self.HEIGHT - 1 - y][x] = 'O'  # Đối thủ
        
        # In bàn cờ
        print("  " + " ".join(str(i+1) for i in range(self.WIDTH)))
        for row in board:
            print("|" + " ".join(row) + "|")
        print("+" + "-"*(2*self.WIDTH-1) + "+")

    def possible_Non_Losing_Moves(self):
        possible_mask = self.possible()
        oppoment_win = self.oppoment_winning_position()
        forced_moves = possible_mask & oppoment_win
        if forced_moves != 0:
            if forced_moves & (forced_moves -1) != 0:
                return 0
            else:
                possible_mask = forced_moves
        return possible_mask & ~(oppoment_win >> 1)
    
    def moveScore(self, move):
        return self.popcount(self.compute_winning_position(self.current_position | move, self.mask))

    def can_play(self, col):
        return (self.mask & self.top_mask_col(col)) == 0
    
    def playCol(self, col):
        return self.play((self.mask + self.bottom_mask_col(col)) & self.column_mask(col))
    
    def is_winning_move(self, col):
        return self.winning_position() & self.possible() & self.column_mask(col) != 0

    def winning_position(self):
        return Position.compute_winning_position(self.current_position, self.mask)

    def oppoment_winning_position(self):
        return self.compute_winning_position(self.current_position ^ self.mask, self.mask)

    def possible(self):
        """Trả về danh sách các cột có thể chơi"""
        return (self.mask +Position.bottom_mask) & Position.board_mask

    @staticmethod
    def popcount(x):
        return bin(x).count('1')

    @staticmethod
    def compute_winning_position(position, mask):
        # Vertical
        r = (position << 1) & (position << 2) & (position << 3)

        #Horizontal
        p = (position << (Position.HEIGHT+1)) & (position << 2*(Position.HEIGHT+1))
        r |= p & (position << 3*(Position.HEIGHT+1))
        r |= p & (position >> (Position.HEIGHT+1))
        p = (position >> (Position.HEIGHT+1)) & (position >> 2*(Position.HEIGHT+1))
        r |= p & (position << (Position.HEIGHT+1))
        r |= p & (position >> 3*(Position.HEIGHT+1))

        #Diagonal 1
        p = (position << Position.HEIGHT) & (position << 2*Position.HEIGHT)
        r |= p & (position << 3*Position.HEIGHT)
        r |= p & (position >> Position.HEIGHT)
        p = (position >> Position.HEIGHT) & (position >> 2*Position.HEIGHT)
        r |= p & (position << Position.HEIGHT)
        r |= p & (position >> 3*Position.HEIGHT)

        #Diagonal 2
        p = (position << (Position.HEIGHT+2)) & (position << 2*(Position.HEIGHT+2))
        r |= p & (position << 3*(Position.HEIGHT+2))
        r |= p & (position >> (Position.HEIGHT+2))
        p = (position >> (Position.HEIGHT+2)) & (position >> 2*(Position.HEIGHT+2))
        r |= p & (position << (Position.HEIGHT+2))
        r |= p & (position >> 3*(Position.HEIGHT+2))

        return r & (Position.board_mask ^ mask)
    @staticmethod
    def top_mask_col(col):
        return np.uint64(1) << (Position.HEIGHT - 1) << col * (Position.HEIGHT + 1)

    @staticmethod
    def bottom_mask_col(col):
        return np.uint64(1) << col * (Position.HEIGHT + 1)

    @staticmethod
    def column_mask(col):
        return ((np.uint64(1) << Position.HEIGHT) - 1) << col * (Position.HEIGHT + 1)

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
    position.current_position, position.mask, position.moves = convert_to_bitboard(empty_board,1)
    position.print_board()

if __name__ == "__main__":
    main()