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
        mask |= np.uint64(1) << (height + 1) * w
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
            if pos < len(self.entries):  # Thêm điều kiện kiểm tra
                self.entries[pos] = self.entries[pos-1]
            pos -= 1
        
        if pos < len(self.entries):  # Thêm điều kiện kiểm tra
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

    def nb_movesnb_moves(self):
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

class Solver:
    
    def __init__(self, max_depth = 14):
        self.node_count = 0
        self.max_depth = max_depth  # Độ sâu tối đa (None = không giới hạn)
        self.column_order = [3, 2, 4, 1, 5, 0, 6]
        self.transposition_table = TranspositionTable(Position.WIDTH*(Position.HEIGHT + 1), self.log2(Position.MAX_SCORE - Position.MIN_SCORE + 1) + 1,23)  

    def log2(self, n):
        if n <= 1:
            return 0
        return int(np.log2(n/2) + 1)
    
    def evaluate(self, position):
        """
        Đánh giá vị trí hiện tại bằng cách đếm các hàng 4 tiềm năng
        Trả về điểm số: (số hàng tiềm năng của người chơi) - (số hàng tiềm năng của đối thủ)
        """
        # Kiểm tra nếu có người thắng
        if position.is_winning_move(0):  # Giả sử 0 là cột đầu tiên, cần kiểm tra tất cả cột
            for col in range(Position.WIDTH):
                if position.is_winning_move(col):
                    return float('inf') if position.nb_moves() % 2 == 1 else float('-inf')
        
        # Tính số hàng 4 tiềm năng cho mỗi người chơi
        def count_potential_fours(pos, mask):
            count = 0
            
            # Kiểm tra hàng ngang (7 vị trí × 4 cách)
            for y in range(Position.HEIGHT):
                for x in range(Position.WIDTH - 3):
                    window = 0
                    for i in range(4):
                        bit_pos = (x + i) * (Position.HEIGHT + 1) + y
                        if not (mask & (1 << bit_pos)):
                            window = -1  # Có ô trống trong cửa sổ
                            break
                    if window != -1:
                        count += 1
            
            # Kiểm tra hàng dọc (4 vị trí × 7 cách)
            for x in range(Position.WIDTH):
                for y in range(Position.HEIGHT - 3):
                    window = 0
                    for i in range(4):
                        bit_pos = x * (Position.HEIGHT + 1) + y + i
                        if not (mask & (1 << bit_pos)):
                            window = -1
                            break
                    if window != -1:
                        count += 1
            
            # Kiểm tra chéo lên (4 vị trí × 4 cách)
            for x in range(Position.WIDTH - 3):
                for y in range(Position.HEIGHT - 3):
                    window = 0
                    for i in range(4):
                        bit_pos = (x + i) * (Position.HEIGHT + 1) + y + i
                        if not (mask & (1 << bit_pos)):
                            window = -1
                            break
                    if window != -1:
                        count += 1
            
            # Kiểm tra chéo xuống (4 vị trí × 4 cách)
            for x in range(Position.WIDTH - 3):
                for y in range(3, Position.HEIGHT):
                    window = 0
                    for i in range(4):
                        bit_pos = (x + i) * (Position.HEIGHT + 1) + y - i
                        if not (mask & (1 << bit_pos)):
                            window = -1
                            break
                    if window != -1:
                        count += 1
            
            return count
        
        # Số hàng tiềm năng cho người chơi hiện tại và đối thủ
        current_player = position.current_position ^ position.mask
        opponent = position.current_position
        
        current_potential = count_potential_fours(current_player, position.mask)
        opponent_potential = count_potential_fours(opponent, position.mask)
        
        return current_potential - opponent_potential

    def negamax(self, P, alpha, beta, depth=0):
        assert alpha < beta
        self.node_count += 1

        # Kiểm tra điều kiện dừng theo độ sâu
        if self.max_depth is not None and depth >= self.max_depth:
            return self.evaluate(P)

        possible = P.possible_Non_Losing_Moves()
        if possible == 0:
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
            if val > Position.MAX_SCORE - Position.MIN_SCORE + 1:  # lower bound
                min_val = val + 2*Position.MIN_SCORE - Position.MAX_SCORE - 2
                if alpha < min_val:
                    alpha = min_val
                    if alpha >= beta:
                        return alpha
            else:  # upper bound
                max_val = val + Position.MIN_SCORE - 1
                if beta > max_val:
                    beta = max_val
                    if alpha >= beta:
                        return beta

        moves = MoveSorter()
        for i in reversed(range(Position.WIDTH)):
            move = possible & Position.column_mask(self.column_order[i])
            if move != 0:
                moves.add(move, P.moveScore(move))

        best_score = -float('inf')
        while True:
            next_move = moves.get_next()
            if next_move == 0:
                break

            P2 = Position(P)
            P2.play(next_move)
            score = -self.negamax(P2, -beta, -alpha, depth + 1)

            if score >= beta:
                # Đảm bảo giá trị nằm trong phạm vi cho phép trước khi lưu
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

        # Đảm bảo giá trị nằm trong phạm vi cho phép trước khi lưu
        value_to_store = alpha - Position.MIN_SCORE + 1
        max_allowed = (1 << self.transposition_table.value_size) - 1
        if value_to_store > max_allowed:
            value_to_store = max_allowed
        elif value_to_store < 0:
            value_to_store = 0
        self.transposition_table.put(key, value_to_store)
        
        return best_score
    def search_at_depth(self, P,depth):

        if P.canWinNext():
            return (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        min_score = -(Position.WIDTH * Position.HEIGHT - P.nb_moves()) // 2
        max_score = (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        while min_score < max_score:
            med = min_score + (max_score - min_score) // 2

            # Điều chỉnh điểm giữa để ưu tiên test vùng gần 0
            if med <= 0 and min_score // 2 < med:
                med = min_score // 2
            elif med >= 0 and max_score // 2 > med:
                med = max_score // 2

            # Dùng null-window để kiểm tra xem điểm thực tế lớn hơn hay nhỏ hơn `med`
            score = self.negamax(P, med, med + 1)

            if score <= med:
                max_score = score
            else:
                min_score = score

        return min_score

    def set_max_depth(self, depth):
        """Thiết lập độ sâu tối đa cho thuật toán"""
        self.max_depth = depth

def is_prime(n: int) -> bool:
    """Kiểm tra số nguyên tố"""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def next_prime(n: int) -> int:
    """Tìm số nguyên tố nhỏ nhất >= n"""
    while not is_prime(n):
        n += 1
    return n

def log2(n: int) -> int:
    """Tính logarit cơ số 2"""
    return 0 if n <= 1 else log2(n // 2) + 1

class TranspositionTable:
    def __init__(self, key_size: int = 64, value_size: int = 16, log_size: int = 20):
        """
        Khởi tạo bảng băm
        :param key_size: số bit của khóa (tối đa 64)
        :param value_size: số bit của giá trị (tối đa 64)
        :param log_size: log2 của kích thước bảng
        """
        assert key_size <= 64, "key_size quá lớn"
        assert value_size <= 64, "value_size quá lớn"
        assert log_size <= 64, "log_size quá lớn"

        self.key_size = key_size
        self.value_size = value_size
        self.log_size = log_size

        # Tính kích thước bảng (số nguyên tố)
        self.size = next_prime(1 << log_size)
        
        # Xác định kiểu dữ liệu phù hợp
        self.key_t = self._get_uint_type(key_size - log_size)
        self.value_t = self._get_uint_type(value_size)
        
        # Tạo mảng lưu trữ
        self.K = np.zeros(self.size, dtype=self.key_t)
        self.V = np.zeros(self.size, dtype=self.value_t)

    def _get_uint_type(self, bits: int) -> type:
        """Xác định kiểu numpy phù hợp cho số bit"""
        if bits <= 8:
            return np.uint8
        elif bits <= 16:
            return np.uint16
        elif bits <= 32:
            return np.uint32
        return np.uint64

    def index(self, key: int) -> int:
        """Hàm băm - trả về vị trí trong bảng"""
        return key % self.size

    def reset(self) -> None:
        """Đặt lại bảng về trạng thái ban đầu"""
        self.K.fill(0)
        self.V.fill(0)

    def put(self, key: int, value: int) -> None:
        """Thêm cặp key-value vào bảng"""
        assert key >> self.key_size == 0, "Key vượt quá kích thước bit quy định"
        assert value >> self.value_size == 0, "Value vượt quá kích thước bit quy định"
        
        pos = self.index(key)
        self.K[pos] = key  # Lưu key (có thể bị cắt bớt nếu key_t nhỏ hơn key_size)
        self.V[pos] = value

    def get(self, key: int) -> int:
        """Lấy giá trị từ bảng bằng key"""
        assert key >> self.key_size == 0, "Key vượt quá kích thước bit quy định"
        
        pos = self.index(key)
        return int(self.V[pos]) if self.K[pos] == (self.key_t)(key) else 0

    def __del__(self):
        """Hủy bảng khi đối tượng bị xóa"""
        if hasattr(self, 'K'):
            del self.K
        if hasattr(self, 'V'):
            del self.V

def convert_to_bitboard(board: List[List[int]], current_player: int):
    WIDTH, HEIGHT = 7, 6
    mask = np.uint64(0)
    current = np.uint64(0)
    moves = 0

    for col in range(WIDTH):
        for row in range(HEIGHT):
            if board[col][row] != 0:  # Ô không trống
                bit = col * (HEIGHT + 1) + row
                mask |= (1 << bit)
                if board[col][row] == current_player:
                    current |= (1 << bit)
                moves += 1

    return mask, current, moves

def best_move(position : Position, valid_moves: List[int], solver: Solver):
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
            
            # Kiểm tra nếu người chơi có thể thắng ở nước tiếp theo
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
    
    # Nếu không tìm được nước đi tốt, chọn nước đi hợp lệ đầu tiên
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

        position.mask, position.current_position, position.moves = convert_to_bitboard(game_state.board, game_state.current_player)
         
        selected_move = best_move(position, game_state.valid_moves,solver)

        return AIResponse(move=selected_move)
    except Exception as e:
        print(f"Error: {str(e)}")
        if game_state.valid_moves:
            return AIResponse(move=random.choice(game_state.valid_moves))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)