import numpy as np
import time
import sys
import math

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
    def solve(self, P, weak=False):

        if P.canWinNext():
            return (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        min_score = -(Position.WIDTH * Position.HEIGHT - P.nb_moves()) // 2
        max_score = (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        if weak:
            min_score = -1
            max_score = 1

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

from typing import TypeVar, Union

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

import struct

class OpeningBook:
    def __init__(self, book_file=None):
        self.width = 7
        self.height = 6
        self.depth = -1
        self.book = {}
        if book_file:
            self.load_book(book_file)
    
    def load_book(self, filename):
        try:
            with open(filename, 'rb') as f:
                # Đọc header (6 bytes)
                header = f.read(6)
                if len(header) != 6:
                    raise ValueError("Invalid header size")
                
                width, height, depth, key_bytes, value_bytes, log_size = struct.unpack('BBBBBB', header)
                
                # Validate header
                if width != self.width or height != self.height:
                    raise ValueError(f"Invalid board size in book: {width}x{height}, expected 7x6")
                
                if depth > self.width * self.height:
                    raise ValueError(f"Invalid depth in book: {depth}")
                
                if key_bytes > 8 or value_bytes != 1:
                    raise ValueError(f"Invalid key/value size: key={key_bytes}, value={value_bytes}")
                
                # Tính toán kích thước dữ liệu
                size = self._next_prime(1 << log_size)
                key_size = size * key_bytes
                value_size = size * value_bytes
                
                # Đọc keys và values
                keys = f.read(key_size)
                values = f.read(value_size)
                
                if len(keys) != key_size or len(values) != value_size:
                    raise ValueError("Invalid data size")
                
                # Xử lý dữ liệu tùy thuộc vào key_bytes
                if key_bytes == 4:
                    fmt = f'<{size}I'  # unsigned int 32-bit
                elif key_bytes == 8:
                    fmt = f'<{size}Q'  # unsigned long long 64-bit
                else:
                    raise ValueError(f"Unsupported key size: {key_bytes} bytes")
                
                keys = struct.unpack(fmt, keys)
                values = struct.unpack(f'{size}B', values)  # values là 1 byte mỗi entry
                
                # Tạo dictionary để tra cứu nhanh
                self.book = {k: v for k, v in zip(keys, values) if v != 0}
                self.depth = depth
                
                print(f"Loaded opening book with {len(self.book)} positions (max depth: {self.depth})")
                
        except Exception as e:
            print(f"Error loading opening book: {e}")
            self.book = {}
            self.depth = -1
    
    def _next_prime(self, n):
        """Tìm số nguyên tố nhỏ nhất >= n"""
        if n <= 2:
            return 2
        if n % 2 == 0:
            n += 1
        while not self._is_prime(n):
            n += 2
        return n
    
    def _is_prime(self, n):
        """Kiểm tra số nguyên tố"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def get(self, position):
        """Lấy nước đi từ opening book nếu có"""
        if position.nb_moves() > self.depth:
            return None
        
        key = position.key3()
        move = self.book.get(key)
        if move is not None and move != 0:
            return move - 1  # Chuyển từ 1-7 sang 0-6
        return None

def human_vs_ai():
    """Chế độ chơi người vs AI"""
    position = Position()
    solver = Solver()
    
    while True:
        position.print_board()
        
        # Lượt người chơi
        if position.nb_moves() % 2 == 0:
            print("Lượt của bạn (X), nhập cột (1-7):")
            try:
                col = int(input()) - 1
                if col < 0 or col >= Position.WIDTH:
                    print("Vui lòng nhập số từ 1 đến 7!")
                    continue
                if not position.can_play(col):
                    print("Cột này đã đầy!")
                    continue
                if position.is_winning_move(col):
                    position.playCol(col)
                    position.print_board()
                    print("Bạn đã thắng!")
                    break
                position.playCol(col)
            except ValueError:
                print("Vui lòng nhập số!")
                continue
        # Lượt AI
        else:
            print("AI đang suy nghĩ...")
            start_time = time.time()
            best_col = None
            best_score = -float('inf')
            
            # Kiểm tra nếu AI có thể thắng ngay
            for col in range(Position.WIDTH):
                if position.can_play(col) and position.is_winning_move(col):
                    best_col = col
                    position.playCol(best_col)
                    position.print_board()
                    print("AI đã thắng!")
                    print(f"Thời gian suy nghĩ: {time.time() - start_time:.2f} giây")
                    return
            
            # Nếu không có nước thắng ngay, tìm nước tốt nhất
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
            
            position.playCol(best_col)
            end_time = time.time()
            print(f"Thời gian suy nghĩ: {end_time - start_time:.2f} giây")
            print(f"AI chọn cột {best_col+1}")
            
            # Kiểm tra nếu AI vừa thắng
            if position.is_winning_move(best_col):
                position.print_board()
                print("AI đã thắng!")
                return

if __name__ == "__main__":
    human_vs_ai()


#if _name_ == "_main_":
 #   test_positions = [
  #      274552224131661,
   #     5455174361263362,
    #   37313333717124171162542,
      #  6614446666373154,
     #   24617524315172127,
    #]

    #for pos in test_positions:
     #   compare(pos)