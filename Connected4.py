import pygame
import sys
import numpy as np
import time
import math
from pygame.locals import *

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 600
BOARD_WIDTH = 7
BOARD_HEIGHT = 6
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE/2 - 5)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Game variables
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Connect 4")
font = pygame.font.SysFont("comicsansms", 40)
small_font = pygame.font.SysFont("comicsansms", 24)

def bottom(width, height):
    mask = np.uint64(0)
    for w in range(width):
        mask |= np.uint64(1) << (height + 1) * w
    return mask

class Position:
    WIDTH = BOARD_WIDTH
    HEIGHT = BOARD_HEIGHT
    MIN_SCORE = -(WIDTH*HEIGHT)//2 + 3
    MAX_SCORE = (WIDTH*HEIGHT+1)//2 - 3
    bottom_mask = bottom(WIDTH, HEIGHT)
    board_mask = bottom_mask * ((np.uint64(1) << HEIGHT)-1)

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

    def can_play(self, col):
        return (self.mask & self.top_mask_col(col)) == 0
    
    def playCol(self, col):
        return self.play((self.mask + self.bottom_mask_col(col)) & self.column_mask(col))
    
    def is_winning_move(self, col):
        return self.winning_position() & self.possible() & self.column_mask(col) != 0

    def winning_position(self):
        return Position.compute_winning_position(self.current_position, self.mask)

    def possible(self):
        return (self.mask + Position.bottom_mask) & Position.board_mask

    @staticmethod
    def compute_winning_position(position, mask):
        # Vertical
        r = (position << 1) & (position << 2) & (position << 3)

        # Horizontal
        p = (position << (Position.HEIGHT+1)) & (position << 2*(Position.HEIGHT+1))
        r |= p & (position << 3*(Position.HEIGHT+1))
        r |= p & (position >> (Position.HEIGHT+1))
        p = (position >> (Position.HEIGHT+1)) & (position >> 2*(Position.HEIGHT+1))
        r |= p & (position << (Position.HEIGHT+1))
        r |= p & (position >> 3*(Position.HEIGHT+1))

        # Diagonal 1
        p = (position << Position.HEIGHT) & (position << 2*Position.HEIGHT)
        r |= p & (position << 3*Position.HEIGHT)
        r |= p & (position >> Position.HEIGHT)
        p = (position >> Position.HEIGHT) & (position >> 2*Position.HEIGHT)
        r |= p & (position << Position.HEIGHT)
        r |= p & (position >> 3*Position.HEIGHT)

        # Diagonal 2
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

    def nb_moves(self):
        return self.moves

    def key(self):
        return self.current_position + self.mask

class Solver:
    def __init__(self):
        self.node_count = 0
        self.column_order = [3, 2, 4, 1, 5, 0, 6]  # Center columns first

    def negamax(self, P, alpha, beta):
        self.node_count += 1

        # Check for draw game
        if P.nb_moves() == Position.WIDTH * Position.HEIGHT:
            return 0

        # Check if current player can win next move
        for x in range(Position.WIDTH):
            col = self.column_order[x]
            if P.can_play(col) and P.is_winning_move(col):
                return (Position.WIDTH * Position.HEIGHT + 1 - P.nb_moves()) // 2

        max_score = (Position.WIDTH * Position.HEIGHT - 1 - P.nb_moves()) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta

        for x in range(Position.WIDTH):
            col = self.column_order[x]
            if P.can_play(col):
                P2 = Position(P)
                P2.playCol(col)
                score = -self.negamax(P2, -beta, -alpha)

                if score >= beta:
                    return score

                if score > alpha:
                    alpha = score

        return alpha

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

def draw_board(board):
    # Draw blue board
    pygame.draw.rect(screen, BLUE, (0, SQUARE_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT - SQUARE_SIZE))
    
    # Draw black circles for empty slots
    for c in range(BOARD_WIDTH):
        for r in range(BOARD_HEIGHT):
            pygame.draw.circle(screen, BLACK, 
                             (int(c * SQUARE_SIZE + SQUARE_SIZE/2), 
                             int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE/2)), 
                             RADIUS)
    
    # Draw pieces
    for c in range(BOARD_WIDTH):
        for r in range(BOARD_HEIGHT):
            if board[r][c] == 1:  # Player 1 (Red)
                pygame.draw.circle(screen, RED, 
                                 (int(c * SQUARE_SIZE + SQUARE_SIZE/2), 
                                 int((BOARD_HEIGHT - r) * SQUARE_SIZE + SQUARE_SIZE/2)), 
                                 RADIUS)
            elif board[r][c] == 2:  # Player 2 (Yellow)
                pygame.draw.circle(screen, YELLOW, 
                                 (int(c * SQUARE_SIZE + SQUARE_SIZE/2), 
                                 int((BOARD_HEIGHT - r) * SQUARE_SIZE + SQUARE_SIZE/2)), 
                                 RADIUS)
    
    pygame.display.update()

def create_board():
    return np.zeros((BOARD_HEIGHT, BOARD_WIDTH))

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[BOARD_HEIGHT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(BOARD_HEIGHT):
        if board[r][col] == 0:
            return r

def winning_move(board, piece):
    # Check horizontal locations
    for c in range(BOARD_WIDTH-3):
        for r in range(BOARD_HEIGHT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations
    for c in range(BOARD_WIDTH):
        for r in range(BOARD_HEIGHT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(BOARD_WIDTH-3):
        for r in range(BOARD_HEIGHT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(BOARD_WIDTH-3):
        for r in range(3, BOARD_HEIGHT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)

def main_menu():
    while True:
        screen.fill(BLACK)
        draw_text("CONNECT 4", font, WHITE, SCREEN_WIDTH/2, 100)
        draw_text("1. Player vs AI", small_font, WHITE, SCREEN_WIDTH/2, 250)
        draw_text("2. Player vs Player", small_font, WHITE, SCREEN_WIDTH/2, 300)
        draw_text("3. Exit", small_font, WHITE, SCREEN_WIDTH/2, 350)
        
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == KEYDOWN:
                if event.key == K_1:
                    game_loop(True)
                if event.key == K_2:
                    game_loop(False)
                if event.key == K_3:
                    pygame.quit()
                    sys.exit()

def game_loop(ai_opponent):
    board = create_board()
    game_over = False
    turn = 0  # 0 for Player 1, 1 for Player 2 or AI
    
    position = Position()
    solver = Solver()
    
    draw_board(board)
    pygame.display.update()
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == MOUSEMOTION and not game_over:
                pygame.draw.rect(screen, BLACK, (0, 0, SCREEN_WIDTH, SQUARE_SIZE))
                posx = event.pos[0]
                if turn == 0:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARE_SIZE/2)), RADIUS)
                else:
                    if not ai_opponent:
                        pygame.draw.circle(screen, YELLOW, (posx, int(SQUARE_SIZE/2)), RADIUS)
                pygame.display.update()
            
            if event.type == MOUSEBUTTONDOWN and not game_over:
                pygame.draw.rect(screen, BLACK, (0, 0, SCREEN_WIDTH, SQUARE_SIZE))
                
                # Ask for Player 1 Input
                if turn == 0:
                    posx = event.pos[0]
                    col = int(posx // SQUARE_SIZE)
                    
                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, 1)
                        
                        position.playCol(col)
                        
                        if winning_move(board, 1):
                            draw_text("Player 1 wins!", font, RED, SCREEN_WIDTH/2, SQUARE_SIZE/2)
                            game_over = True
                
                # Ask for Player 2 Input or AI move
                elif turn == 1:
                    if ai_opponent:
                        # AI move
                        start_time = time.time()
                        best_col = None
                        best_score = -float('inf')
                        
                        # Check if AI can win in the next move
                        for col in range(BOARD_WIDTH):
                            if is_valid_location(board, col):
                                row = get_next_open_row(board, col)
                                temp_board = board.copy()
                                drop_piece(temp_board, row, col, 2)
                                if winning_move(temp_board, 2):
                                    best_col = col
                                    break
                        
                        if best_col is None:
                            # Find best move using solver
                            for x in range(Position.WIDTH):
                                col = solver.column_order[x]
                                if position.can_play(col):
                                    P2 = Position(position)
                                    P2.playCol(col)
                                    score = -solver.solve(P2)
                                    if score > best_score or best_col is None:
                                        best_col = col
                                        best_score = score
                        
                        # Make sure we have a valid move
                        if best_col is None:
                            for col in range(BOARD_WIDTH):
                                if is_valid_location(board, col):
                                    best_col = col
                                    break
                        
                        if is_valid_location(board, best_col):
                            row = get_next_open_row(board, best_col)
                            drop_piece(board, row, best_col, 2)
                            
                            position.playCol(best_col)
                            
                            if winning_move(board, 2):
                                draw_text("AI wins!", font, YELLOW, SCREEN_WIDTH/2, SQUARE_SIZE/2)
                                game_over = True
                    else:
                        # Player 2 move
                        posx = event.pos[0]
                        col = int(posx // SQUARE_SIZE)
                        
                        if is_valid_location(board, col):
                            row = get_next_open_row(board, col)
                            drop_piece(board, row, col, 2)
                            
                            position.playCol(col)
                            
                            if winning_move(board, 2):
                                draw_text("Player 2 wins!", font, YELLOW, SCREEN_WIDTH/2, SQUARE_SIZE/2)
                                game_over = True
                
                draw_board(board)
                
                turn += 1
                turn = turn % 2
                
                if game_over:
                    pygame.time.wait(3000)
                    main_menu()
        
        # If it's AI's turn and we're in AI mode
        if ai_opponent and turn == 1 and not game_over:
            # AI move
            start_time = time.time()
            best_col = None
            best_score = -float('inf')
            
            # Check if AI can win in the next move
            for col in range(BOARD_WIDTH):
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    temp_board = board.copy()
                    drop_piece(temp_board, row, col, 2)
                    if winning_move(temp_board, 2):
                        best_col = col
                        break
            
            if best_col is None:
                # Find best move using solver
                for x in range(Position.WIDTH):
                    col = solver.column_order[x]
                    if position.can_play(col):
                        P2 = Position(position)
                        P2.playCol(col)
                        score = -solver.solve(P2)
                        if score > best_score or best_col is None:
                            best_col = col
                            best_score = score
            
            # Make sure we have a valid move
            if best_col is None:
                for col in range(BOARD_WIDTH):
                    if is_valid_location(board, col):
                        best_col = col
                        break
            
            if is_valid_location(board, best_col):
                row = get_next_open_row(board, best_col)
                drop_piece(board, row, best_col, 2)
                
                position.playCol(best_col)
                
                if winning_move(board, 2):
                    draw_text("AI wins!", font, YELLOW, SCREEN_WIDTH/2, SQUARE_SIZE/2)
                    game_over = True
            
            draw_board(board)
            
            turn += 1
            turn = turn % 2
            
            if game_over:
                pygame.time.wait(3000)
                main_menu()

if __name__ == "__main__":
    main_menu()