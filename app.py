rom fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import random
import math
import time
from functools import lru_cache
from typing import List

app = FastAPI()

# Game constants
ROWS = 6
COLS = 7
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4
MAX_DEPTH = 10
TIME_LIMIT = 9.0
TT_SIZE_LIMIT = 1_000_000  # Giới hạn kích thước Transposition Table
MIN_TT_DEPTH = 4  # Chỉ lưu trạng thái ở độ sâu >= 4

# Initialize Zobrist hash table
ZOBRIST_TABLE = np.random.randint(1, 2**64 - 1, size=(ROWS, COLS, 3), dtype=np.uint64)

# Transposition Table
TRANSPOSITION_TABLE = {}

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int
    thinking_time: float = 0.0

# Compute Zobrist hash for the current board state
def compute_zobrist_hash(board):
    hash_value = np.uint64(0)
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece != EMPTY:
                piece_index = 0 if piece == PLAYER_PIECE else 1
                hash_value ^= ZOBRIST_TABLE[r, c, piece_index]
    return hash_value

# Evaluate a window of 4 consecutive slots
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 1000  # Tăng điểm cho chiến thắng
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 50  # Tăng điểm cho cơ hội thắng gần
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10  # Điểm cho cơ hội tiềm năng
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 100  # Điểm phạt nặng hơn cho cơ hội thắng của đối thủ
    return score

# Cached scoring function for the board
@lru_cache(maxsize=1024)
def score_position_cached(board_bytes, piece):
    board = np.frombuffer(board_bytes, dtype=int).reshape(ROWS, COLS).tolist()
    score = 0
    # Horizontal
    for r in range(ROWS):
        row_array = board[r]
        for c in range(COLS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)
    # Vertical
    for c in range(COLS):
        col_array = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)
    # Positive diagonal
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    # Negative diagonal
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            window = [board[r + i][c - i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    return score

# Drop a piece and update Zobrist hash
def drop_piece(board, row, col, piece, current_hash):
    board[row][col] = piece
    piece_index = 0 if piece == PLAYER_PIECE else 1
    new_hash = current_hash ^ ZOBRIST_TABLE[row, col, piece_index]
    return new_hash

# Check if a column is a valid move
def is_valid_location(board, col):
    return board[0][col] == 0

# Get the next open row in a column
def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

# Get all valid columns
def get_valid_locations(board):
    return [col for col in range(COLS) if is_valid_location(board, col)]

# Check for a winning move
def winning_move(board, piece):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(WINDOW_LENGTH)):
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(WINDOW_LENGTH)):
                return True
    # Positive diagonal
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(WINDOW_LENGTH)):
                return True
    # Negative diagonal
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i][c - i] == piece for i in range(WINDOW_LENGTH)):
                return True
    return False

# Check if the game has ended
def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# Order moves for better Alpha-Beta pruning
def order_moves(board, valid_locations, piece, current_hash):
    move_scores = []
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = [r.copy() for r in board]
        new_hash = drop_piece(temp_board, row, col, piece, current_hash)
        board_bytes = np.array(temp_board).tobytes()
        score = score_position_cached(board_bytes, piece)
        move_scores.append((col, score))
    move_scores.sort(key=lambda x: x[1], reverse=True)
    return [move[0] for move in move_scores]

# Minimax with Alpha-Beta pruning and Transposition Table
def minimax(board, depth, alpha, beta, maximizing_player, start_time, current_hash):
    if time.time() - start_time > TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")

    # Check Transposition Table
    hash_key = current_hash
    if hash_key in TRANSPOSITION_TABLE:
        tt_entry = TRANSPOSITION_TABLE[hash_key]
        if tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'exact':
                return tt_entry['best_move'], tt_entry['score']
            elif tt_entry['flag'] == 'lower_bound':
                alpha = max(alpha, tt_entry['score'])
            elif tt_entry['flag'] == 'upper_bound':
                beta = min(beta, tt_entry['score'])
            if alpha >= beta:
                return tt_entry['best_move'], tt_entry['score']

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 1_000_000_000
            elif winning_move(board, PLAYER_PIECE):
                return None, -1_000_000_000
            else:
                return None, 0
        else:
            board_bytes = np.array(board).astype(int).tobytes()
            return None, score_position_cached(board_bytes, AI_PIECE)

    if maximizing_player:
        ordered_cols = order_moves(board, valid_locations, AI_PIECE, current_hash)
    else:
        ordered_cols = order_moves(board, valid_locations, PLAYER_PIECE, current_hash)

    best_col = random.choice(valid_locations) if valid_locations else None
    best_score = -math.inf if maximizing_player else math.inf

    for col in ordered_cols:
        if time.time() - start_time > TIME_LIMIT:
            break

        row = get_next_open_row(board, col)
        temp_board = [r.copy() for r in board]
        piece = AI_PIECE if maximizing_player else PLAYER_PIECE
        new_hash = drop_piece(temp_board, row, col, piece, current_hash)

        try:
            _, new_score = minimax(temp_board, depth - 1, alpha, beta, not maximizing_player, start_time, new_hash)
        except TimeoutError:
            break

        if maximizing_player:
            if new_score > best_score:
                best_score = new_score
                best_col = col
            alpha = max(alpha, best_score)
        else:
            if new_score < best_score:
                best_score = new_score
                best_col = col
            beta = min(beta, best_score)

        if alpha >= beta:
            break

    # Store in Transposition Table only if depth >= MIN_TT_DEPTH
    if depth >= MIN_TT_DEPTH and len(TRANSPOSITION_TABLE) < TT_SIZE_LIMIT:
        if best_score <= alpha:
            flag = 'upper_bound'
        elif best_score >= beta:
            flag = 'lower_bound'
        else:
            flag = 'exact'
        TRANSPOSITION_TABLE[hash_key] = {
            'depth': depth,
            'score': best_score,
            'flag': flag,
            'best_move': best_col
        }

    return best_col, best_score

# API endpoint with Iterative Deepening starting from depth 4
@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    board = game_state.board
    if not game_state.valid_moves:
        raise HTTPException(status_code=400, detail="No valid moves available")

    start_time = time.time()
    best_move = random.choice(game_state.valid_moves)
    best_score = -math.inf
    depth = 4  # Bắt đầu từ độ sâu 4
    current_hash = compute_zobrist_hash(board)

    while depth <= MAX_DEPTH and (time.time() - start_time) < TIME_LIMIT:
        try:
            move, score = minimax(board, depth, -math.inf, math.inf, True, start_time, current_hash)
            if move is not None and move in game_state.valid_moves:
                if score > best_score:
                    best_score = score
                    best_move = move
        except TimeoutError:
            break
        depth += 1

    thinking_time = time.time() - start_time
    return AIResponse(move=best_move, thinking_time=thinking_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
