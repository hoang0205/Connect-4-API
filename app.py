from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import random
import math
import time
from functools import lru_cache
from typing import List

app = FastAPI()

ROWS = 6
COLS = 7
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4
MAX_DEPTH = 10  # Độ sâu tối đa có thể thử
TIME_LIMIT = 9.0  # Giới hạn thời gian là 9 giây

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int
    thinking_time: float = 0.0

# Hàm tính điểm cho một cửa sổ 4 ô liên tiếp
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4
    return score

# Hàm tính điểm tổng thể của bảng
@lru_cache(maxsize=1024)
def score_position_cached(board_bytes, piece):
    board = np.frombuffer(board_bytes, dtype=int).reshape(ROWS, COLS).tolist()
    score = 0
    # Tính điểm các hàng
    for r in range(ROWS):
        row_array = board[r]
        for c in range(COLS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)
    # Tính điểm các cột
    for c in range(COLS):
        col_array = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)
    # Tính điểm đường chéo chính
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    # Tính điểm đường chéo phụ
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            window = [board[r + i][c - i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    return score

# Thả quân cờ vào bảng
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Kiểm tra vị trí hợp lệ
def is_valid_location(board, col):
    return board[0][col] == 0

# Lấy hàng trống tiếp theo trong cột
def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

# Lấy danh sách các cột hợp lệ
def get_valid_locations(board):
    return [col for col in range(COLS) if is_valid_location(board, col)]

# Kiểm tra chiến thắng
def winning_move(board, piece):
    # Kiểm tra hàng ngang
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(WINDOW_LENGTH)):
                return True
    # Kiểm tra cột dọc
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(WINDOW_LENGTH)):
                return True
    # Kiểm tra đường chéo chính
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(WINDOW_LENGTH)):
                return True
    # Kiểm tra đường chéo phụ
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i][c - i] == piece for i in range(WINDOW_LENGTH)):
                return True
    return False

# Kiểm tra trạng thái kết thúc trò chơi
def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# Sắp xếp các nước đi để tối ưu Alpha-Beta
def order_moves(board, valid_locations, piece):
    move_scores = []
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = [r.copy() for r in board]
        drop_piece(temp_board, row, col, piece)
        board_bytes = np.array(temp_board).tobytes()
        score = score_position_cached(board_bytes, piece)
        move_scores.append((col, score))
    move_scores.sort(key=lambda x: x[1], reverse=True)
    return [move[0] for move in move_scores]

# Hàm Minimax với Alpha-Beta Pruning
def minimax(board, depth, alpha, beta, maximizing_player, start_time):
    if time.time() - start_time > TIME_LIMIT:
        raise TimeoutError("Hết thời gian suy nghĩ")

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 1_000_000_000
            elif winning_move(board, PLAYER_PIECE):
                return None, -1_000_000_000
            else:  # Hòa
                return None, 0
        else:
            board_bytes = np.array(board).astype(int).tobytes()
            return None, score_position_cached(board_bytes, AI_PIECE)

    if maximizing_player:
        ordered_cols = order_moves(board, valid_locations, AI_PIECE)
    else:
        ordered_cols = order_moves(board, valid_locations, PLAYER_PIECE)

    best_col = random.choice(valid_locations) if valid_locations else None

    if maximizing_player:
        value = -math.inf
        for col in ordered_cols:
            if time.time() - start_time > TIME_LIMIT:
                return best_col, value

            row = get_next_open_row(board, col)
            temp_board = [r.copy() for r in board]
            drop_piece(temp_board, row, col, AI_PIECE)

            try:
                _, new_score = minimax(temp_board, depth - 1, alpha, beta, False, start_time)
            except TimeoutError:
                return best_col, value

            if new_score > value:
                value = new_score
                best_col = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

    else:  # minimizing player
        value = math.inf
        for col in ordered_cols:
            if time.time() - start_time > TIME_LIMIT:
                return best_col, value

            row = get_next_open_row(board, col)
            temp_board = [r.copy() for r in board]
            drop_piece(temp_board, row, col, PLAYER_PIECE)

            try:
                _, new_score = minimax(temp_board, depth - 1, alpha, beta, True, start_time)
            except TimeoutError:
                return best_col, value

            if new_score < value:
                value = new_score
                best_col = col

            beta = min(beta, value)
            if beta <= alpha:
                break

    return best_col, value

# Endpoint để thực hiện nước đi với Iterative Deepening
@app.post("/api/connect4-move11")
async def make_move(game_state: GameState) -> AIResponse:
    board = game_state.board
    if not game_state.valid_moves:
        raise HTTPException(status_code=400, detail="Không có nước đi hợp lệ")

    start_time = time.time()
    best_move = random.choice(game_state.valid_moves)
    best_score = -math.inf
    depth = 1

    while depth <= MAX_DEPTH and (time.time() - start_time) < TIME_LIMIT:
        try:
            move, score = minimax(board, depth, -math.inf, math.inf, True, start_time)
            if move is not None and move in game_state.valid_moves:
                if score > best_score:
                    best_score = score
                    best_move = move
        except TimeoutError:
            break  # Dừng nếu hết thời gian
        depth += 1

    thinking_time = time.time() - start_time
    return AIResponse(move=best_move, thinking_time=thinking_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
