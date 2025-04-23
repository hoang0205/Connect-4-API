from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import random
import math
import time
from functools import lru_cache

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}


ROWS = 6
COLS = 7
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4
DIFFICULTY_DEPTH = 6
TIME_LIMIT = 9.5  # Giới hạn thời gian thực sự là 9.5s để dự phòng


class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]


class AIResponse(BaseModel):
    move: int
    thinking_time: float = 0.0


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[0][col] == 0


def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1


def get_valid_locations(board):
    return [col for col in range(COLS) if is_valid_location(board, col)]


def winning_move(board, piece):
    # Kiểm tra hàng ngang
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c + i] == piece for i in range(4)):
                return True

    # Kiểm tra hàng dọc
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(4)):
                return True

    # Kiểm tra đường chéo chính (/)
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True

    # Kiểm tra đường chéo phụ (\)
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True

    return False


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 10000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 100
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10
    elif window.count(piece) == 1 and window.count(EMPTY) == 3:
        score += 1

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 120
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 5

    return score


def score_position(board, piece):
    score = 0

    # Ưu tiên cột giữa
    center_array = [int(i) for i in list(np.array(board)[:, COLS // 2])]
    center_count = center_array.count(piece)
    score += center_count * 8

    # Đánh giá hàng ngang
    for r in range(ROWS):
        row_array = [int(i) for i in board[r]]
        for c in range(COLS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Đánh giá hàng dọc
    for c in range(COLS):
        col_array = [int(i) for i in list(np.array(board)[:, c])]
        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Đánh giá đường chéo chính (/)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Đánh giá đường chéo phụ (\)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


@lru_cache(maxsize=None)
def score_position_cached(board_bytes, piece):
    board = np.frombuffer(board_bytes, dtype=int).reshape((ROWS, COLS))
    return score_position(board.tolist(), piece)


def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(
        get_valid_locations(board)) == 0


def order_moves(board, valid_locations, piece):
    """Sắp xếp các nước đi theo thứ tự từ tốt nhất đến kém nhất"""
    scored_moves = []

    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = [row.copy() for row in board]
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        scored_moves.append((score, col))

    # Sắp xếp giảm dần theo điểm số
    scored_moves.sort(reverse=True)
    return [col for score, col in scored_moves]


def minimax(board, depth, alpha, beta, maximizing_player, start_time):
    """Thuật toán minimax với kiểm tra thời gian chặt chẽ"""

    # Kiểm tra thời gian sau mỗi nút được đánh giá
    if time.time() - start_time > TIME_LIMIT:
        raise TimeoutError("Hết thời gian suy nghĩ")

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    # Điều kiện dừng
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

    # Sắp xếp các nước đi để tối ưu alpha-beta pruning
    if maximizing_player:
        ordered_cols = order_moves(board, valid_locations, AI_PIECE)
    else:
        ordered_cols = order_moves(board, valid_locations, PLAYER_PIECE)

    best_col = random.choice(valid_locations) if valid_locations else None

    if maximizing_player:
        value = -math.inf
        for col in ordered_cols:
            # Kiểm tra thời gian
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
            # Kiểm tra thời gian
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


@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    board = game_state.board
    if not game_state.valid_moves:
        raise HTTPException(status_code=400, detail="Không có nước đi hợp lệ")

    # Luôn có nước đi mặc định nếu minimax không kịp trả về
    best_move = random.choice(game_state.valid_moves)
    start_time = time.time()

    try:
        # Thực hiện minimax với giới hạn thời gian chặt chẽ
        move, _ = minimax(board, DIFFICULTY_DEPTH, -math.inf, math.inf, True, start_time)
        if move is not None and move in game_state.valid_moves:
            best_move = move
    except TimeoutError:
        print("⚠️ Hết thời gian cho Minimax")
    except Exception as e:
        print(f"Lỗi trong quá trình tìm kiếm: {e}")

    thinking_time = time.time() - start_time

    return AIResponse(move=best_move, thinking_time=thinking_time)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)