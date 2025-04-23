from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import random
import math
import time
from functools import lru_cache
import concurrent.futures
from dataclasses import dataclass

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


# Các hằng số
ROWS = 6
COLS = 7
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4
MAX_DEPTH = 10  # Độ sâu tối đa cho iterative deepening
TIME_LIMIT = 9.5  # Giới hạn thời gian (giây)

# Các trọng số đánh giá
CENTER_WEIGHT = 8
THREE_IN_A_ROW_WEIGHT = 100
TWO_IN_A_ROW_WEIGHT = 10
ONE_IN_A_ROW_WEIGHT = 1
OPPONENT_THREE_WEIGHT = -120
OPPONENT_TWO_WEIGHT = -5

# Bảng trans có sẵn (transposition table)
transposition_table = {}


class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    difficulty: Optional[int] = 6  # 1-10, mặc định là 5


class AIResponse(BaseModel):
    move: int
    depth_reached: int
    positions_evaluated: int
    thinking_time: float


@dataclass
class MoveInfo:
    col: int
    score: int
    depth_reached: int
    positions_evaluated: int


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
        score += THREE_IN_A_ROW_WEIGHT
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += TWO_IN_A_ROW_WEIGHT
    elif window.count(piece) == 1 and window.count(EMPTY) == 3:
        score += ONE_IN_A_ROW_WEIGHT

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score += OPPONENT_THREE_WEIGHT
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score += OPPONENT_TWO_WEIGHT

    return score


def score_position(board, piece):
    score = 0

    # Ưu tiên cột giữa
    center_array = [int(i) for i in list(np.array(board)[:, COLS // 2])]
    center_count = center_array.count(piece)
    score += center_count * CENTER_WEIGHT

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


def board_to_key(board):
    """Chuyển đổi bảng thành khóa có thể hash để lưu vào bảng trans"""
    return tuple(tuple(row) for row in board)


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


def minimax(board, depth, alpha, beta, maximizing_player, start_time, positions_evaluated):
    """Thuật toán minimax với alpha-beta pruning và kiểm tra thời gian"""

    # Kiểm tra thời gian mỗi 100 nút được đánh giá
    positions_evaluated[0] += 1
    if positions_evaluated[0] % 100 == 0 and time.time() - start_time > TIME_LIMIT:
        raise TimeoutError("Hết thời gian suy nghĩ")

    # Kiểm tra bảng trans
    board_key = board_to_key(board)
    if board_key in transposition_table and transposition_table[board_key][0] >= depth:
        stored_depth, stored_value, stored_move = transposition_table[board_key]
        if stored_depth >= depth:
            return stored_move, stored_value

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    # Điều kiện dừng
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 1000000
            elif winning_move(board, PLAYER_PIECE):
                return None, -1000000
            else:  # Hòa
                return None, 0
        else:
            return None, score_position(board, AI_PIECE)

    # Sắp xếp các nước đi để tối ưu alpha-beta pruning
    if maximizing_player:
        ordered_cols = order_moves(board, valid_locations, AI_PIECE)
    else:
        ordered_cols = order_moves(board, valid_locations, PLAYER_PIECE)

    best_col = random.choice(valid_locations) if valid_locations else None

    if maximizing_player:
        value = -math.inf
        for col in ordered_cols:
            row = get_next_open_row(board, col)
            temp_board = [r.copy() for r in board]
            drop_piece(temp_board, row, col, AI_PIECE)

            _, new_score = minimax(temp_board, depth - 1, alpha, beta, False, start_time,
                                   positions_evaluated)

            if new_score > value:
                value = new_score
                best_col = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

    else:  # minimizing player
        value = math.inf
        for col in ordered_cols:
            row = get_next_open_row(board, col)
            temp_board = [r.copy() for r in board]
            drop_piece(temp_board, row, col, PLAYER_PIECE)

            _, new_score = minimax(temp_board, depth - 1, alpha, beta, True, start_time,
                                   positions_evaluated)

            if new_score < value:
                value = new_score
                best_col = col

            beta = min(beta, value)
            if beta <= alpha:
                break

    # Lưu kết quả vào bảng trans
    transposition_table[board_key] = (depth, value, best_col)

    return best_col, value


def iterative_deepening(board, max_depth, start_time):
    """Iterative deepening: tăng dần độ sâu tìm kiếm theo thời gian cho phép"""
    best_move = random.choice(get_valid_locations(board))
    best_score = -math.inf
    depth_reached = 0
    positions_evaluated = [0]  # Sử dụng list để có thể thay đổi giá trị bên trong hàm

    # Bắt đầu từ độ sâu 1, tăng dần lên
    for depth in range(1, max_depth + 1):
        try:
            # Nếu còn đủ thời gian, thực hiện minimax ở độ sâu hiện tại
            move, score = minimax(board, depth, -math.inf, math.inf, True, start_time,
                                  positions_evaluated)

            if move is not None:
                best_move = move
                best_score = score
                depth_reached = depth

            # Nếu đã tìm thấy nước thắng, không cần tìm kiếm sâu hơn
            if score >= 100000:
                break

        except TimeoutError:
            # Nếu hết thời gian, dừng tìm kiếm và trả về kết quả tốt nhất đã tìm được
            break

    return MoveInfo(
        col=best_move,
        score=best_score,
        depth_reached=depth_reached,
        positions_evaluated=positions_evaluated[0]
    )


@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    board = game_state.board
    if not game_state.valid_moves:
        raise HTTPException(status_code=400, detail="Không có nước đi hợp lệ")

    # Xóa bảng trans để tránh dùng kết quả cũ
    transposition_table.clear()

    # Điều chỉnh độ sâu tìm kiếm dựa trên độ khó
    max_depth = min(MAX_DEPTH, 1 + game_state.difficulty)

    start_time = time.time()
    best_move = random.choice(game_state.valid_moves)

    try:
        # Sử dụng iterative deepening để tận dụng tốt thời gian suy nghĩ
        move_info = iterative_deepening(board, max_depth, start_time)
        best_move = move_info.col
        depth_reached = move_info.depth_reached
        positions_evaluated = move_info.positions_evaluated
    except Exception as e:
        print(f"Lỗi trong quá trình tìm kiếm nước đi: {e}")
        depth_reached = 0
        positions_evaluated = 0

    thinking_time = time.time() - start_time

    # Kiểm tra xem nước đi có hợp lệ không
    if best_move not in game_state.valid_moves:
        best_move = random.choice(game_state.valid_moves)

    return AIResponse(
        move=best_move,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)