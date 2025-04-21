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
DIFFICULTY_DEPTH = 7

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

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
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r-i][c+i] == piece for i in range(4)):
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
    center_array = [int(i) for i in list(np.array(board)[:, COLS // 2])]
    center_count = center_array.count(piece)
    score += center_count * 8
    for r in range(ROWS):
        row_array = [int(i) for i in board[r]]
        for c in range(COLS - 3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)
    for c in range(COLS):
        col_array = [int(i) for i in list(np.array(board)[:, c])]
        for r in range(ROWS - 3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    return score

@lru_cache(maxsize=None)
def score_position_cached(board_bytes, piece):
    board = np.frombuffer(board_bytes, dtype=int).reshape((ROWS, COLS))
    return score_position(board.tolist(), piece)

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def order_moves(board, valid_locations, piece):
    scored_moves = []
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = [r.copy() for r in board]
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        scored_moves.append((score, col))
    scored_moves.sort(reverse=True)
    return [col for score, col in scored_moves]

def minimax(board, depth, alpha, beta, maximizing_player):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 1_000_000_000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -1_000_000_000)
            else:
                return (None, 0)
        else:
            board_bytes = np.array(board).astype(int).tobytes()
            return (None, score_position_cached(board_bytes, AI_PIECE))
    if maximizing_player:
        value = -math.inf
        best_col = random.choice(valid_locations)
        for col in order_moves(board, valid_locations, AI_PIECE):
            row = get_next_open_row(board, col)
            temp_board = [r.copy() for r in board]
            drop_piece(temp_board, row, col, AI_PIECE)
            new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = math.inf
        best_col = random.choice(valid_locations)
        for col in order_moves(board, valid_locations, PLAYER_PIECE):
            row = get_next_open_row(board, col)
            temp_board = [r.copy() for r in board]
            drop_piece(temp_board, row, col, PLAYER_PIECE)
            new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        board = game_state.board
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        selected_move, _ = minimax(board, DIFFICULTY_DEPTH, -math.inf, math.inf, True)
        return AIResponse(move=selected_move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
