from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import random
import time
import threading

app = FastAPI()

# Thông số game
ROWS, COLS = 6, 7
WINDOW_LENGTH = 4
MAX_DEPTH = 10
TIME_LIMIT = 7.5
TT_SIZE_LIMIT = 1_000_000
MIN_TT_DEPTH = 1

# Zobrist table: Python int, không cần NumPy
ZOBRIST_TABLE = [
    [ (random.getrandbits(64), random.getrandbits(64)) for _ in range(COLS) ] 
    for _ in range(ROWS)
]

class GameState(BaseModel):
    board: List[List[int]]           # 0=empty, 1 or 2
    current_player: int              # 1 hoặc 2
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int
    thinking_time: float

class Connect4Engine:
    def __init__(self, board: List[List[int]], ai_piece: int):
        self.board = board
        self.ai_piece = ai_piece
        self.human_piece = 3 - ai_piece
        self.current_hash = self._compute_hash()
        self.TT = {}  # Transposition Table

    def _compute_hash(self) -> int:
        h = 0
        for r in range(ROWS):
            for c in range(COLS):
                p = self.board[r][c]
                if p:
                    # index 0 cho human, 1 cho AI
                    idx = 1 if p == self.ai_piece else 0
                    h ^= ZOBRIST_TABLE[r][c][idx]
        return h

    def _drop(self, board: List[List[int]], row: int, col: int, piece: int, h: int) -> Tuple[int,int]:
        board[row][col] = piece
        idx = 1 if piece == self.ai_piece else 0
        return board, (h ^ ZOBRIST_TABLE[row][col][idx])

    def _is_valid(self, col: int) -> bool:
        return self.board[0][col] == 0

    def _next_row(self, col: int) -> Optional[int]:
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return None

    def _valid_moves(self) -> List[int]:
        return [c for c in range(COLS) if self._is_valid(c)]

    def _win_check(self, b: List[List[int]], piece: int) -> bool:
        # ghép 4 ngang, dọc, chéo
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(b[r][c+i] == piece for i in range(4)): return True
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(b[r+i][c] == piece for i in range(4)): return True
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(b[r+i][c+i] == piece for i in range(4)): return True
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                if all(b[r+i][c-i] == piece for i in range(4)): return True
        return False

    def _is_terminal(self, b: List[List[int]]) -> bool:
        return self._win_check(b, self.ai_piece) or self._win_check(b, self.human_piece) or not self._valid_moves()

    def _evaluate_window(self, window: List[int], piece: int) -> int:
        score = 0
        opp = self.human_piece if piece == self.ai_piece else self.ai_piece
        cnt_p = window.count(piece)
        cnt_e = window.count(0)
        cnt_o = window.count(opp)
        if cnt_p == 4:
            score += 10000
        elif cnt_p == 3 and cnt_e == 1:
            score += 100
        elif cnt_p == 2 and cnt_e == 2:
            score += 10
        if cnt_o == 3 and cnt_e == 1:
            score -= 80
        return score

    def score(self, b: List[List[int]], piece: int) -> int:
        s = 0
        # ngang, dọc, 2 chéo
        for r in range(ROWS):
            for c in range(COLS - 3):
                s += self._evaluate_window(b[r][c:c+4], piece)
        for c in range(COLS):
            col_arr = [b[r][c] for r in range(ROWS)]
            for r in range(ROWS - 3):
                s += self._evaluate_window(col_arr[r:r+4], piece)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                s += self._evaluate_window([b[r+i][c+i] for i in range(4)], piece)
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                s += self._evaluate_window([b[r+i][c-i] for i in range(4)], piece)
        return s

    def order_moves(self, moves: List[int], piece: int, h: int) -> List[int]:
        scored = []
        for col in moves:
            row = self._next_row(col)
            if row is None: continue
            tb = [row.copy() for row in self.board]  # shallow copy
            _, nh = self._drop(tb, row, col, piece, h)
            scored.append((col, self.score(tb, piece)))
        return [c for c, _ in sorted(scored, key=lambda x: -x[1])]

    def minimax(self, b, depth, alpha, beta, maximizing, start_time, h) -> Tuple[Optional[int], int]:
        # time cutoff
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError
        # TT lookup
        if h in self.TT:
            entry = self.TT[h]
            if entry['depth'] >= depth:
                flag = entry['flag']; val = entry['score']
                if flag == 'exact': return entry['move'], val
                if flag == 'lower': alpha = max(alpha, val)
                if flag == 'upper': beta = min(beta, val)
                if alpha >= beta: return entry['move'], val

        valid = self._valid_moves()
        terminal = self._is_terminal(b)
        if depth == 0 or terminal:
            if terminal:
                if self._win_check(b, self.ai_piece): return None, 10**9
                if self._win_check(b, self.human_piece): return None, -10**9
                return None, 0
            return None, self.score(b, self.ai_piece)

        piece = self.ai_piece if maximizing else self.human_piece
        ordered = self.order_moves(valid, piece, h)
        best_move = None
        best_score = -1e18 if maximizing else 1e18

        for col in ordered:
            if time.time() - start_time > TIME_LIMIT: break
            row = self._next_row(col)
            tb = [r.copy() for r in b]
            _, nh = self._drop(tb, row, col, piece, h)
            try:
                _, sc = self.minimax(tb, depth-1, alpha, beta, not maximizing, start_time, nh)
            except TimeoutError:
                break
            if maximizing and sc > best_score:
                best_score, best_move = sc, col; alpha = max(alpha, sc)
            if not maximizing and sc < best_score:
                best_score, best_move = sc, col; beta = min(beta, sc)
            if alpha >= beta: break

        # TT store
        if depth >= MIN_TT_DEPTH and len(self.TT) < TT_SIZE_LIMIT:
            flag = ('exact' if alpha < best_score < beta else
                    'lower' if best_score >= beta else
                    'upper')
            self.TT[h] = {'depth': depth, 'score': best_score, 'flag': flag, 'move': best_move}

        return best_move, best_score

    def find_best(self, valid_moves: List[int]) -> Tuple[int, float]:
        start = time.time()
        best_move = random.choice(valid_moves)
        best_score = -1e18
        depth = 4
        while depth <= MAX_DEPTH and (time.time() - start) < TIME_LIMIT:
            try:
                mv, sc = self.minimax(self.board, depth, -1e18, 1e18, True, start, self.current_hash)
                if mv in valid_moves and sc > best_score:
                    best_move, best_score = mv, sc
            except TimeoutError:
                break
            depth += 1
        return best_move, time.time() - start

@app.post("/api/connect4-move")
async def make_move(gs: GameState) -> AIResponse:
    if not gs.valid_moves:
        raise HTTPException(400, "No valid moves")
    engine = Connect4Engine(gs.board, gs.current_player)
    mv, t = engine.find_best(gs.valid_moves)
    return AIResponse(move=mv, thinking_time=round(t, 4))
