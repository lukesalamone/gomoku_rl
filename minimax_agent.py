from utils import check_five_in_row, encode_position


class MinimaxAgent:
    def __init__(self, size, depth):
        self.board_size = size
        self.max_depth = depth

    def select_move(self, board, color, available_moves):
        self.ai_color_val = 1 if color == 0 else 2
        self.op_color_val = 2 if color == 0 else 1
        best_score = -float('inf')
        squares = self.vacant_squares(board)
        move = None
        
        for y,x in squares:
            board[y][x] = self.ai_color_val
            score = self.alphabeta(board, 0, -float('inf'), float('inf'), False)
            board[y][x] = 0
            if score > best_score:
                best_score = score
                move = [y,x]
        return encode_position(move, self.board_size)

    def alphabeta(self, board, depth, alpha, beta, is_turn):
        winner = self.check_winner(board)
        if winner:
            return 9999 * winner

        if depth >= self.max_depth:
            return self.static_eval(board)

        best = -float('inf') if is_turn else float('inf')
        squares = self.vacant_squares(board)

        for y,x in squares:
            board[y][x] = self.ai_color_val if is_turn else self.op_color_val
            score = self.alphabeta(board, depth+1, alpha, beta, not is_turn)
            board[y][x] = 0
            best = max(score, best) if is_turn else min(score, best)

            if is_turn:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            
            if alpha >= beta:
                break
        return best

    def static_eval(self,board):
        def adj_block_score(streak):
            scoreMatrix = [0, 2, 4, 8, 16, 32]
            return 0 if (streak < 0 or streak >= len(scoreMatrix)) else scoreMatrix[streak]

        def score_consec(square, current, streak, score):
            if square != current:
                if current == 0:
                    current = square
                    streak = 1
                else:
                    score += current * adj_block_score(streak)
                    current = square
                    streak = 1
            else:
                streak += 0 if square == 0 else 1
            return {'current':current, 'streak':streak, 'score':score}

        def horizontal_score(board):
            score = 0
            for row in board:
                current, streak = 0, 0
                for square in row:
                    current, streak, score = score_consec(square, current, streak, score).values()
                score += current * adj_block_score(streak)
            return -score
            
        def vertical_score(board):
            score = 0
            for i in range(len(board[0])):
                current, streak = 0, 0
                for j in range(len(board)):
                    current, streak, score = score_consec(board[j][i], current, streak, score).values()
                score += current * adj_block_score(streak)
            return -score

        def diagonal_score(board):
            score = 0
            res = {'d1': {}, 'd2': {}, 'd3': {}, 'd4': {}}
            L = len(board)
            for i in range(4, L):
                for key in res:
                    res[key] = {'streak': 0, 'current': 0, 'score': 0}
                for j in range(0, i):
                    res['d1'] = score_consec(board[i-j][j], res['d1']['current'], 
                                            res['d1']['streak'], res['d1']['score'])
                    res['d2'] = score_consec(board[L-1-j][i-j], res['d2']['current'],
                                            res['d2']['streak'], res['d2']['score'])
                    res['d3'] = score_consec(board[j][L-1-i+j], res['d3']['current'], 
                                            res['d3']['streak'], res['d3']['score'])
                    res['d4'] = score_consec(board[L-1-i+j][L-1-j], res['d4']['current'], 
                                            res['d4']['streak'], res['d4']['score'])
                score += sum([x['score'] for x in res.values()])
            return -score

        return horizontal_score(board) + vertical_score(board) + diagonal_score(board)


    def check_winner(self, board):
        result = check_five_in_row(board)
        if result == 0:
            return 0
        elif result == self.ai_color_val:
            return 1
        else:
            return -1

    def vacant_squares(self, board):
        vacant = []

        for row in range(len(board)):
            for col in range(len(board[row])):
                if not board[row][col]:
                    vacant.append((row, col))
        return vacant
        
