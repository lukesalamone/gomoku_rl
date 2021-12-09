import torch

# return 1 if player1 wins, 2 if player2 wins, 0 for no winner
def check_five_in_row(board):
    def check(square, obj):
        if square:
            if square == obj['current']:
                obj['streak'] += 1
            else:
                obj['streak'] = 1
                obj['current'] = square
        else:
            obj['streak'] = 0
            obj['current'] = 0

        return obj['current'] if obj['streak'] == 5 else 0

    for i in range(len(board)):
        res = {k:{'streak':0, 'current':0} for k in ['hor','ver','dg1','dg2','dg3','dg4']}

        for j in range(len(board[i])):
            # check horizontals
            winner = check(board[i][j], res['hor'])
            if(winner):
                return winner

            # check verticals
            winner = check(board[j][i], res['ver'])
            if(winner):
                return winner

            # check all four diagonals
            if i < 4 or j > i:
                continue

            winner = check(board[i-j][j], res['dg1'])
            if(winner):
                return winner

            L = len(board)
            winner = check(board[L-1-j][i-j], res['dg2'])
            if(winner):
                return winner

            winner = check(board[j][L-1-i+j], res['dg3'])
            if(winner):
                return winner

            winner = check(board[L-1-i+j][L-1-j], res['dg4'])
            if(winner):
                return winner
    return 0

def encode_board(board):
    def gather_player_moves(board, player):
        size = len(board)
        result = torch.zeros((size, size))
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == player:
                    result[i][j] == 1.
        return result
    
    white_layer = gather_player_moves(board, player=1)
    black_layer = gather_player_moves(board, player=2)
    return torch.unsqueeze(
                torch.stack((white_layer,black_layer)),
                dim=0
            )

# (2,3) => 21
def encode_position(pos, size):
    y,x = pos
    return y*size + x

def decode_position(pos, size):
    y = pos//size
    x = pos%size
    return y,x