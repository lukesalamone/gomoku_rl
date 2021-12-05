import gym
import gym_gomoku

class GomokuEnvironment:
    def __init__(self, size):
        assert size in [9, 19], 'unrecognized board size!'

        self.size = size
        self.env = gym.make('Gomoku9x9-v0' if size == 9 else 'Gomoku19x19-v0')
        self.reset()

    # apply an action to the board
    def step(self, action):
        # assert self.env.state.color == self.env.player_color # it's the player's turn
        
        # If already terminal, then don't do anything
        if self.env.done:
            return self.env.state.board.encode(), 0., True
        
        # Player play
        self.env.state = self.env.state.act(action)
        self.env.action_space.remove(action)
        
        # Reward: if nonterminal, there is no 5 in a row, then the reward is 0
        if not self.env.state.board.is_terminal():
            self.env.done = False
            return self.env.state.board.encode(), 0., False
        
        # We're in a terminal state. Reward is 1 if won, -1 if lost
        self.env.done = True
        
        winner = self.check_five_in_row(self.env.state.board.board_state)
        win_color = 'black' if winner == 1 else 'white'
        reward = 0.
        if winner == 0: # draw
            reward = 0.
        else:
            reward = 1 if winner == 1 else -1
        return self.env.state.board.encode(), reward, True

    # return list of available moves
    def available_moves(self):
        return self.env.state.board.get_legal_action()

    def reset(self):
        self.env.reset()
        return self

    def board(self):
        return self.env.state.board.board_state

    # return 1 if player1 wins, 2 if player2 wins, 0 for no winner
    def check_five_in_row(self, board):
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

    