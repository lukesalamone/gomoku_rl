import gym
import gym_gomoku

from utils import check_five_in_row

class GomokuEnvironment:
    def __init__(self, size):
        assert size in [9, 19], 'unrecognized board size!'

        self.size = size
        self.env = gym.make('Gomoku9x9-v0' if size == 9 else 'Gomoku19x19-v0')
        self.reset()

    # apply an action to the board
    def step(self, action):        
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
        
        # returns 1 if winner is white, 2 if winner is black
        winner = check_five_in_row(self.env.state.board.board_state)

        # map those outcomes to 1 for white win, -1 for black win
        reward = [0,1,-1][winner]

        return self.env.state.board.encode(), reward, True

    # return list of available moves
    def available_moves(self):
        return self.env.state.board.get_legal_action()

    def reset(self):
        self.env.reset()
        return self

    def board(self):
        return self.env.state.board.board_state

    