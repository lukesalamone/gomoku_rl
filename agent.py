from model import GomokuNet





class GomokuAgent:
    def __init__(self, size):
        # initialize neural net
        self.net = GomokuNet(size=size)

    def train(self, examples):
        pass

    def select_move(self, available_moves, board):
        pass