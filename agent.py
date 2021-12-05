from model import GomokuNet
from tqdm.autonotebook import tqdm
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time

class GomokuDataset(Dataset):
    def __init__(self, data, encoderFn):
        self.states, self.colors, self.actions = zip(*data)
        self.len = len(data)
        self.encoderFn = encoderFn

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.encoderFn(self.states[idx], self.colors[idx])
        y = torch.tensor(self.actions[idx], dtype=float, requires_grad=True)
        return x,y

    # convert NxN board matrix into 3xNxN tensor
    def encode_board(self, board, color):
        def gather_player_moves(board, player):
            result = torch.zeros((self.board_size, self.board_size), requires_grad=True)
            for i in range(len(board)):
                for j in range(len(board[i])):
                    if board[i][j] == player:
                        result[i][j] == 1.
            return result
        
        white_layer = gather_player_moves(board, player=1)
        black_layer = gather_player_moves(board, player=2)
        shape = (self.board_size, self.board_size)
        color_layer = torch.ones(shape, requires_grad=True) if color == 1 else torch.zeros(shape, requires_grad=True)
        return torch.stack((white_layer,black_layer,color_layer))



class NormalizedEuclideanLoss(nn.Module):
    def __init__(self, size):
        super(NormalizedEuclideanLoss, self).__init__()
        self.size = size
        self.softmax = torch.nn.Softmax(dim=1)


        # maximum distance two moves can be apart from one another. For
        # example on a 9x9 board the maximum distance in either dimension
        # is (9-1)**2 = 8**2 = 64. So the maximum distance in both 
        # directions is 64+64 = 128.

        self.norm = 2 * (size - 1)**2



    # loss is euclidean distance of predicted move from actual 
    # normalized by the maximum distance self.norm
    def forward(self, inputs, targets, smooth=1):
        # the predicted move is the argmax of the board
        # use softmax to norm probabilities to 1
        if len(inputs.size()) < 2:
            inputs = inputs.unsqueeze(0)

        inputs = torch.argmax(self.softmax(inputs), dim=1)

        inputX = inputs%self.size
        inputY = inputs/self.size

        targetX = targets%self.size
        targetY = targets/self.size

        loss = (inputX - targetX)**2 + (inputY - targetY)**2
        loss = (loss / self.norm)**0.5
        return torch.mean(loss)



class GomokuAgent:
    def __init__(self, size, training_mode=True):
        # initialize neural net
        self.net = GomokuNet(size=size)
        self.training_mode = training_mode
        self.size = size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    # train based on previous board positions
    def train(self, examples):
        start_time = time.time()
        LEARNING_RATE = 0.001
        TRAIN_EPOCHS = 10
        training_data = GomokuDataset(examples, self.net.encode_board)
        dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
        lossFn = NormalizedEuclideanLoss(self.size)
        optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.net.to(self.device)

        for epoch in tqdm(range(TRAIN_EPOCHS)):
            print(f'TRAINING: epoch {epoch+1}/{TRAIN_EPOCHS}')
            # running_loss = 0.0

            for i,(x,y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                predictions = self.net(torch.squeeze(x, dim=1))
                loss = lossFn(predictions, y)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()
        print(f'finished training in {int(time.time() - start_time)} seconds')
        return self

    # select move by evaluating network directly
    def select_move(self, board, color, available_moves):
        available_moves = set(available_moves)
        board = self.net.encode_board(board, color)
        board = board.to(self.device)
        suggestions = self.net(board)
        # print(suggestions)

        mask = torch.ones(self.size ** 2).to(self.device)

        # zero out the probabilities for illegal moves
        for i in range(len(mask)):
            if i not in available_moves:
                mask[i] = 0.0

        if self.training_mode:
            # sample from the probability distribution
            softmax = torch.nn.Softmax(dim=0)
            suggestions = softmax(suggestions) * mask
            selected = torch.multinomial(suggestions, 1).item()
            return selected

        else:
            # select the move with the most confidence
            return torch.argmax(suggestions*mask).item()