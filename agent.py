from os import replace
from tqdm.autonotebook import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

from model import GomokuNet
from utils import encode_board, decode_position, encode_position
class GomokuDataset(Dataset):
    def __init__(self, data):
        self.states, self.values = zip(*data)
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = encode_board(self.states[idx])
        y = torch.tensor(self.values[idx], dtype=float, requires_grad=True)
        return x,y

class GomokuAgent:
    def __init__(self, size, wandb):
        # initialize neural net
        self.net = GomokuNet(size=size)
        self.wandb = wandb
        self.wandb.watch(self.net)
        self.size = size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    # train based on previous board positions
    def train(self, examples):
        def select_examples(_examples, buffer_size):
            m = len(_examples) - buffer_size
            probs_old = [1/m/2 for _ in range(m)]
            probs_new = [1/buffer_size/2 for _ in range(buffer_size)]
            probs = probs_old + probs_new
            return np.random.choice(_examples, size=buffer_size, replace=False, p=probs)

        start_time = time.time()
        # self.wandb.log()
        print(f'training with {len(examples)} examples')
        LEARNING_RATE = 0.001
        TRAIN_EPOCHS = 10
        BUFFER_SIZE = 2000

        if len(examples) > BUFFER_SIZE:
            examples = select_examples(examples, BUFFER_SIZE)

        training_data = GomokuDataset(examples)
        dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
        optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.net.to(self.device)

        for epoch in tqdm(range(TRAIN_EPOCHS)):
            print(f'TRAINING: epoch {epoch+1}/{TRAIN_EPOCHS}')
            running_loss = 0.0

            for i,(x,y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                predictions = self.net(torch.squeeze(x, dim=1))
                loss = F.mse_loss(predictions, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.wandb.log({'loss':loss})
                self.wandb.log({'running_loss':running_loss})
        print(f'finished training in {int(time.time() - start_time)} seconds')
        return self

    # select move by evaluating network directly
    # TODO experiment with training_mode to sample rather than greedy select
    def select_move(self, board, color, available_moves):
        available_moves = set(available_moves)
        available_moves = [decode_position(pos, self.size) for pos in available_moves]

        board = encode_board(board)
        board = board.to(self.device)

        # evaluate value function for each position IF WE WENT THERE
        # this is equivalent to minimax of depth 1
        best_val = -float('inf') if color == 1 else float('inf')
        best_move = None
        for y,x in available_moves:
            # print(board.size())
            if color == 1:
                board[0,0,y,x] = 1
            else:
                board[0,1,y,x] = 1

            value = self.net(board)

            if color == 1:
                board[0,0,y,x] = 0
            else:
                board[0,1,y,x] = 0

            if value > best_val and color == 1:
                best_val = value
                best_move = (y,x)
            elif value < best_val and color == -1:
                best_val = value
                best_move = (y,x)
        
        y,x =  best_move
        return encode_position(best_move, self.size)

    def save(self, iteration):
        torch.save(self.net, f'iteration_{iteration}_{int(time.time())}.pt')