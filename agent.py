from os import replace
from tqdm import tqdm
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
        self.states, self.colors, self.values = zip(*data)
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = encode_board(self.states[idx], self.colors[idx])
        y = torch.tensor(self.values[idx], dtype=torch.float32, requires_grad=True)
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
        self.training_mode = True
        self.softmax = torch.nn.Softmax(dim=0)

    # train based on previous board positions
    def train(self, examples):
        def select_examples(_examples, buffer_size):
            m = len(_examples) - buffer_size
            n = len(_examples)
            probs_old = [1/m/2 for _ in range(m)]
            probs_new = [1/buffer_size/2 for _ in range(buffer_size)]
            probs = probs_old + probs_new
            idx = np.random.choice([x for x in range(n)], size=buffer_size, replace=False, p=probs)
            return [_examples[x] for x in idx]

        start_time = time.time()
        print(f'training with {len(examples)} examples')
        LEARNING_RATE = 0.001
        TRAIN_EPOCHS = 10
        BUFFER_SIZE = 50_000
        BATCH_SIZE = 16384

        if len(examples) > BUFFER_SIZE:
            examples = select_examples(examples, BUFFER_SIZE)

        training_data = GomokuDataset(examples)
        dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.net.to(self.device)

        for epoch in tqdm(range(TRAIN_EPOCHS)):
            # print(f'TRAINING: epoch {epoch+1}/{TRAIN_EPOCHS}')
            running_loss = 0.0

            for i,(x,y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                predictions = self.net(torch.squeeze(x, dim=1)).squeeze(dim=1)
                loss = F.mse_loss(predictions, y).float()
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
        def place_move(board, y, x, color_idx):
            board[0,color_idx,y,x] = 1
            return board
        
        available_moves = set(available_moves)
        available_moves_decoded = [decode_position(pos, self.size) for pos in available_moves]

        board = encode_board(board, color)

        color_idx = 0 if color == 1 else 1
        options = []
        
        for y,x in available_moves_decoded:
            options.append(place_move(board.detach(), y, x, color_idx))
        
        options = torch.cat(options).to(self.device)
        self.net.eval()
        with torch.no_grad():
            values = self.net(options).squeeze(dim=1)

            if self.training_mode:
                # sample from probability distribution
                idx = torch.multinomial(self.softmax(values), 1)
                best_move = available_moves_decoded[idx]
            else:
                # greedy select
                best_move = available_moves_decoded[torch.argmax(values)]
            return encode_position(best_move, self.size)

    def save(self, iteration):
        filename = f'iteration_{iteration}_{int(time.time())}.pt'
        print(f'saved model as {filename}')
        torch.save(self.net, filename)