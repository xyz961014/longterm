from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import sys
sys.path.append("../..")
from CRAN.data import dataloader
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LSTM = "lstm"

def parseargs(arg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data path of ptb data directory")
    return parser.parse_args(arg)

def repackage_state(s):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(s, torch.Tensor):
        return s.detach()
    else:
        return list(repackage_state(v) for v in s)


class PTBModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._rnn_mode = config.rnn_mode
        self._cell = None
        self._initial_states = None
        self.outputs = torch.tensor([])
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.num_layers = config.num_layers

        self.drop = nn.Dropout(1 - config.keep_prob)
        
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.lstm = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=1 - config.keep_prob)
        self.lstmcell = nn.LSTMCell(config.embedding_dim, config.hidden_size)
        self.hidden2tag = nn.Linear(config.hidden_size, self.vocab_size)

        self.init_weights()

    def init_weights(self):
        init_scale = self.config.init_scale
        self.embeddings.weight.data.uniform_(-init_scale, init_scale)
        self.hidden2tag.weight.data.uniform_(-init_scale, init_scale)
        self.hidden2tag.weight.data.zero_()

    def to(self, device):
        super().to(device)
        for i, (h, c) in enumerate(self._initial_states):
            self._initial_states[i] = h.to(device), c.to(device)
        self.outputs = self.outputs.to(device)


    def forward(self, inputs):
        inputs = self.drop(self.embeddings(inputs))
        #states = self._initial_states

        #outputs, states = self._build_graph(inputs)
        outputs, states = self.lstm(inputs)
        outputs = self.drop(outputs)

        logits = self.hidden2tag(outputs)
        #print(logits.shape)
        tags = logits
        #tags = F.log_softmax(logits, dim=1)
        #tags = tags.view([self.batch_size, self.num_steps, self.vocab_size])
        #tags = torch.transpose(tags, 0, 1)
        

        return tags, states
    
    def _build_graph(self, inputs):
        self._cell = self.wrap_cells([self._get_cell() for _ in range(self.num_layers)], self._rnn_mode)

        #self.outputs = torch.tensor([])
        outputs = self.outputs
        states = self._initial_states
        
        for time_step in range(self.num_steps):
            output, states = self._cell(inputs[time_step, :, :], states)
            outputs = torch.cat((outputs, output))
        outputs = outputs.view([self.num_steps, -1, self.config.hidden_size])
        return outputs, states

    def _get_cell(self):
        if self._rnn_mode == "lstm":
            return self.lstmcell

    def set_initial_states(self, states):
        self._initial_states = states

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def initialize_state_to_zero(self):
        if self._rnn_mode == "lstm":
            self._initial_states = [(torch.zeros([self.batch_size, self.config.hidden_size]), torch.zeros([self.batch_size, self.config.hidden_size])) for _ in range(self.num_layers)]
        else:
            self._initial_states = [torch.zeros([self.batch_size, self.config.hidden_size]) for _ in range(self.num_layers)]
        return self._initial_states

    class wrap_cells:
        def __init__(self, cells, rnn_mode):
            self._cells = cells
            self.rnn_mode = rnn_mode

        def __len__(self):
            return len(self.cells)

        def __call__(self, inputs, states):
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                cur_state = states[i]
                #print(cur_inp.shape, cur_state[0].shape)
                new_state = cell(cur_inp, cur_state)
                if self.rnn_mode == "lstm":
                    cur_inp = new_state[0]
                else:
                    cur_inp = new_state
                new_states.append(new_state)
            return cur_inp, new_states


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 10.0
  max_grad_norm = 0.25
  num_layers = 2
  num_steps = 35
  hidden_size = 200
  embedding_dim = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  eval_batch_size = 10
  vocab_size = 10000
  disp_freq = 5
  rnn_mode = LSTM


def get_config():
    return SmallConfig()


def run_epoch(model, data_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()
    model.set_batch_size(data_loader.batch_size)
    states = model.initialize_state_to_zero()
    model.to(device)

    for i, (data, targets) in enumerate(data_loader):
        states = repackage_state(states)
        model.zero_grad()
        data, targets = data.to(device), targets.to(device)
        data = torch.transpose(data, 0, 1).contiguous()
        targets = torch.transpose(targets, 0, 1).contiguous()
        model.set_initial_states(states)
        try:
            logits, states = model(data)
            #print(logits.shape, targets.shape)
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
        except:
            print(logits)

        torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

        if i % model.config.disp_freq == 0 and i > 0:
            cur_loss = total_loss / model.config.disp_freq
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch + 1, i, len(data_loader), model.config.learning_rate,
                elapsed * 1000 / model.config.disp_freq, cur_loss, np.exp(cur_loss)))
            total_loss = 0.
            start_time = time.time()

def evaluate(model, eval_data, criterion):
    model.eval()
    total_loss = 0.
    model.set_batch_size(eval_data.batch_size)
    states = model.initialize_state_to_zero()
    model.to(device)

    with torch.no_grad():
        for i, (data, targets) in enumerate(eval_data):
            data, targets = data.to(device), targets.to(device)
            data = torch.transpose(data, 0, 1).contiguous()
            targets = torch.transpose(targets, 0, 1).contiguous()
            model.set_initial_states(states)

            logits, states = model(data)
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(eval_data)


    
def main(args):
    config = get_config()

    corpus = dataloader.Corpus(args.data_path)

    train_loader = corpus.get_train_loader(batch_size=config.batch_size, num_steps=config.num_steps)
    valid_loader = corpus.get_valid_loader(batch_size=config.eval_batch_size, num_steps=config.num_steps)
    test_loader = corpus.get_test_loader(batch_size=config.eval_batch_size, num_steps=config.num_steps)
    print(corpus.vocabulary.num_words)
    config.vocab_size = corpus.vocabulary.num_words
    
    model = PTBModel(config)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    #train_data, valid_data, test_data, voc = dataloader.ptb_raw_data(args.data_path)
    #train_set = dataloader.PtbDataset(train_data, num_steps=config.num_steps)
    #valid_set = dataloader.PtbDataset(valid_data, num_steps=config.num_steps)
    #test_set = dataloader.PtbDataset(test_data, num_steps=config.num_steps)

    #train_set.bathify(config.batch_size)
    #valid_set.bathify(config.eval_batch_size)
    #test_set.bathify(config.eval_batch_size)

    #train_loader = DataLoader(train_set, batch_size=config.batch_size, drop_last=True)
    #valid_loader = DataLoader(valid_set, batch_size=config.batch_size, drop_last=True)
    #test_loader = DataLoader(test_set, batch_size=config.batch_size, drop_last=True)


    for epoch in range(config.max_max_epoch):
        epoch_start_time = time.time()
        run_epoch(model, train_loader, loss_function, optimizer, epoch)
        valid_loss = evaluate(model, valid_loader, loss_function)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch + 1, (time.time() - epoch_start_time),valid_loss, np.exp(valid_loss)))
        print('-' * 89)
        #for i, (data, targets) in enumerate(train_loader):
            #model.train()
            #states = repackage_state(states)
            #model.zero_grad()
            #data, targets = data.to(device), targets.to(device)
            #model.set_initial_states(states)

            #logits, states = model(data)
            ##targets = torch.transpose(targets, 0, 1)
            ##targets = targets.view([-1,])
            ##print(logits.shape, targets.shape)
            ##loss = torch.cat(tuple(loss_function(logits[n], targets[n]).view(1) for n in range(targets.size(0))), 0)
            ##loss = torch.sum(loss, 0)
            #loss = loss_function(logits.view(-1, model.vocab_size), targets.view(-1))
            #costs += loss.item()
            #iters += 1

            #loss.backward()
            #optimizer.step()

            #with torch.no_grad():
            #    if i % (len(train_loader) // 10) == 10:
            #        print("%.3f perplexity: %.3f -cost: %.3f speed: %.0f wps" %
            #            (i * 1.0 / len(train_loader), np.exp(costs / iters), loss, iters * model.num_steps  / (time.time() - start_time)))

        #with torch.no_grad():
        #    model.eval()
        #    valid_iters = 0
        #    valid_costs = 0.0
        #    for i, (data, targets) in enumerate(valid_loader):
        #        data, targets = data.to(device), targets.to(device)
    
        #        logits, states = model(data)
        #        targets = torch.transpose(targets, 0, 1)
        #        loss = torch.cat(tuple(loss_function(logits[n], targets[n]).view(1) for n in range(targets.size(0))), 0)
        #        loss = torch.sum(loss, 0)
        #        valid_costs += loss.item()
        #        valid_iters += model.num_steps
    
        #    print("Epoch %d valid perplexity: %.3f " %
        #                (epoch, np.exp(valid_costs / valid_iters)))














if __name__ == "__main__":
    args = parseargs()
    main(args)
            


        


