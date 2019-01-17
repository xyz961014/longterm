import torch
import torch.nn as nn
import torch.nn.functional as F

from CRAN.models.CRANUnit import CRANUnit

class CRANModel(nn.Module):
    def __init__(self, args):
        """
        Arguments of Model:
        num_steps: num of words in one step
        vocab_size: vocabulary size
        embedding_dim: dimension of embedding
        Arguments of CRANUnit:
            embedding_dim: dimension of embedding
            hidden_size: size of hidden state
            arguments of cache:
                N: size of the Cache
                dk: dimension of key
                dv: dimension of value
                L: max length of a sequence stored in one value
                k: select top k values
        """
        super().__init__()
        self.args = args
        self.cranunit = CRANUnit(args)
        self.embeddings = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.num_steps = args.num_steps
        self.hidden2tag = nn.Linear(args.hidden_size, args.vocab_size)

    def to(self, device):
        super().to(device)
        self.cranunit.to(device)

    def forward(self, inputs):
        inputs = self.embeddings(inputs)

        hiddens = self._build_graph(inputs)
        #print("modelhiddens", hiddens.shape)

        logits = self.hidden2tag(hiddens)
        return logits

    def _build_graph(self, inputs):
        outputs = torch.tensor([], device=inputs.device)
        for step in range(self.num_steps):
            output = self.cranunit(inputs[step])
            outputs = torch.cat((outputs, output.view([1]+list(output.size()))), 0)
        return outputs

    def set_batch_size(self, batch_size):
        self.cranunit.set_batch_size(batch_size)
