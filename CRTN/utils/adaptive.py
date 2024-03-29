import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from CRTN.utils.fancy_dropout import embedded_dropout
from torchnlp.nn import LockedDropout

CUDA_MAJOR = int(torch.version.cuda.split('.')[0])
CUDA_MINOR = int(torch.version.cuda.split('.')[1])

class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, init_std=0.02,
                 dropemb=0.0, sample_softmax=False):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.dropemb = dropemb

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, padding_idx=1, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i, padding_idx=1))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))
        
        self.init_weights(init_std)
    
    def init_weights(self, init_std):
        for i in range(len(self.emb_projs)):
            nn.init.normal_(self.emb_projs[i], 0.0, init_std)


    def forward(self, inp):

        if self.div_val == 1:
            embed = embedded_dropout(self.emb_layers[0], inp, 
                                     dropout=self.dropemb if self.training else 0)
            #embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = torch.nonzero(mask_i).squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                #emb_i = self.emb_layers[i](inp_i)
                emb_i = embedded_dropout(self.emb_layers[i], inp_i, 
                                         dropout=self.dropemb if self.training else 0)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, init_std=0.02,
                 proj_init_std=0.01, keep_order=False, mos=False, n_experts=10, dropmos=0.5):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.mos = mos
        self.n_experts = n_experts

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.init_std = init_std
        self.proj_init_std = proj_init_std

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        if mos:
            self.prior = nn.Linear(d_proj, n_experts, bias=False)
            self.latent = nn.Sequential(nn.Linear(d_proj, n_experts * d_embed), nn.Tanh())
            self.dropmos = LockedDropout(dropmos)

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(
                        nn.Parameter(torch.Tensor(d_proj, d_embed))
                    )
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))

        self.keep_order = keep_order
        self.init_weights()

    def init_weights(self):
        if self.n_clusters > 0:
            nn.init.normal_(self.cluster_weight, 0.0, self.init_std)
            nn.init.constant_(self.cluster_bias, 0.0)
        for i in range(len(self.out_projs)):
            if self.out_projs[i] is not None:
                nn.init.normal_(self.out_projs[i], 0.0, self.proj_init_std)

        for i in range(len(self.out_layers)):
            nn.init.normal_(self.out_layers[i].weight, 0.0, self.init_std)
            nn.init.constant_(self.out_layers[i].bias, 0.0)

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=False, output=False, temperature=1.0):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''
        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.mos:
            nhid = hidden.size(-1)
            prior_logit = self.prior(hidden).reshape(-1, self.n_experts)
            prior = F.softmax(prior_logit, dim=-1)
            hidden = self.latent(hidden)
            hidden = self.dropmos(hidden).reshape(-1, nhid)
            target = target.reshape(-1)
        else:
            hidden = hidden.reshape(-1, hidden.size(-1))
            target = target.reshape(-1)

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            logit = logit / temperature
            if self.mos:
                prob = F.softmax(logit, dim=-1).view(-1, self.n_experts, self.n_token)
                log_prob = torch.einsum("bk,bkn->bkn", prior, prob).sum(1).log()
            else:
                log_prob = F.log_softmax(logit, dim=-1)
            nll = -log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                    #bias_i = weight_i.new_zeros(weight_i.size(0))
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias
                    #bias_i = weight_i.new_zeros(weight_i.size(0))

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logit = head_logit / temperature
            if self.mos:
                head_prob = F.softmax(head_logit, dim=-1).view(-1, self.n_experts, head_weight.size(0))
                head_logprob = torch.einsum("bk,bkn->bkn", prior, head_prob).sum(1).log()
            else:
                head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target,
                    dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            tail_probs = []
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = torch.nonzero(mask_i)
                indices_i.squeeze_()

                if not output:
                    if indices_i.numel() == 0:
                        continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    if self.mos:
                        hidden_i = hidden.reshape(-1, self.n_experts, nhid).index_select(0, indices_i)
                        hidden_i = hidden_i.reshape(-1, nhid)
                        prior_i = prior.index_select(0, indices_i)
                    else:
                        hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logit_i = tail_logit_i / temperature
                    if self.mos:
                        tail_prob_i = F.softmax(tail_logit_i, dim=-1).view(-1, self.n_experts, weights[i].size(0))
                        tail_logprob_i = torch.einsum("bk,bkn->bkn", prior_i, tail_prob_i).sum(1).log()
                    else:
                        tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    if output:
                        tail_logit_output = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                        tail_logit_output = tail_logit_output / temperature
                        if self.mos:
                            tail_prob_output = F.softmax(tail_logit_output, dim=-1).view(-1, self.n_experts, weight_i.size(0))
                            tail_logprob_output = torch.einsum("bk,bkn->bkn", prior, tail_prob_output).sum(1).log()
                        else:
                            tail_logprob_output = F.log_softmax(tail_logit_output, dim=1)
                        tail_probs.append(tail_logprob_output)

                    logprob_i = head_logprob_i[:, -i] \
                              + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)

                if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)
        
        if not output:
            return nll
        else:
            return (head_logprob, tail_probs)
