"""
My implementation of the Transformer in pytorch
"""
# pylint: disable=W0221, W0603
# W0221: forward overridden with different paramters
# W0603: use of globals

import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    """
    The EncoderDecoder module
    """

    def __init__(self, encoder, decoder, generator, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, src_mask, tgt, tgt_mask):
        """ Given source and previous prediction, infer the next target """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """ Encoding part of the module """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """ Decoding part of the module """
        return self.decoder(memory, src_mask, self.tgt_embed(tgt), tgt_mask)

class Generator(nn.Module):
    """ The Generator class to generate symbols in vocabulary out of an embedding vector """

    def __init__(self, d_module, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_module, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ## The Encoder Module

# It is composed of six identical EncoderLayers plus output normalization.
# It should be noted that, after the paper publication, an improvement was
# discovered, i.e. to normalize the input embedding vector in the encoder.
# The additional normalization step caused the encoding layers to start with
# normalization and end with residual addition.

def clone(module, num):
    """ Utility routine to make num identical copies of module """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

class Encoder(nn.Module):
    """ The Encoder module """

    # The encoder module is composed of six EncoderLayers
    # followed by a layer normalization

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, src_embed, src_mask):
        for encoder_layer in self.layers:
            src_embed = encoder_layer(src_embed, src_mask)
        # normalization is necessary since each EncoderLayer ends with residual addition.
        return self.norm(src_embed)

class LayerNorm(nn.Module):
    """ The layer normalization module """

    def __init__(self, feature_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(feature_size))
        self.bias = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.bias

class EncoderLayer(nn.Module):
    """ The module for one layer of the Encoder. It consists of the following:
    1. input normalization
    2. self attention & dropout
    3. residual addition and normalization
    4. a fully connected layer
    5. residual addition"""

    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.norm_attn_input = LayerNorm(d_model)
        self.norm_fc_input = LayerNorm(d_model)
        self.size = d_model

    def forward(self, x, mask):
        input_x = x

        norm_x = self.norm_attn_input(input_x)
        attn_out = self.dropout(self.self_attn(norm_x, norm_x, norm_x, mask))
        input_x = input_x + attn_out

        norm_x = self.norm_fc_input(input_x)
        fc_out = self.dropout(self.feed_forward(norm_x))
        return input_x + fc_out

class Decoder(nn.Module):
    """ The Decoder Module. It consists of six layers of DecoderLayers
    plus output normalization. """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, memory, src_mask, tgt, tgt_mask):
        for layer in self.layers:
            tgt = layer(memory, src_mask, tgt, tgt_mask)
        return self.norm(tgt)

class DecoderLayer(nn.Module):
    """ The DecoderLayer module. In addition to the two sub-layers
    in each encoder layer, the decoder layer has a third sub-layer,
    which performs multi-headed attention of the encoder output.
    """
    def __init__(self, d_model, src_attn, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.norm_self_attn_input = LayerNorm(d_model)
        self.norm_src_attn_input = LayerNorm(d_model)
        self.norm_fc_input = LayerNorm(d_model)
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.size = d_model

    def forward(self, memory, src_mask, tgt, tgt_mask):
        input_x = tgt

        norm_x = self.norm_self_attn_input(input_x)
        self_attn_out = self.dropout(self.self_attn(norm_x, norm_x, norm_x, tgt_mask))
        input_x = input_x + self_attn_out

        norm_x = self.norm_src_attn_input(input_x)
        src_attn_out = self.dropout(self.src_attn(norm_x, memory, memory, src_mask))
        input_x = input_x + src_attn_out

        norm_x = self.norm_fc_input(input_x)
        fc_out = self.dropout(self.feed_forward(norm_x))
        return input_x + fc_out

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    mask_value = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask_value) == 0

def attention(query, key, value, mask=None, dropout=None):
    """ The scaled dot product attention """
    dim_k = query.size(-1)
    dot_prod = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    if mask is not None:
        dot_prod = dot_prod.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(dot_prod, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """ Multi headed attention allows attention to different
        locations from different subspaces. """
    def __init__(self, head_count, model_size, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert model_size % head_count == 0
        self.linears = clone(nn.Linear(model_size, model_size), 4)
        self.head_count = head_count
        self.head_size = model_size // head_count
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # use the same mask for all heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        q_trans, k_trans, v_trans = \
            [l(x).view(batch_size, -1, \
             self.head_count, self.head_size).transpose(1, 2) for l, x in \
             zip(self.linears, (query, key, value))]
        val, self.attn = attention(q_trans, k_trans, v_trans, mask=mask, dropout=self.dropout)
        val = val.transpose(1, 2).contiguous().view(batch_size, -1, \
                self.head_count * self.head_size)
        return self.linears[-1](val)

class PositionwiseFeedForward(nn.Module):
    """
    The position-wise feed-forward network has two linear models with Relu activation in between.
    the feed-forward network is shared among word embeddings of all positions.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.lt1 = nn.Linear(d_model, d_ff)
        self.lt2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lt2(self.dropout(F.relu(self.lt1(x))))

class Embeddings(nn.Module):
    """
    We use the usual learned embedding to convert the input/output of vocab size
    to vector of dimension d_model.
    """
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionEmbedding(nn.Module):
    """
    The position embedding are sine and cosine functions so that
    embedding vectors at p+k is a linear transformation from the
    embedding vectors at position p, and the transformation coef-
    ficients are indpendant of the value p.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_embed = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-1 * torch.arange(0, d_model, 2, \
            dtype=torch.float) * math.log(10000.0) / d_model)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer("pe", pos_embed)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, d_model=512, d_ff=2048, \
    head_count=8, dropout=0.1, layer_count=6):
    """
    The routine to build a full model
    """
    mh_attn = MultiHeadedAttention(head_count, d_model, dropout=dropout)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
    pos_embed = PositionEmbedding(d_model, dropout=0.1)
    encoder_layer = EncoderLayer(d_model, mh_attn, feed_forward, dropout=dropout)
    encoder = Encoder(encoder_layer, layer_count)
    decoder_layer = DecoderLayer(d_model, mh_attn, \
        mh_attn, feed_forward, dropout)
    decoder = Decoder(decoder_layer, layer_count)
    generator = Generator(d_model, tgt_vocab)
    src_embed = nn.Sequential(Embeddings(src_vocab, d_model), pos_embed)
    tgt_embed = nn.Sequential(Embeddings(tgt_vocab, d_model), pos_embed)
    enc_dec_model = EncoderDecoder(encoder, decoder, generator, src_embed, tgt_embed)

    for param in enc_dec_model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return enc_dec_model

class Batch:
    """
    Each training sentence pair is transformed into $N$ training
    sentence pairs, each of which consists the source sentence and
    a new target sentence formed with the first $k$ tokens from the
    original target sentence, where $N$ is the total number of tokens
    in the target sentence and $0 < k < N$. Each newly generated
    target sentence is associated with a target mask reflecting the
    number of tokens, $k$, from the original target sentence.
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.tgt_y = tgt[:, 1:]
            self.ntokens = (self.tgt_y != pad).data.sum().type(torch.FloatTensor)

    def make_std_mask(self, tgt, pad):
        """To make a standard mask"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(batch_iter, full_model, loss_compute):
    """
    A Generic Training Loop
    Given a batch data iterator, a model and a loss function, run
    a training loop and report loss (and other information) every 50 steps.
    """
    total_loss = 0
    tokens = 0
    total_tokens = 0
    start = time.time()
    for i, batch in enumerate(batch_iter):
        out = full_model.forward(batch.src, batch.src_mask, batch.tgt, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % \
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

# ### the Batch Size Function
# The training utilizes [torchtext package](https://github.com/pytorch/text)
# for accessing datasets and preprocessing. For this purpose and dynamic
# batching, a batch size calculation function is to be provided.

MAX_SRC_IN_BATCH = 0
MAX_TGT_IN_BATCH = 0
def batch_size_fn(new, count, _):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global MAX_SRC_IN_BATCH, MAX_TGT_IN_BATCH
    if count == 1:
        MAX_SRC_IN_BATCH = 0
        MAX_TGT_IN_BATCH = 0
    MAX_SRC_IN_BATCH = max(MAX_SRC_IN_BATCH, len(new.src))
    MAX_TGT_IN_BATCH = max(MAX_TGT_IN_BATCH, len(new.trg) + 2)
    # torchtext automatically pads sequences to the maximum sequence length
    src_elements = count * MAX_SRC_IN_BATCH
    tgt_elements = count * MAX_TGT_IN_BATCH
    return max(src_elements, tgt_elements)

class NoamOpt:
    """
    Optim wrapper that implements a dynamic learning rate.
    The Adam optimizer with beta_1=0.9, beta_2=0.98 and epsilon=10^{-9}
    is used with an adaptive learning rate. The learning rate is designed
    to linearly increase until a given warmup_step, and then decrease proportially
    to sqrt{step_number}. Additionally the learning rate is inversely proportionla
    to sqrt{d_{model}}. i.e.
    lrate = factor x d_{model}^{-1/2} x min{{step_number}^{-1/2},
    {step_number}/{{warmup_step}^{3/2}}}
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for param_g in self.optimizer.param_groups:
            param_g['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(in_model):
    """The routine to get the standard optimizaer"""
    return NoamOpt(in_model.src_embed[0].d_model, 2, 4000, \
            torch.optim.Adam(in_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    """
    We use label smoothing along with the KL divergence for the loss function.
        size - the vocabulary size
        smoothing - the smooth factor
        padding_index - the index with padding
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.size = size
        self.smoothing = smoothing
        self.padding_index = padding_idx
        self.true_dist = None

    def forward(self, x, target):
        assert self.size == x.size(1)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        true_dist[:, self.padding_index] = 0
        mask = torch.nonzero(target.data == self.padding_index)
        if mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, in_criterion, opt=None):
        self.generator = generator
        self.criterion = in_criterion
        self.opt = opt

    def __call__(self, out, label, norm):
        out = self.generator(out)
        loss = self.criterion(out.contiguous().view(-1, out.size(-1)),
                              label.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """The routine for greeedy decoding"""
    memory = model.encode(src, src_mask)
    out_sofar = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len-1):
        out = model.decode(memory, src_mask, \
                           Variable(out_sofar), \
                           Variable(subsequent_mask(out_sofar.size(1)) \
                                    .type_as(src_mask.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        out_sofar = torch.cat([out_sofar, \
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return out_sofar
