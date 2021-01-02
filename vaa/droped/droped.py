"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn
from vaa.layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from vaa.utils import replace_masked
import math
from torch.nn.modules.transformer import *

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(TransformerESIM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        # if self.dropout:
        #     self._rnn_dropout = RNNDropout(p=self.dropout)
        #
        #

        self._attention = SoftmaxAttention(self.hidden_size, dropout=self.dropout)
        # self._composition = Seq2SeqEncoder(nn.LSTM,
        #                                    self.hidden_size,
        #                                    self.hidden_size,
        #                                    bidirectional=True)

        self.pos_encoder = PositionalEncoding(self.hidden_size, self.dropout)
        encoder_layers = TransformerEncoderLayer(d_model=384, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)

        self._combine = nn.Sequential(nn.Dropout(p=self.dropout),
                                      nn.Linear(4*self.hidden_size, self.hidden_size),
                                      nn.Tanh(),
                                      nn.Dropout(p=self.dropout))

        self._classification = nn.Sequential(nn.Linear(self.hidden_size, self.num_classes))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, premises, hypotheses):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises = premises[:, :min(128,premises.size()[1]), :]
        hypotheses = hypotheses[:, :min(128, hypotheses.size()[1]), :]

        premises_mask_key = (torch.sum(premises, dim=-1) == 0) # 需要被mask的 true
        hypotheses_mask_key = (torch.sum(hypotheses, dim=-1) == 0)
        premises_mask = 1 - premises_mask_key.float() # 不需要被mask的 1
        hypotheses_mask = 1 - hypotheses_mask_key.float()
        premises_lengths = premises_mask.sum(dim=-1).long()
        hypotheses_lengths = hypotheses_mask.sum(dim=-1).long()

        projected_premises, projected_hypotheses = self._attention(premises, premises_mask,
                                                                   hypotheses, hypotheses_mask)
        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        # projected_premises = self.pos_encoder(projected_premises.transpose(1, 0).contiguous())
        # projected_hypotheses = self.pos_encoder(projected_hypotheses.transpose(1, 0).contiguous())
        # mask1 = self._generate_square_subsequent_mask(len(projected_premises)).to(projected_premises.device)
        # mask2 = self._generate_square_subsequent_mask(len(projected_hypotheses)).to(projected_hypotheses.device)
        # v_ai = self.transformer_encoder(projected_premises).transpose(1, 0).contiguous()
        # v_bj = self.transformer_encoder(projected_hypotheses).transpose(1, 0).contiguous()

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1).transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1).transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        adv_logits = self._combine(v)
        logits = self._classification(adv_logits)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities, adv_logits

