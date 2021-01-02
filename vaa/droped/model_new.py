"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn
from vaa.layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, LinerEncoder
from vaa.utils import get_mask, replace_masked
# from allennlp.modules.elmo import Elmo, batch_to_ids

class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
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
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        self.transformer_model = nn.Transformer(d_model=self.embedding_dim, nhead=4,
                                                num_encoder_layers=3, num_decoder_layers=3)

        self._composition = nn.LSTM(self.embedding_dim, self.hidden_size, bidirectional=True, batch_first=True)

        self._classification = nn.Sequential(nn.Linear(self.hidden_size*2, self.num_classes))


    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths,
                embedd=False,
                premises_mask=None,
                hypotheses_mask=None):
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
        if premises_mask is None:
            premises_mask = get_mask(premises, premises_lengths).to(self.device)
            hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)

        if embedd:
            embedded_premises = premises
            embedded_hypotheses = hypotheses
        else:
            embedded_premises = self._word_embedding(premises)
            embedded_hypotheses = self._word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        # encoded_premises = self._encoding(embedded_premises, premises_lengths)
        # encoded_hypotheses = self._encoding(embedded_hypotheses, hypotheses_lengths)

        v = self.transformer_model(embedded_premises.transpose(0, 1), embedded_hypotheses.transpose(0,1)).transpose(0,1)
        _, (hn, cn) = self._composition(v)
        hn = hn.transpose(0, 1).contiguous()
        logits = self._classification(hn.view(hn.size()[0], -1))
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities, hn, \
               (embedded_premises, embedded_hypotheses, premises_mask, hypotheses_mask)


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
