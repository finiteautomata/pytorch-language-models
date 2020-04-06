import torch.nn as nn
from torchqrnn import QRNN


class QRNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, hidden_size,
                 cell_class=nn.GRU, dropout=0.20, zoneout=.0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)

        self.qrnn = QRNN(embedding_dim, hidden_size, num_layers=2, window=2, dropout=dropout, zoneout=zoneout)
        #self.rnn = cell_class(embedding_dim, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hidden=None):
        """
        Inputs are supposed to be just one step (i.e. one letter)
        """

        # inputs = [batch_size, seqlen]
        emb = self.embedding(inp)
        # emb = [batch, seqlen, embedding_dim]
        emb = emb.permute(1, 0, 2)
        # emb = [seqlen, batch, embedding_dim]
        outputs, hidden = self.qrnn(emb, hidden)
        # outputs = [seqlen, batch, hidden_size]
        outputs = outputs.permute(1, 0, 2)
        # outputs = [batch, seqlen, hidden_size]

        # hidden = [batch, hidden_dim]

        out = self.fc(outputs)
        # out = [batch, vocab size]

        return out, hidden
