import torch.nn as nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx, hidden_size,
                 cell_class=nn.GRU, dropout=0.20):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.rnn = cell_class(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, inp, hidden=None):
        """
        Inputs are supposed to be just one step (i.e. one letter)
        """
        # inputs = [batch_size, ]
        emb = self.embedding(inp)
        # emb = [batch, embedding_dim]
        # As all my examples are of the same length, there is no use
        # in packing the input to the RNN
        rnn_outputs, hidden = self.rnn(emb, hidden)
        # hidden = [batch, hidden_dim]

        out = self.fc(self.dropout(rnn_outputs))
        # out = [batch, vocab size]

        return out, hidden
