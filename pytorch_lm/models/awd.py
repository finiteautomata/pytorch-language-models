from fastai.text.models import AWD_LSTM, LinearDecoder
import torch.nn as nn

class AWDLanguageModel(nn.Sequential):
    def __init__(
        self, vocab_size, embedding_dim, pad_idx, hidden_size, n_layers=3,
        dropout=0.20, **kwargs):
        encoder = AWD_LSTM(
            vocab_sz=vocab_size, emb_sz = embedding_dim, n_hid=hidden_size, n_layers=n_layers,
            **kwargs
        )

        decoder = LinearDecoder(n_out=vocab_size, n_hid=embedding_dim, output_p=dropout)
        super().__init__(encoder, decoder)
