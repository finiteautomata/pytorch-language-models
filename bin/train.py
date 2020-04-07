"""
Script to train a BERT model
"""
import os
from datetime import datetime
import fire
import torch
from torch import nn
import torch.optim as optim
import torchtext
from torchtext import data
from torchtext.datasets import WikiText2
from pytorch_lm.models import RNNLanguageModel, QRNNLanguageModel
from pytorch_lm.training import evaluate, training_cycle
from pytorch_lm.saving import save_model, load_model

AVAILABLE_MODELS = {
    "rnn": RNNLanguageModel,
    "qrnn": QRNNLanguageModel,
}

def create_model(model_name, TEXT, model_args):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"{model_name} not valid; must be one of {AVAILABLE_MODELS.keys()}")
    print(f"Creating model {model_name}...\n")

    PAD_IDX = TEXT.vocab.stoi["<pad>"]
    UNK_IDX = TEXT.vocab.stoi["<unk>"]

    model_args["pad_idx"] = PAD_IDX
    model_args["embedding_dim"] = TEXT.vocab.vectors.shape[1]
    model_args["vocab_size"] = TEXT.vocab.vectors.shape[0]

    model = AVAILABLE_MODELS[model_name](**model_args)

    # Set weight for UNK to a random normal
    if TEXT.vocab.vectors is not None:
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)
        model.embedding.weight.data[UNK_IDX] = torch.randn(model_args["embedding_dim"])
    print("Done\n")
    return model

def create_optimizer(model, optimizer, lr):
    optimizers = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
    }

    if optimizer not in optimizers:
        raise ValueError(f"{optimizer} should be in {optimizers.keys()}")
    else:
        print(f"Creating {optimizer} with lr = {lr}")

        optimizer = optimizers[optimizer](model.parameters(), lr=lr)

        return optimizer


def train_lm(
    model_name, output_path, epochs=5, batch_size=32, bptt_len=35,
    lr=1e-3, optimizer="adam", min_freq=5, model_args={},
    scheduler_patience=5, scheduler_threshold=1e-4, early_stopping_tolerance=5):
    """
    Train and save a language model
    Arguments
    ---------
    model_name: string
        Can be "RNN", "QRNN"

    output_path: a path
        Where to save the model

    lr: float
        Learning rate, default = 1e-3

    model_args: dict
        Arguments to be passed to the createdmodel


    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    TEXT = data.Field(
        tokenizer_language='en',
        lower=True,
        init_token='<sos>',
        eos_token='<eos>',
        batch_first=True,
    )


    train, valid, test = WikiText2.splits(TEXT)

    TEXT.build_vocab(train, vectors="glove.6B.300d", min_freq=min_freq)

    print(f"We have {len(TEXT.vocab)} tokens in our vocabulary")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=batch_size,
        bptt_len=bptt_len, # this is where we specify the sequence length
        device=device,
        repeat=False
    )

    model = create_model(model_name, TEXT, model_args=model_args)

    optimizer = create_optimizer(model, optimizer, lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Using LR Scheduler with patience {scheduler_patience} and threshold {scheduler_threshold}")
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=scheduler_patience, threshold=scheduler_threshold
    )

    model = model.to(device)
    criterion = criterion.to(device)

    model_path = output_path

    training_cycle(
        epochs=epochs,
        model=model, train_iter=train_iter, valid_iter=valid_iter,
        optimizer=optimizer, criterion=criterion, scheduler=lr_scheduler,
        model_path=model_path, early_stopping_tolerance=early_stopping_tolerance
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    valid_loss, valid_perplexity = evaluate(model, valid_iter, criterion)
    test_loss, test_perplexity = evaluate(model, test_iter, criterion)


    print(f"Valid loss      : {valid_loss:.2f}")
    print(f"Valid perplexity: {valid_perplexity:.2f}\n")

    print(f"Test loss      : {test_loss:.2f}")
    print(f"Test perplexity: {test_perplexity:.2f}")


    save_model(model, TEXT, output_path)

if __name__ == "__main__":
    fire.Fire(train_lm)
