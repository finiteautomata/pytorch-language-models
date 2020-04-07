"""
Script to train a BERT model
"""
import os
from datetime import datetime
import fire
import torch
from torchtext import data
from torchtext.datasets import WikiText2
import torch.nn as nn
from pytorch_lm.training import evaluate
from pytorch_lm import load_model


def evaluate_lm(model_path):
    """
    Evaluate language model against Wiki2
    Arguments
    ---------
    model_path: string
        Can be "RNN", "QRNN"
    """


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, TEXT = load_model(model_path, device)


    train, valid, test = WikiText2.splits(TEXT)


    BATCH_SIZE = 32
    BPTT_LEN = 30

    train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        bptt_len=BPTT_LEN, # this is where we specify the sequence length
        device=device,
        repeat=False)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    valid_loss, valid_perplexity = evaluate(model, valid_iter, criterion)
    test_loss, test_perplexity = evaluate(model, test_iter, criterion)


    print(f"Valid loss      : {valid_loss:.3f}")
    print(f"Valid perplexity: {valid_perplexity:.2f}\n")

    print(f"Test loss      : {test_loss:.3f}")
    print(f"Test perplexity: {test_perplexity:.2f}")

if __name__ == "__main__":
    fire.Fire(evaluate_lm)
