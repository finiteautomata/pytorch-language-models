from tqdm.auto import tqdm
import torch
import math
import numpy as np


def display_lr(lr):
    pow = math.floor(math.log10(lr))
    return f"{lr*(10**(-pow)):.2f}e{pow}"

def train(model, iterator, optimizer, criterion, clip_norm=None):
    """
    Trains the model for one full epoch
    """
    epoch_loss = 0
    epoch_perplexity = 0

    model.train()

    epoch_bar = tqdm(iterator, total=len(iterator))

    i = 0
    for batch in epoch_bar:
        i += 1
        optimizer.zero_grad()
        text = batch.text
        trg = batch.target.view(-1)

        preds = model(text)[0]
        preds = preds.view(-1, preds.shape[-1])

        loss = criterion(preds, trg)

        loss.backward()

        total_norm = 0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** (1. / 2)

        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_perplexity += np.exp(loss.item())

        lr = optimizer.param_groups[0]["lr"]

        epoch_bar.set_description(f"norm = {total_norm:.5f} loss = {epoch_loss / i:.4f} LR = {display_lr(lr)}")

    return epoch_loss / len(iterator), epoch_perplexity / len(iterator)


def evaluate(model, iterator, criterion=None):
    """
    Evaluates the model on the given iterator
    """
    epoch_loss = .0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            trg = batch.target.view(-1)

            preds = model(text)[0]
            preds = preds.view(-1, preds.shape[-1])


            loss = criterion(preds, trg)

            epoch_loss += loss.item()

        loss = epoch_loss / len(iterator)

        perplexity = np.exp(loss)

    return loss, perplexity

def training_cycle(model, train_iter, valid_iter, epochs,
                   optimizer, criterion, scheduler, model_path,
                   early_stopping_tolerance=None, ncols=None):

    best_valid_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")

        train_loss, train_perplexity = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_perplexity = evaluate(model, valid_iter, criterion)

        scheduler.step(valid_loss)

        desc = f' Train Loss: {train_loss:.5f} Perp: {train_perplexity:.3f}'
        desc += f' Val. Loss: {valid_loss:.5f} Perp: {valid_perplexity:.3f}'

        print(desc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"Best model so far (Loss {best_valid_loss:.5f} Perp {valid_perplexity:.2f}) saved at {model_path}")
        else:
            epochs_without_improvement += 1
            if early_stopping_tolerance and epochs_without_improvement >= early_stopping_tolerance:
                print("Early stopping")
                break
