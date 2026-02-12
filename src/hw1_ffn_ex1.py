#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt

import time
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(self, n_classes: int, n_features: int, hidden_size: int, layers: int, activation_type: str, dropout: float, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__() 
        # here we define the layers and activations
        self.first_hidden_layer = nn.Linear(n_features, hidden_size)    # for the weight matrix connecting input to first hidden layer 
        self.remaining_hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers - 1)]) # for the weight matrices connecting hidden layers to hidden layers
        self.output_layer = nn.Linear(hidden_size, n_classes)           # for the weight matrix connecting last hidden layer to output 
        
        if activation_type.lower() == "relu":
            self.hidden_activation = nn.ReLU()
        elif activation_type.lower() == "tanh":
            self.hidden_activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")

        # self.output_activation = nn.Softmax(dim=1) # -> Model must return raw logits (scores) for nn.CrossEntropyLoss, which applies Softmax internally
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, **kwargs): # Defines data flow, automatically called when doing model(x)
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        # e.g., for 5 hidden layers:
        x = self.first_hidden_layer(x)
        x = self.hidden_activation(x)
        x = self.dropout(x)
        
        for layer in self.remaining_hidden_layers:
            x = layer(x)
            x = self.hidden_activation(x)
            x = self.dropout(x)
        
        x = self.output_layer(x)
        # x = self.output_activation(x) -> Model must return raw logits (scores)  for nn.CrossEntropyLoss, which applies Softmax internally
        return x

    
def train_batch(X: torch.Tensor, y: torch.Tensor, model: nn.Module, optimizer: torch.optim.Adam | torch.optim.SGD, criterion: nn.CrossEntropyLoss, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """
    model.train()
    optimizer.zero_grad()
    y_hat = model(X)
    loss = criterion(y_hat, y)
    loss.backward() # Automatic gradient computation
    optimizer.step()
    return loss.item() # .item() to get the Python float value from the single-value tensor

@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    model.eval()
    y_hat = model(X) # shape: (n_examples x n_classes)
    preds = torch.argmax(y_hat, dim=1) # dim=1 because we want the index of the max value along the classes dimension
    return preds # "preds" (in plural) because it contains predictions for all examples

@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.CrossEntropyLoss):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    model.eval() # set the model to evaluation mode, which deactivates dropout, batchnorm, and combined with @torch.no_grad() disables gradient calculation
    # with torch.no_grad(): -> not necessary because of the decorator (annotation) above
    y_hat = model(X)
    loss = criterion(y_hat, y)
    predictions = torch.argmax(y_hat, dim=1)
    accuracy = (predictions == y).float().mean() # accuracy is the mean of correct predictions
    return loss.item(), accuracy.item()


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int, help="""Number of epochs to train for. You should not need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int, help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation', choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='data/emnist-letters.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_features = dataset.X.shape[1]

    print(f"N features: {n_features}")
    print(f"N classes: {n_classes}")

    # initialize the model
    model = FeedforwardNetwork(
        n_classes,
        n_features,
        opt.hidden_size,
        opt.layers,
        opt.activation,
        opt.dropout
    )#.to(utils.get_best_device())

    # move to device if available
    # train_X, train_y = train_X.to(utils.get_best_device()), train_y.to(utils.get_best_device())
    # dev_X, dev_y = dev_X.to(utils.get_best_device()), dev_y.to(utils.get_best_device())
    # test_X, test_y = test_X.to(utils.get_best_device()), test_y.to(utils.get_best_device())

    # get an optimizer
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optimizer_class = optimizers[opt.optimizer]
    optimizer = optimizer_class(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # get a loss criterion -> as this is a multi class classification problem, we use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(0, opt.epochs + 1)
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    start = time.time()

    # model.eval() -> not necessary here because evaluate() already sets the model to eval mode
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accuracies.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accuracies.append(initial_val_acc)
    print('initial val acc: {:.4f}'.format(initial_val_acc))

    for ii in epochs[1:]:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        # model.train() -> not necessary here because train_batch() already sets the model to train mode
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        # model.eval() -> not necessary here because evaluate() already sets the model to eval mode
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('train loss: {:.4f}| train_acc: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
            epoch_train_loss, train_acc, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        train_accuracies.append(train_acc)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final Test Accuracy: {:.4f}'.format(test_acc))

    # plot
    config = (
        f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
        f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
        f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
    )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }
    
    accs = {
        "Train Accuracy": train_accuracies,
        "Valid Accuracy": valid_accuracies
    }

    plot(epochs, losses, filename=f'q2-ffn-1/losses-{config}.pdf')
    plot(epochs, accs, filename=f'q2-ffn-1/accs-{config}.pdf')
    print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(valid_accuracies):.4f}")


if __name__ == '__main__':
    main()
