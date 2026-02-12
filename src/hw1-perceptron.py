#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json

import numpy as np

import utils

class Perceptron:
    def __init__(self, n_classes, n_features):
        self.W = np.zeros((n_classes, n_features)) 
        self.k = 0 # num of mistakes

    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def update_weight(self, x_i, y_i):
        """
        x_i (n_features,): a single training example
        y_i (scalar): the gold label for that example
        """
        y_hat = self.predict(np.asmatrix(x_i)) 
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i
            self.k += 1

    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        n_examples = X.shape[0]
        for i in range(n_examples):
            self.update_weight(X[i], y[i])

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        return np.argmax(np.dot(self.W, X.T), axis=0) 

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Compute accuracy: fraction of correct predictions
        return np.mean(y_pred == y)

def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_features = X_train.shape[1]

    # initialize the model
    model = Perceptron(n_classes, n_features)

    epochs = np.arange(1, args.epochs + 1) # epochs = [1, 2, ..., args.epochs]
    print(f"Training for {args.epochs} epochs")

    valid_accuracies = []
    train_accuracies = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        model.train_epoch(X_train, y_train)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        # Decide whether to save the model to args.save_path based on its
        # validation score
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = Perceptron.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accuracies), "validation": (epochs, valid_accuracies)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int, help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="data/emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="q1-perceptron/Q1-perceptron-best.model")
    parser.add_argument("--accuracy-plot", default="q1-perceptron/Q1-perceptron-accs.pdf")
    parser.add_argument("--scores", default="q1-perceptron/Q1-perceptron-scores.json")
    args = parser.parse_args()
    main(args)
