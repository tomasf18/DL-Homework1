#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json

import numpy as np
import numpy.random as npr

import utils

class Functions:
    def _relu(z):
        return np.maximum(0.0, z)

    def _dRelu(z):
        return np.where(z <= 0, 0, 1)
    
    def _softmax(x):
        """Softmax function for a vector of inputs."""
        exp_x = np.exp(x - np.max(x))  # Subtract max to avoid overflow
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    #def _dSoftmax(x):
        #return np.array([Functions._softmax_derivative_one(i) for i in x])
    
    alias = {
        'relu': (_relu, _dRelu),
        'softmax': (_softmax, None)
    }
    

class MultiLayerPerceptron:
    def __init__(self, n_classes, n_features, learning_rate, hidden_size, activ_h, activ_o):
        self.W1 = np.random.normal(0.1, 0.1, (n_features, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.normal(0.1, 0.1, (hidden_size, n_classes))
        self.b2 = np.zeros(n_classes)
        self.activation = Functions.alias[activ_h]
        self.output_activation = Functions.alias[activ_o]
        self.epsilon = 1e-6
        self.lr=learning_rate

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def forward(self, x, cache=False):
        z1 = x @ self.W1 + self.b1
        h1 = self.activation[0](z1)
        z2 = h1 @ self.W2 + self.b2
        probs = self.output_activation[0](z2)
        if cache:
            return probs, (x, z1, h1, z2)
        return probs
    

    def predict(self, X):
        return np.array([self.forward(x_i) for x_i in X]).argmax(axis=1)

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

    def train_epoch(self, X, y):
        err = 0
        for x_i, y_i in zip(X, y):
            probs, (x0, z1, h1, _) = self.forward(x_i, cache=True)
            err -= np.log(probs[y_i] + self.epsilon)

            grad_z2 = probs.copy()
            grad_z2[y_i] -= 1.0  # dL/dz2
            grad_W2 = np.outer(h1, grad_z2)
            grad_b2 = grad_z2

            grad_h1 = grad_z2 @ self.W2.T
            grad_z1 = grad_h1 * self.activation[1](z1)
            grad_W1 = np.outer(x0, grad_z1)
            grad_b1 = grad_z1

            # SGD step
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
        return err / len(X)
    


def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = MultiLayerPerceptron(n_classes, n_feats, hidden_size=100, learning_rate=0.001, activ_h='relu', activ_o='softmax')

    epochs = np.arange(1, args.epochs + 1)

    train_losses = []
    train_accs   = []
    valid_accs   = []

    start = time.time()
    best_valid = 0.0
    best_epoch = -1

    for epoch in epochs:
        print(f"Training epoch {epoch}")

        # Shuffle training data for SGD
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        train_loss = model.train_epoch(X_train_shuffled, y_train_shuffled)

        # Evaluate on train and validation sets
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(f"train loss: {train_loss:.4f} | "
              f"train acc: {train_acc:.4f} | val acc: {valid_acc:.4f}")
        
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = int(epoch)
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training took {minutes} minutes and {seconds} seconds")

    # Reload best model and evaluate on the test set
    print("Reloading best checkpoint")
    best_model = MultiLayerPerceptron.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)
    print(f"Best model test acc: {test_acc:.4f}")

    # Plot accuracies over epochs
    utils.plot(
        "Epoch", "Accuracy",
        {
            "training": (epochs, train_accs),
            "validation": (epochs, valid_accs),
        },
        filename=args.accuracy_plot
    )

      # Plot training loss over epochs
    utils.plot(
        "Epoch", "Train loss",
        {
            "train_loss": (epochs, train_losses),
        },
        filename=args.loss_plot
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
    parser.add_argument(
        '--epochs', default=20, type=int,
    )
    parser.add_argument(
        '--data-path', type=str, default="data/emnist-letters.npz",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--save-path", default="q1-mlp/Q1-mlp-best.pkl",
    )
    parser.add_argument(
        "--accuracy-plot", default="q1-mlp/Q1-mlp-accs.pdf",
    )
    parser.add_argument(
        "--loss-plot", default="q1-mlp/Q1-mlp-loss.pdf",
    )
    parser.add_argument(
        "--scores", default="q1-mlp/Q1-mlp-scores.json",
    )
    args = parser.parse_args()
    main(args)
