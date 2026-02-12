#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import time
import utils

import csv
import copy
from pathlib import Path

from hw1_ffn_ex1 import FeedforwardNetwork, train_batch, evaluate  


depths = [1, 3, 5, 7, 9]
hidden_size = 32
lr = 0.001
dropout = 0.0
l2 = 0.0001
optimizer_name = "adam" 
activation = "relu"
batch_size = 64
n_epochs = 30

device = utils.get_best_device()
print("Using device:", device)


def run_for_depth(n_layers, data_path):
    utils.configure_seed(seed=42)

    data = utils.load_dataset(data_path)
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device).manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0] # 26
    n_features = dataset.X.shape[1]

    # initialize in device
    model = FeedforwardNetwork(
        n_classes=n_classes,
        n_features=n_features,
        hidden_size=hidden_size,
        layers=n_layers,
        activation_type=activation,
        dropout=dropout
    ).to(device)

    # move whole dataset tensors to device
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    dev_X = dev_X.to(device)
    dev_y = dev_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    # initial eval (before training)
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    
    train_losses.append(initial_train_loss)
    train_accuracies.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accuracies.append(initial_val_acc)

    best_val_acc = initial_val_acc
    best_state = copy.deepcopy(model.state_dict())

    # training loop
    for _ in range(1, n_epochs + 1):
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            Xb = X_batch.to(device)
            yb = y_batch.to(device)
            loss = train_batch(Xb, yb, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        # mean train loss for epoch
        mean_epoch_train_loss = float(torch.tensor(epoch_train_losses).mean().item()) 

        # evaluate on full train and dev
        train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        train_losses.append(mean_epoch_train_loss)
        train_accuracies.append(train_acc)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)

        # track best val accuracy and save model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # after epochss, restore best model and evaluate on test and train
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_X, test_y, criterion)
    train_loss_at_best, train_acc_at_best = evaluate(model, train_X, train_y, criterion)

    return {
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "lr": lr,
        "dropout": dropout,
        "l2": l2,
        "initial_validation_accuracy": initial_val_acc,
        "best_validation_accuracy": best_val_acc,
        "test_accuracy_at_best_validation": test_acc,
        "test_loss_at_best_validation": test_loss,
        "train_accuracy_at_best_validation": train_acc_at_best,
        "train_loss_at_best_validation": train_loss_at_best,
        "train_losses_per_epoch": train_losses,
        "train_accuracies_per_epoch": train_accuracies,
        "valid_losses_per_epoch": valid_losses,
        "validation_accuracies_per_epoch": valid_accuracies,
        "final_epoch_train_accuracy": train_accuracies[-1]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, default="data/emnist-letters.npz")
    parser.add_argument("-output-dir", type=str, default="q2-ffn-3/")
    args = parser.parse_args()
    
    data = utils.load_dataset(args.data)
    dataset = utils.ClassificationDataset(data)
    print(f"Dataset sizes: train {dataset.X.shape[0]}, dev {dataset.dev_X.shape[0]}, test {dataset.test_X.shape[0]}")
    print(f"Features: {dataset.X.shape[1]}, Classes: {torch.unique(dataset.y).shape[0]}")
    print("\nGrid specs:")
    print("depths:", depths)
    print("\nBest-performing configuration obtained for the 32-unit model:")
    print("hidden size:", hidden_size)
    print("lr:", lr)
    print("dropout:", dropout)
    print("l2:", l2)
    print("optimizer:", optimizer_name)
    print("activation:", activation)
    print("batch_size:", batch_size)
    print("n_epochs:", n_epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = out_dir / "results_depth_grid_search.csv"

    results = {}

    with csv_output_path.open("w", newline="") as csvfile:
        fieldnames = [
            "n_layers", "hidden_size", "lr", "dropout", "l2",
            "initial_validation_accuracy", "best_validation_accuracy",
            "test_accuracy_at_best_validation", "test_loss_at_best_validation",
            "train_accuracy_at_best_validation", "train_loss_at_best_validation",
            "final_epoch_train_accuracy"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        start_time = time.time()
        total_runs = len(depths)
        run_idx = 0

        for L in depths:
            run_idx += 1
            print(f"\nRun {run_idx}/{total_runs} -> depth={L}")
            current_experiment_result = run_for_depth(L, args.data)
            results[L] = current_experiment_result

            writer.writerow({
                "n_layers": current_experiment_result["n_layers"],
                "hidden_size": current_experiment_result["hidden_size"],
                "lr": current_experiment_result["lr"],
                "dropout": current_experiment_result["dropout"],
                "l2": current_experiment_result["l2"],
                "initial_validation_accuracy": f"{current_experiment_result['initial_validation_accuracy']:.6f}",
                "best_validation_accuracy": f"{current_experiment_result['best_validation_accuracy']:.6f}",
                "test_accuracy_at_best_validation": f"{current_experiment_result['test_accuracy_at_best_validation']:.6f}",
                "test_loss_at_best_validation": f"{current_experiment_result['test_loss_at_best_validation']:.6f}",
                "train_accuracy_at_best_validation": f"{current_experiment_result['train_accuracy_at_best_validation']:.6f}",
                "train_loss_at_best_validation": f"{current_experiment_result['train_loss_at_best_validation']:.6f}",
                "final_epoch_train_accuracy": f"{current_experiment_result['final_epoch_train_accuracy']:.6f}"
            })
            csvfile.flush()

    elapsed = time.time() - start_time
    print(f"\nAll runs finished. Results written to {csv_output_path}. Elapsed seconds: {int(elapsed)}")

    #(a)
    print("\nWriting highest validation accuracy per depth to file...")
    highest_val_acc_path = out_dir / "a_highest_val_acc_per_depth.txt"
    with highest_val_acc_path.open("w") as f:
        f.write("Highest validation accuracy per depth (L):\n")
        f.write("{:>2s} {:>20s}\n".format("L", "best_val_acc"))
        for L in sorted(results.keys()):
            f.write(f"{L:>2d} {results[L]['best_validation_accuracy']:>20.6f}\n")
            
    print(f"\nSaved highest validation accuracy per depth to {highest_val_acc_path}")

    #(b) -> decided to use the 3 plots to be sure everything is guaranteed to be present regarding the exercise instructions 
    print("\nPlotting train loss and validation accuracy curves for best depth...")
    best_depth_size = max(results.keys(), key=lambda k: results[k]["best_validation_accuracy"])
    best_result = results[best_depth_size]
    print(f"\nBest depth by validation accuracy: L={best_depth_size} with validation accuracy {best_result['best_validation_accuracy']:.4f}")

    epochs = list(range(0, len(best_result["train_losses_per_epoch"])))
    curves = {
        "Train Loss": (epochs, best_result["train_losses_per_epoch"]),
        "Validation Accuracy": (epochs, best_result["validation_accuracies_per_epoch"])
    }
    utils.plot(
        x_label="Epoch",
        y_label="Value",
        curves=curves,
        filename=str(out_dir / f"b_best_depth_L{best_depth_size}_trainloss_valacc.pdf")
    )
    print(f"Saved plot for best depth to {out_dir / f'b_best_depth_L{best_depth_size}_trainloss_valacc.pdf'}")

    # training loss vs val loss
    print("\nPlotting final training and validation loss curves for best depth...")
    curves_loss = {
        "Train Loss": (epochs, best_result["train_losses_per_epoch"]),
        "Validation Loss": (epochs, best_result["valid_losses_per_epoch"])
    }

    utils.plot(
        x_label="Epoch",
        y_label="Loss",
        curves=curves_loss,
        filename=str(out_dir / f"b_best_depth_L{best_depth_size}_loss_curves.pdf")
    )
    print(f"Saved training/validation loss curve to {out_dir / f'b_best_depth_L{best_depth_size}_loss_curves.pdf'}")

    # training accuracy vs val accuracy
    print("\nPlotting final training and validation accuracy curves for best depth...")
    curves_acc = {
        "Train Accuracy": (epochs, best_result["train_accuracies_per_epoch"]),
        "Validation Accuracy": (epochs, best_result["validation_accuracies_per_epoch"])
    }

    utils.plot(
        x_label="Epoch",
        y_label="Accuracy",
        curves=curves_acc,
        filename=str(out_dir / f"b_best_depth_L{best_depth_size}_accuracy_curves.pdf")
    )
    print(f"Saved training/validation accuracy curve to {out_dir / f'b_best_depth_L{best_depth_size}_accuracy_curves.pdf'}")
    
    report_out_path = out_dir / f"b_best_depth_L{best_depth_size}_test_accuracy.txt"
    with report_out_path.open("w") as f:
        f.write(f"Best depth by validation accuracy: L={best_depth_size}\n")
        f.write(f"Test accuracy at best validation: {best_result['test_accuracy_at_best_validation']:.6f}\n")
    print(f"Saved test accuracy report for best depth to {report_out_path}")

    #(c)
    print("\nPlotting final-epoch train accuracy vs depth...")
    x_axis = sorted(results.keys())
    y_axis = [results[L]["final_epoch_train_accuracy"] for L in x_axis]
    curves = {"Train Acc (final epoch)": (x_axis, y_axis)}
    utils.plot(
        x_label="Depth (number of hidden layers)",
        y_label="Train accuracy (final epoch)",
        curves=curves,
        filename=str(out_dir / "c_last_epoch_train_acc_vs_depth.pdf")
    )
    print(f"Saved final-epoch train-acc vs depth plot to {out_dir / 'c_last_epoch_train_acc_vs_depth.pdf'}")
    print("\nFinished.")

if __name__ == "__main__":
    main()
