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


n_layers = 1
n_hidden_units_list = [16, 32, 64, 128, 256]
lr_list = [0.01, 0.001, 0.0001, 0.00001]
dropout_list = [0.0, 0.3]
l2_list = [0.0, 0.0001]
activation = 'relu'
batch_size = 64
n_epochs = 30
optimizer_name = 'adam'

device = utils.get_best_device()
print("Using device:", device)


def run_experiment(hidden_size, lr, dropout, l2, data_path):
    utils.configure_seed(seed=42)

    data = utils.load_dataset(data_path)
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device).manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
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

    # move whole-dataset tensors to device 
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

    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    
    train_losses.append(initial_train_loss)
    train_accuracies.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accuracies.append(initial_val_acc)
    
    best_val_acc = initial_val_acc
    best_state = copy.deepcopy(model.state_dict())

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
        "validation_accuracies_per_epoch": valid_accuracies
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output-dir', type=str, default="q2-ffn-2/", help="Output directory path")
    parser.add_argument('-data', type=str, default="data/emnist-letters.npz", help="Path to dataset file")
    args = parser.parse_args()

    data = utils.load_dataset(args.data)
    dataset = utils.ClassificationDataset(data)
    print(f"Dataset sizes: train {dataset.X.shape[0]}, dev {dataset.dev_X.shape[0]}, test {dataset.test_X.shape[0]}")
    print(f"Features: {dataset.X.shape[1]}, Classes: {torch.unique(dataset.y).shape[0]}")
    print("\nGrid specs:")
    print("hidden sizes:", n_hidden_units_list)
    print("lrs:", lr_list)
    print("dropouts:", dropout_list)
    print("l2s:", l2_list)
    print("\nOther params:")
    print("n_layers:", n_layers)
    print("optimizer:", optimizer_name)
    print("activation:", activation)
    print("batch_size:", batch_size)
    print("n_epochs:", n_epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = output_dir / "results_grid_search.csv"

    with csv_output_path.open("w", newline="") as csvfile:
        fieldnames = [
            "hidden_size", "lr", "dropout", "l2",
            "initial_validation_accuracy", "best_validation_accuracy",
            "test_accuracy_at_best_validation", "test_loss_at_best_validation",
            "train_accuracy_at_best_validation", "train_loss_at_best_validation"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        start_time = time.time()
        total_runs = len(n_hidden_units_list) * len(lr_list) * len(dropout_list) * len(l2_list)
        run_idx = 0

        best_overall_val = -1.0
        best_overall_row = None
        best_overall_statistics = None
        best_per_hidden_size = {}  # hidden_size -> row dict (best by val acc for that width)

        for hidden in n_hidden_units_list:
            for lr in lr_list:
                for dropout in dropout_list:
                    for l2 in l2_list:
                        run_idx += 1
                        print(f"\nRun {run_idx}/{total_runs} -> hidden={hidden}, lr={lr}, dropout={dropout}, l2={l2}")
                        current_experiment_result = run_experiment(hidden, lr, dropout, l2, args.data)

                        writer.writerow({
                            "hidden_size": current_experiment_result["hidden_size"],
                            "lr": current_experiment_result["lr"],
                            "dropout": current_experiment_result["dropout"],
                            "l2": current_experiment_result["l2"],
                            "initial_validation_accuracy": f"{current_experiment_result['initial_validation_accuracy']:.6f}",
                            "best_validation_accuracy": f"{current_experiment_result['best_validation_accuracy']:.6f}",
                            "test_accuracy_at_best_validation": f"{current_experiment_result['test_accuracy_at_best_validation']:.6f}",
                            "test_loss_at_best_validation": f"{current_experiment_result['test_loss_at_best_validation']:.6f}",
                            "train_accuracy_at_best_validation": f"{current_experiment_result['train_accuracy_at_best_validation']:.6f}",
                            "train_loss_at_best_validation": f"{current_experiment_result['train_loss_at_best_validation']:.6f}"
                        })
                        csvfile.flush()

                        if current_experiment_result["best_validation_accuracy"] > best_overall_val:
                            best_overall_val = current_experiment_result["best_validation_accuracy"]
                            best_overall_row = current_experiment_result
                            #  for plotting
                            best_overall_statistics = {
                                "train_losses_per_epoch": current_experiment_result["train_losses_per_epoch"],
                                "train_accuracies_per_epoch": current_experiment_result["train_accuracies_per_epoch"],
                                "valid_losses_per_epoch": current_experiment_result["valid_losses_per_epoch"],
                                "validation_accuracies_per_epoch": current_experiment_result["validation_accuracies_per_epoch"]
                            }

                        # update best per hidden width
                        hidden_size = int(current_experiment_result["hidden_size"])
                        if hidden_size not in best_per_hidden_size or current_experiment_result["best_validation_accuracy"] > best_per_hidden_size[hidden_size]["best_validation_accuracy"]:
                            best_per_hidden_size[hidden_size] = current_experiment_result

        elapsed = time.time() - start_time
        print(f"\nAll runs finished. Results written to {csv_output_path}. Elapsed seconds: {int(elapsed)}")

    #(a)
    print("\nWriting best configuration per hidden size (by validation accuracy) to file...")
    results_a_path = output_dir / "a_best_per_hidden_size.txt"
    with results_a_path.open("w") as f:
        f.write("Best configuration per hidden size (by validation accuracy):\n")
        f.write("{:>12s} {:>12s} {:>10s} {:>10s} {:>20s} {:>20s}\n".format(
            "hidden_size", "lr", "dropout", "l2", "best_validation_accuracy", "test_accuracy_at_best_validation"
        ))
        for hidden_size in sorted(best_per_hidden_size.keys()):
            experiment_result = best_per_hidden_size[hidden_size]
            line = "{:>12d} {:>12.5g} {:>10.2f} {:>10.6g} {:>20.4f} {:>20.4f}\n".format(
                int(experiment_result["hidden_size"]), experiment_result["lr"], experiment_result["dropout"], 
                experiment_result["l2"], experiment_result["best_validation_accuracy"], experiment_result["test_accuracy_at_best_validation"]
            )
            f.write(line)
    
    print(f"Saved best configuration per hidden size results to {results_a_path}")

    #(b) -> decided to use the 3 plots to be sure everything is guaranteed to be present regarding the exercise instructions
    print("\nPlotting best overall validation accuracy model...")
    if best_overall_statistics is not None and best_overall_row is not None:
        # train loss and val acc   
        epochs = list(range(1, len(best_overall_statistics["train_losses_per_epoch"]) + 1))

        curves = {
            "Train Loss": (epochs, best_overall_statistics["train_losses_per_epoch"]),
            "Validation Accuracy": (epochs, best_overall_statistics["validation_accuracies_per_epoch"])
        }

        utils.plot(
            x_label="Epoch",
            y_label="Value",
            curves=curves,
            filename=str(output_dir/"b_best_overall_train_loss_and_val_acc.pdf")
        )

        print(f"Saved best overall validation accuracy model plot to {output_dir / 'b_best_overall_train_loss_and_val_acc.pdf'}")

        # training vs val loss 
        print("\nPlotting train vs validation losses for best overall model...")
        curves_loss = {
            "Train Loss": (epochs, best_overall_statistics["train_losses_per_epoch"]),
            "Validation Loss": (epochs, best_overall_statistics["valid_losses_per_epoch"])
        }

        utils.plot(
            x_label="Epoch",
            y_label="Loss",
            curves=curves_loss,
            filename=str(output_dir / "b_best_overall_train_and_val_loss.pdf")
        )
        
        print(f"Saved train vs validation loss plot to {output_dir / 'b_best_overall_train_and_val_loss.pdf'}")
        
        # training vs val accuracy 
        print("\nPlotting train vs validation accuracies for best overall model...")
        curves_acc = {
            "Train Accuracy": (epochs, best_overall_statistics["train_accuracies_per_epoch"]),
            "Validation Accuracy": (epochs, best_overall_statistics["validation_accuracies_per_epoch"])
        }

        utils.plot(
            x_label="Epoch",
            y_label="Accuracy",
            curves=curves_acc,
            filename=str(output_dir / "b_best_overall_train_and_val_accuracy.pdf")
        )

        print(f"Saved train vs validation accuracy plot to {output_dir / 'b_best_overall_train_and_val_accuracy.pdf'}")
        print("Best overall configuration (by validation acc):")
        print(f" hidden={best_overall_row['hidden_size']}, lr={best_overall_row['lr']}, dropout={best_overall_row['dropout']}, l2={best_overall_row['l2']}")
        print(f" best_val_acc={best_overall_row['best_validation_accuracy']:.4f}, test_acc_at_best_val={best_overall_row['test_accuracy_at_best_validation']:.4f}")

    #(c)
    print("\nPlotting train-accuracy at best validation of each hidden size vs hidden size...")
    widths = []
    best_train_accuracies_per_hidden_size = []

    for hidden_size in sorted(best_per_hidden_size.keys()):
        widths.append(hidden_size)
        best_train_accuracies_per_hidden_size.append(best_per_hidden_size[hidden_size]["train_accuracy_at_best_validation"])

    curves = {"Train Accuracy at Best Val": (widths, best_train_accuracies_per_hidden_size)}
    
    utils.plot(
        x_label="Hidden Units",
        y_label="Train Accuracy (at best val)",
        curves=curves,
        filename=str(output_dir / "c_train_acc_vs_width.pdf")
    )

    print(f"Saved train-acc vs width plot to {output_dir / 'c_train_acc_vs_width.pdf'}")
    print("\nFinished.")


if __name__ == "__main__":
    main()
