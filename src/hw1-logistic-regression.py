import argparse
import time
import pickle
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import utils

# -------------------- LogReg Helper functions --------------------

def softmax(Z):
    """
    Softmax over columns (classes). 
    Z: shape (num_classes, num_examples)
    
    returns A: same shape, softmax per column
    """
    Z_max = np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z - Z_max) # if I didn't subtract Z_max, could overflow (eg. exp(1000))
    sum_exp = np.sum(expZ, axis=0, keepdims=True)
    return expZ / sum_exp


def to_one_hot(y, num_classes):
    """
    y: shape (n_examples,) or (n_examples,1)
    
    returns Y_one_hot: shape (num_classes, n_examples)
    """
    y = np.array(y).reshape(-1)
    num_examples = y.shape[0]
    Y = np.zeros((num_classes, num_examples))
    Y[y, np.arange(num_examples)] = 1.0 # -> only the corresponding class is 1
    return Y


def center(X): # I noticed performs better than normalization (bruh, utils.load_dataset normalizes to [0,1] already...)
    """
    X: (num_features, num_examples)
    returns X_centered: same shape
    """
    means_vector = X.mean(axis=1, keepdims=True) # (num_features, 1)
    return (X - means_vector)


# -------------------- PCA helper functions --------------------

def pca_fit(X_centered, k):
    """
    Fit PCA on X_centered (normalized) and return (components, means_vector, explained_variance_ratio).
    X_centered: (num_features, num_examples)   -- centered data
    k: number of components to keep
    
    Returns:
    components: (num_features, k)       -- principal axes (columns are components)
    explained_variance_ratio: array shape (k,) share of variance per component
    """
    
    num_examples = X_centered.shape[1]

    # compute covariance matrix (num_features x num_features)
    Cov = np.dot(X_centered, X_centered.T) / num_examples # the covariance matrix is a square matrix giving the covariance between each pair of elements of the data 

    # eigen-decomposition (symmetric matrix) -> "eigh" for returning sorted ascending
    eigvals, eigvecs = np.linalg.eigh(Cov)  # eigenvectors are (nonzero) vectors that have their direction unchanged (or reversed) by a given linear transformation
                                            # in PCA, the eigenvectors of the covariance matrix represent the directions of maximum variance (principal components/axes)
    
    # sort descending by eigenvalue magnitude (this is, by variance explained)
    idx_desc = np.argsort(eigvals)[::-1] # get indices for sorting descending
    eigvals_sorted_desc = eigvals[idx_desc] # sorted eigenvalues descending
    eigvecs = eigvecs[:, idx_desc] # rearranged eigenvectors accordingly 

    # pick top-k
    components = eigvecs[:, :k]
    # explained variance ratio for top-k
    total_var = eigvals_sorted_desc.sum()
    explained_variance_ratio = (eigvals_sorted_desc[:k] / total_var) # main goal is to have this >= 0.9

    return components, explained_variance_ratio


def pca_transform(X_centered, components):
    """
    Takes the principal components obtained from pca_fit and projects X_centered onto them. FInal shape of X_reduced is (k, num_examples).
    X_centered: (num_features, num_examples)
    components: (num_features, k)
    
    returns: X_reduced (k, num_examples)
    """
    Z = np.dot(components.T, X_centered)  # (k, num_examples)
    return Z

# -------------------- LogisticRegression class --------------------

class LogisticRegression:
    """
    Multiclass logistic regression with L2 regularization and SGD updates.

    Expected input shapes:
      W: (num_classes, num_features)
      b: (num_classes, 1)
      X: (num_features, num_examples)
      Y_one_hot: (num_classes, num_examples)
    """

    def __init__(self, num_classes, num_features, learning_rate=0.0001, l2_penalty=0.00001, batch_size=1, seed=42):
        self.num_classes = num_classes
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.l2 = l2_penalty
        self.batch_size = batch_size

        rng = np.random.RandomState(seed)
        self.W = rng.randn(num_classes, num_features) * 0.01
        self.b = np.zeros((num_classes, 1))

    def compute_cost_and_grads(self, X, Y_one_hot):
        """
        X: (num_features, num_examples)
        Y_one_hot: (num_classes, num_examples)
        returns cost (scalar) and grads dW, db with same shapes as W and b
        """
        num_examples = X.shape[1]
        Z = np.dot(self.W, X) + self.b  # (num_classes, num_examples)
        A = softmax(Z)

        epsilon = 1e-12
        log_probs = np.log(A + epsilon) # to avoid log(0)
        loss = (1 / num_examples) * np.sum(Y_one_hot * -log_probs) # negative log likelihood (cross-entropy) -> see lecture03, slide 36
        regularization_term = (self.l2 / (2 * num_examples)) * np.sum(np.square(self.W))
        cost = loss + regularization_term

        dZ = A - Y_one_hot  # (num_classes, num_examples)
        dW = (1 / num_examples) * (dZ.dot(X.T))  # (num_classes, num_features)
        dW += (self.l2 / num_examples) * self.W
        db = (1 / num_examples) * np.sum(dZ, axis=1, keepdims=True)

        return cost, dW, db

    def update_on_batch(self, X_batch, Y_batch):
        """
        X_batch: (num_features, batch_size)
        Y_batch: (num_classes, batch_size)
        """
        _, dW, db = self.compute_cost_and_grads(X_batch, Y_batch)

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def train_epoch(self, X, Y_one_hot, shuffle=True):
        """
        X: (num_features, n_examples)
        Y_one_hot: (num_classes, n_examples)
        """
        num_examples = X.shape[1]
        indices = np.arange(num_examples)
        if shuffle:
            np.random.shuffle(indices) # so that each epoch sees data in different order

        epoch_costs = []
        for start in range(0, num_examples, self.batch_size):
            end = min(start + self.batch_size, num_examples)
            batch_idx = indices[start:end]
            X_batch = X[:, batch_idx]
            Y_batch = Y_one_hot[:, batch_idx]

            cost, _, _ = self.compute_cost_and_grads(X_batch, Y_batch)
            epoch_costs.append(cost)

            self.update_on_batch(X_batch, Y_batch)

        return float(np.mean(epoch_costs))

    def predict(self, X):
        """
        X: (num_features, num_examples)
        returns predictions: shape (num_examples,) integers
        """
        Z = np.dot(self.W, X) + self.b
        A = softmax(Z)
        predictions = np.argmax(A, axis=0)
        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        y = np.array(y).reshape(-1)
        return float(np.mean(predictions == y))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


# -------------------- main() --------------------

def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train_raw, y_train = data["train"]
    X_valid_raw, y_valid = data["dev"]
    X_test_raw, y_test = data["test"]
    num_classes = np.unique(y_train).size

    X_train_used = X_train_raw.T
    X_valid_used = X_valid_raw.T
    X_test_used = X_test_raw.T
    
    # Center data -> works better than normalization
    X_train_used = center(X_train_used)
    X_valid_used = center(X_valid_used)
    X_test_used = center(X_test_used)

    # raw and PCA representations (both centered)
    # raw: (num_features, num_examples)
    X_raw_train = X_train_used
    X_raw_valid = X_valid_used
    X_raw_test = X_test_used
    num_features_raw = X_raw_train.shape[0]

    # pca
    k = args.pca_k
    components, explained_var_ratio = pca_fit(X_train_used, k)
    print(f"PCA: kept {k} components, explained variance ratio sum: {explained_var_ratio.sum():.4f}")
    # project to PCA space: each result is (k, num_examples) where k is the number of components to keep (features)
    X_pca_train = pca_transform(X_train_used, components)
    X_pca_valid = pca_transform(X_valid_used, components)
    X_pca_test = pca_transform(X_test_used, components)
    num_features_pca = k

    # hyperparameters grid
    lr_list = [0.001, 0.0001, 0.00001]       # 3 learning rate values
    l2_list = [0.00001, 0.0001]              # 2 l2 penalty values 
    feature_options = [("raw", X_raw_train, X_raw_valid, X_raw_test, num_features_raw),
                       ("pca", X_pca_train, X_pca_valid, X_pca_test, num_features_pca)]

    # storage for results
    grid_results = []

    # ensure base save_path directory exists
    save_base_path_name, save_extension = os.path.splitext(args.save_path)

    # loop over all combinations
    for lr in lr_list:
        for l2 in l2_list:
            for feature_representation, X_train, X_valid, X_test, feat_dim in feature_options:
                print(f"\nRunning config: lr={lr}, l2={l2:.5f}, feature_representation={feature_representation}, epochs={args.epochs}")

                # setup model for this config
                model = LogisticRegression(
                    num_classes=num_classes,
                    num_features=int(feat_dim),
                    learning_rate=lr,
                    l2_penalty=l2,
                    batch_size=args.batch_size,
                    seed=args.seed
                )

                # one-hot labels
                Y_train = to_one_hot(y_train, num_classes)
                Y_valid = to_one_hot(y_valid, num_classes)
                Y_test = to_one_hot(y_test, num_classes)

                # prepare config-specific save path and accuracy plot path
                config_tag = f"_lr{lr}_l2{l2:.5f}_{feature_representation}"
                config_save = f"{save_base_path_name}{config_tag}{save_extension}"
                config_plot = f"{save_base_path_name}{config_tag}_accs.pdf"

                best_valid = 0.0
                best_epoch = -1
                train_accuracies = []
                valid_accuracies = []

                start = time.time()
                for epoch in range(1, args.epochs + 1):
                    print(f"Training epoch {epoch} for config {config_tag}")
                    train_cost = model.train_epoch(X_train, Y_train, shuffle=True)

                    train_acc = model.evaluate(X_train, y_train)
                    valid_acc = model.evaluate(X_valid, y_valid)

                    train_accuracies.append(train_acc)
                    valid_accuracies.append(valid_acc)

                    print('epoch: {:2d} | train cost: {:.6f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                        epoch, train_cost, train_acc, valid_acc))

                    if valid_acc > best_valid:
                        best_valid = valid_acc
                        best_epoch = int(epoch)
                        model.save(config_save)

                elapsed_time = time.time() - start
                print('Training for this config took {:.1f} seconds'.format(elapsed_time))

                # reload best checkpoint for this config and evaluate on test
                best_model = LogisticRegression.load(config_save)
                test_acc = best_model.evaluate(X_test, y_test)

                print(f"Config {config_tag}: best_val={best_valid:.4f} (epoch {best_epoch}), test={test_acc:.4f}")

                # save per-config accuracies plot
                try:
                    utils.plot(
                        "Epoch", "Accuracy",
                        {"train": (np.arange(1, args.epochs + 1), train_accuracies),
                         "valid": (np.arange(1, args.epochs + 1), valid_accuracies)},
                        filename=config_plot
                    )
                except Exception as e:
                    # plotting failure should not stop grid
                    print("Plotting failed for config", config_tag, "with error:", e)

                # store result
                grid_results.append({
                    "learning_rate": float(lr),
                    "l2_penalty": float(l2),
                    "features": feature_representation,
                    "best_valid": float(best_valid),
                    "selected_epoch": int(best_epoch),
                    "test": float(test_acc),
                    "time": elapsed_time,
                    "model_path": config_save,
                    "acc_plot": config_plot
                })

    # after grid, find best configuration by validation accuracy
    best_cfg = max(grid_results, key=lambda r: r["best_valid"])
    print("\nGrid search complete. Results (validation accuracy for each config):")
    for r in grid_results:
        print(f"lr={r['learning_rate']:.0e}, l2={r['l2_penalty']:.0e}, features={r['features']}, best_val={r['best_valid']:.4f}, test={r['test']:.4f}")

    print("\nBest configuration by validation accuracy:")
    print(best_cfg)

    # save grid summary to args.scores
    with open(args.scores, "w") as f:
        json.dump({
            "grid_results": grid_results,
            "best_config": best_cfg
        }, f, indent=4)

    print("\nSaved grid summary to", args.scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="data/emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="q1-logreg/q1-logreg-best.model")
    parser.add_argument("--accuracy-plot", default="q1-logreg/q1-logreg-accs.pdf")
    parser.add_argument("--scores", default="q1-logreg/q1-logreg-scores.json")

    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--l2-penalty', type=float, default=0.00001)
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--use-pca', action='store_true', help='If set, apply PCA feature reduction before training')
    parser.add_argument('--pca-k', type=int, default=90, help='Number of PCA components to keep (if --use-pca).')

    args = parser.parse_args()
    main(args)


"""
a) Implement logistic regression with l2 regularization and a stochastic gradient descent 
(SGD) weight update rule. Report results using a learning
rate of 0.0001 and an l2 penalty of 0.00001. As in the perceptron exercise, train for
20 epochs, plot training and validation accuracies for each epoch, and report the test
accuracy of the best-performing checkpoint.

    "best_config": {
        "learning_rate": 0.0001,
        "l2_penalty": 1e-05,
        "features": "raw",
        "best_valid": 0.6988461538461539,
        "selected_epoch": 20,
        "test": 0.7022115384615385, --------------
        "time": 415.82319593429565,
        "model_path": "q1-logreg/q1-logreg-besta_lr0.0001_l20.00001.model",
        "acc_plot": "q1-logreg/q1-logreg-besta_lr0.0001_l20.00001_accs.pdf"
    }
"""


# b
"""
The 784-dimensional pixel input suggests we should reduce dimensionality. Dimensionality-reduction techniques preserve the essential information while lowering the number of predictors, which improves generalizability and cuts training time and memory. In class we covered PCA. Other options are LDA, t-SNE and autoencoders (also approached in class).

After deeper research we concluded PCA fits our needs: simple to implement, computationally efficient and widely used. PCA reduces dimensionality while preserving most important information. EMNIST images (28x28 = 784 features) contain correlated and noisy pixels (neighboring pixels, blank margins around letters, correlated stroke patterns), so many features are redundant. PCA finds an orthogonal basis (principal components) that captures directions of largest variance, combining redundant variables so that keeping the top-k components retains most signal with far fewer dimensions. This yields faster training, a smaller model (fewer weights) and less memory. Projecting onto top components discards low-variance (often noisy) dimensions, which can improve generalization for a simple linear model. The choice of k can be based on explained-variance ratio (e.g., pick k so >=90% variance is retained), making k measurable. We can test various k to trade off reduction vs. information loss. PCA is unsupervised, so it reduces dimensions without using class labels.

Mathematically, our PCA procedure:

1. Start from normalization (utils.load_dataset scales to [0,1]) and center by subtracting the mean across the training set.
2. Compute covariance matrix of centered data. 
This is a symmetric square matrix giving the covariance between each pair of elements of a given vector/matrix. 
It's diagonal contains variances (i.e., the covariance of each element/feature with itself). 
3. Perform eigen-decomposition of the covariance matrix to obtain eigenvalues and eigenvectors. 
Eigenvectors represent the directions of maximum variance (principal components), and eigenvalues indicate the amount of variance captured by each component. 
More precisely, an eigenvector v of a linear transformation T is scaled by a constant factor λ when the linear transformation is applied to it: Tv = λv. 
The corresponding eigenvalue is the multiplying factor λ (possibly a negative or complex number). Which means that the direction of the eigenvector remains unchanged (or reversed) by the transformation.
And the bigger size of the eigenvalue, the bigger the eigenvector is and the more variance it explains.
4. Sort the eigenvalues in descending order and select the top-k eigenvectors (principal components) corresponding to the largest eigenvalues.
5. Project the centered data onto the selected principal components to obtain the reduced-dimensional representation.

Thus, after testing several k values, we settled on k = 90 components, which retains an explained-variance ratio sum of 0.9315 (93%). This means we keep most of the original variability while reducing the dimensionality from 784 to 90. The training time dropped significantly, and we maintained almost the same best validation and test metrics. For example, with learning_rate = 0.0001 and l2_penalty = 0.00001, training time decreased from ~400s to ~160s, while the best validation accuracy only slightly decreased from 69.92% to 69.63%, and the test accuracy dropped only minimally from 70.27% to 70.00%.
So, PCA is a straightforward, effective fit for our task.

References:
https://drive.google.com/file/d/1iGyC2WbQa21mzGIxAKNtBY0CoHjGjw53/view?usp=sharing (ML class from Bachelor degree)
https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)/07%3A_Spectral_Theory/7.01%3A_Eigenvalues_and_Eigenvectors_of_a_Matrix
https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/04%3A_Expected_Value/4.08%3A_Expected_Value_and_Covariance_Matrices
https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/04%3A_Expected_Value/4.05%3A_Covariance_and_Correlation
https://www.ibm.com/think/topics/dimensionality-reduction
https://www.ibm.com/think/topics/principal-component-analysis
https://arxiv.org/pdf/1404.1100
"""



"""
As in previous exercises, for each configuration train for 20 epochs and save the checkpoint with the
best validation accuracy. After you run the grid, report
• The validation accuracy of every configuration
• The test accuracy of the configuration with the best validation accuracy

    "grid_results": [
        {
            "learning_rate": 0.001,
            "l2_penalty": 1e-05,
            "features": "raw",
            "best_valid": 0.7201923076923077, -------------------------
            "selected_epoch": 20,
            "test": 0.721875,
            "time": 354.25238394737244,
            "model_path": "q1-logreg/q1-logreg-best_lr0.001_l20.00001_raw.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.001_l20.00001_raw_accs.pdf"
        },
        {
            "learning_rate": 0.001,
            "l2_penalty": 1e-05,
            "features": "pca",
            "best_valid": 0.7140865384615385, -------------------------
            "selected_epoch": 18,
            "test": 0.7158653846153846,
            "time": 157.85904669761658,
            "model_path": "q1-logreg/q1-logreg-best_lr0.001_l20.00001_pca.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.001_l20.00001_pca_accs.pdf"
        },
        {
            "learning_rate": 0.001,
            "l2_penalty": 0.0001,
            "features": "raw",
            "best_valid": 0.720625, -------------------------
            "selected_epoch": 17,
            "test": 0.7222596153846154,
            "time": 359.7671411037445,
            "model_path": "q1-logreg/q1-logreg-best_lr0.001_l20.00010_raw.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.001_l20.00010_raw_accs.pdf"
        },
        {
            "learning_rate": 0.001,
            "l2_penalty": 0.0001,
            "features": "pca",
            "best_valid": 0.7151442307692307, -------------------------
            "selected_epoch": 19,
            "test": 0.7153846153846154,
            "time": 188.02508544921875,
            "model_path": "q1-logreg/q1-logreg-best_lr0.001_l20.00010_pca.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.001_l20.00010_pca_accs.pdf"
        },
        {
            "learning_rate": 0.0001,
            "l2_penalty": 1e-05,
            "features": "raw",
            "best_valid": 0.6992307692307692, -------------------------
            "selected_epoch": 20,
            "test": 0.7027403846153846,
            "time": 401.8315472602844,
            "model_path": "q1-logreg/q1-logreg-best_lr0.0001_l20.00001_raw.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.0001_l20.00001_raw_accs.pdf"
        },
        {
            "learning_rate": 0.0001,
            "l2_penalty": 1e-05,
            "features": "pca",
            "best_valid": 0.6963461538461538, -------------------------
            "selected_epoch": 20,
            "test": 0.7000480769230769,
            "time": 165.4070749282837,
            "model_path": "q1-logreg/q1-logreg-best_lr0.0001_l20.00001_pca.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.0001_l20.00001_pca_accs.pdf"
        },
        {
            "learning_rate": 0.0001,
            "l2_penalty": 0.0001,
            "features": "raw",
            "best_valid": 0.6985576923076923, -------------------------
            "selected_epoch": 20,
            "test": 0.7022596153846153,
            "time": 386.7949867248535,
            "model_path": "q1-logreg/q1-logreg-best_lr0.0001_l20.00010_raw.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.0001_l20.00010_raw_accs.pdf"
        },
        {
            "learning_rate": 0.0001,
            "l2_penalty": 0.0001,
            "features": "pca",
            "best_valid": 0.6961057692307693, -------------------------
            "selected_epoch": 20,
            "test": 0.6994711538461539,
            "time": 180.63509154319763,
            "model_path": "q1-logreg/q1-logreg-best_lr0.0001_l20.00010_pca.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr0.0001_l20.00010_pca_accs.pdf"
        },
        {
            "learning_rate": 1e-05,
            "l2_penalty": 1e-05,
            "features": "raw",
            "best_valid": 0.6212019230769231, -------------------------
            "selected_epoch": 20,
            "test": 0.6236057692307693,
            "time": 373.1622657775879,
            "model_path": "q1-logreg/q1-logreg-best_lr1e-05_l20.00001_raw.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr1e-05_l20.00001_raw_accs.pdf"
        },
        {
            "learning_rate": 1e-05,
            "l2_penalty": 1e-05,
            "features": "pca",
            "best_valid": 0.620576923076923, -------------------------
            "selected_epoch": 20,
            "test": 0.6230769230769231,
            "time": 154.29423427581787,
            "model_path": "q1-logreg/q1-logreg-best_lr1e-05_l20.00001_pca.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr1e-05_l20.00001_pca_accs.pdf"
        },
        {
            "learning_rate": 1e-05,
            "l2_penalty": 0.0001,
            "features": "raw",
            "best_valid": 0.6210096153846154, -------------------------
            "selected_epoch": 20,
            "test": 0.6235576923076923,
            "time": 348.17209339141846,
            "model_path": "q1-logreg/q1-logreg-best_lr1e-05_l20.00010_raw.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr1e-05_l20.00010_raw_accs.pdf"
        },
        {
            "learning_rate": 1e-05,
            "l2_penalty": 0.0001,
            "features": "pca",
            "best_valid": 0.620576923076923, -------------------------
            "selected_epoch": 20,
            "test": 0.6230288461538461,
            "time": 156.14168858528137,
            "model_path": "q1-logreg/q1-logreg-best_lr1e-05_l20.00010_pca.model",
            "acc_plot": "q1-logreg/q1-logreg-best_lr1e-05_l20.00010_pca_accs.pdf"
        }
    ],

    "best_config": {
        "learning_rate": 0.001,
        "l2_penalty": 0.0001,
        "features": "raw",
        "best_valid": 0.720625,
        "selected_epoch": 17,
        "test": 0.7222596153846154, -----------------
        "time": 359.7671411037445,
        "model_path": "q1-logreg/q1-logreg-best_lr0.001_l20.00010_raw.model",
        "acc_plot": "q1-logreg/q1-logreg-best_lr0.001_l20.00010_raw_accs.pdf"
    }
"""