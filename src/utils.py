import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

def get_best_device():
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(device)
    
    return device


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(data_path, bias=False):
    """
    Load EMNIST data (provided as a compressed npz file)
    """

    data_dict = np.load(data_path)
    print(f"Loaded dataset from {data_path} with keys: {list(data_dict.keys())}")

    # casting is necessary because the data is stored as uint8 to reduce file size
    X_train = data_dict["X_train"].astype(float) / 256  # normalizes between 0 and 1
    X_valid = data_dict["X_valid"].astype(float) / 256
    X_test = data_dict["X_test"].astype(float) / 256

    y_train = data_dict["y_train"].astype(int)
    y_valid = data_dict["y_valid"].astype(int)
    y_test = data_dict["y_test"].astype(int)

    if np.min(y_train) == 1:
        # This is necessary in case the dataset uses 1-based indexing for class labels
        y_train -= 1
        y_valid -= 1
        y_test -= 1

    if bias:
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) # X_train shape: (num_samples, num_features + 1) =  [f1, f2, ..., fn, bias=1]
        X_valid = np.hstack((X_valid, np.ones((X_valid.shape[0], 1))))
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

    return {
        "train": (X_train, y_train), "dev": (X_valid, y_valid), "test": (X_test, y_test), # Labels start from 0 and go up to 25 (num_classes = num letters on the alphabet)
    }


# data = load_dataset("data/emnist-letters.npz", bias=True)
# print(data["train"])

# image_to_show = 6
# img_and_bias = data["train"][0][image_to_show]
# print(f"Labels: {np.unique(data['train'][1])}")
# print(f"Label of image {image_to_show}: {data['train'][1][image_to_show]}")  
# img = img_and_bias[:-1].reshape(28, 28)  
# plt.imshow(img, cmap="gray")
# plt.show()


# curve_dict, key is label, value is (x, y)
def plot(x_label, y_label, curves, filename=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for curve_label, (x, y) in curves.items():
        plt.plot(x, y, label=curve_label)

    plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.clf()


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        """
        data: the dict returned by utils.load_pneumonia_data
        """
        train_X, train_y = data["train"]
        dev_X, dev_y = data["dev"]
        test_X, test_y = data["test"]

        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.dev_X = torch.tensor(dev_X, dtype=torch.float32)
        self.dev_y = torch.tensor(dev_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
