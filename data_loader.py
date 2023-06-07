import numpy as np
import os


def read_data(directory_path="data/", label=0):
    x = []
    for item in os.listdir(directory_path):
        if item.startswith('.'):
            continue
        data = np.load(os.path.join(directory_path, item))
        for data_item in data:
            x.append(data_item)
    y = [label] * len(x)
    return x, y


def load_data(directory_paths="data/", labels=[], window_size=1, shape=(1, 60, 16), shuffle=True):
    for s in shape:
        assert s in [60, 16, 1]

    if isinstance(directory_paths, str):
        directory_paths = [directory_paths]
    x, y = [], []
    for d_path in directory_paths:
        for label_i, label in enumerate(labels):
            dx, dy = read_data(os.path.join(d_path, label), label_i)
            dx = np.array(dx)
            dy = np.array(dy)
            for n in range(0, len(dx)-window_size):
                x.append(dx[n: n + window_size])
                y.append(dy[n])

    x = _reshape(x, shape)
    x, y = np.array(x), np.array(y)
    if shuffle:
        ind = np.random.randint(len(y), size=len(y))
        x, y = x[ind], y[ind]
    return x, y


def _reshape(x_data, shape):
    shape = tuple(shape)
    if shape == (1, 60, 16):
        # Switch dimensions so that the (16) channels are in the last dimensions
        return np.transpose(np.array(x_data), (0, 1, 3, 2))
    elif shape == (60, 16, 1):
        return np.transpose(np.array(x_data), (0, 3, 2, 1))
    elif shape == (60, 16):
        return np.squeeze(np.transpose(np.array(x_data), (0, 1, 3, 2)))

    raise ValueError("Invalid shape", shape)


if __name__ == '__main__':
    # Example of loading data
    data_x, data_y = load_data("data/robert_sajina", labels=["left", "right", "jump", "none"])
    # Order of labels determines the corresponding y labels (0: left, 1:right, etc)

    # For multiple persons
    data_x, data_y = load_data(["data/robert_sajina", "data/zuza"], labels=["left", "right", "jump", "none"])
