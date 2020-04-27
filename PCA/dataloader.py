import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_HPC_log(log_file, train_ratio=0.8):
    print("Loading", log_file)
    struct_log = pd.read_csv(log_file)
    train_size = round(len(struct_log) * train_ratio)
    x_train = struct_log['Content'].iloc[:train_size]
    y_train = struct_log['ground_truth'].iloc[:train_size]

    x_test = struct_log['Content'].iloc[train_size:]
    y_test = struct_log['ground_truth'].iloc[train_size:]

    def print_status(x, y, status):
        num_total = len(x)
        num_pos = len(y.loc[y == 1])
        print('{}: {} instances, {} anomaly, {} normal'.format(status, num_total, num_pos, num_total - num_pos))

    print_status(struct_log.Content, struct_log.ground_truth, "Total")
    print_status(x_train, y_train, "Train")
    print_status(x_test, y_test, "Test")

    return (x_train, y_train), (x_test, y_test)


def load_HPC_log_splitted(train_file, test_file, target_file):
    print("Loading tagret file", target_file)
    y = np.load(target_file)
    if len(y.shape) > 1:
        y = y.squeeze()
    print("Loading train file ", train_file)
    df_train = pd.read_csv(train_file)
    x_train = df_train.to_numpy()
    if x_train.shape[1] > 16:
        x_train = x_train[:, -16:]
    mms_train = MinMaxScaler()
    x_train = mms_train.fit_transform(x_train)
    print(x_train.shape)
    train_size = len(df_train)
    y_train = y[:train_size]

    print("Loading test file ", test_file)
    df_test = pd.read_csv(test_file)
    x_test = df_test.to_numpy()
    x_test = mms_train.transform(x_test)
    print(x_test.shape)
    test_size = len(df_test)
    y_test = y[train_size:]

    if test_size + train_size != len(y):
        print("OHOH")

    return (x_train, y_train), (x_test, y_test)
