###======= Imports =======###
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from sklearn import preprocessing
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from alpha_vantage.timeseries import TimeSeries 


class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

# normalize
scaler = Normalizer()

###======= Config Dictionary =======###
config = {
    "alpha_vantage": {
        "key": "JG7TR7PA6FKE0DCG",
        "symbol": "MSFT",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.7,
        "val_split_size": 0.2,
        "test_split_size": 0.1
    }, 
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_test": "#561F78",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

###====== Utils Functions ======###
def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

def plot_data(config):
    data_date, data_close_price, num_data_points, display_date_range = download_data(config)

    # plot

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.show()

def pre_process_data(X):
    return scaler.fit_transform(X)


def prepare_data_x(x: np.ndarray, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # use the next day as label
    output = x[window_size:]
    return output


def plot_data_after_split(y_train, y_val, y_test, num_data_points, split_index_train, split_index_val, data_date):
    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)
    to_plot_data_y_test = np.zeros(num_data_points)

    to_plot_data_y_train[config["data"]["window_size"]:split_index_train+config["data"]["window_size"]] = scaler.inverse_transform(y_train)
    to_plot_data_y_val[split_index_train+config["data"]["window_size"]:split_index_val+config["data"]["window_size"]] = scaler.inverse_transform(y_val)
    to_plot_data_y_test[split_index_val+config["data"]["window_size"]:] = scaler.inverse_transform(y_test)

    to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_test = np.where(to_plot_data_y_test == 0, None, to_plot_data_y_test)

    ## plots

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
    plt.plot(data_date, to_plot_data_y_test, label="Prices (test)", color=config["plots"]["color_test"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - showing training and validation data")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()

def split_dataset_train_val_test(data_date, X, num_data_points, display_rate_range):
    X_normalized = pre_process_data(X)
    data_x, data_x_unseen = prepare_data_x(X_normalized, window_size=config["data"]["window_size"])
    data_y = prepare_data_y(X_normalized, window_size=config["data"]["window_size"])

    # Split dataset
    split_index_train = int(data_y.shape[0] * config["data"]["train_split_size"])
    split_index_val = int(data_y.shape[0] * config["data"]["val_split_size"]) + split_index_train
    
    data_x_train = data_x[:split_index_train]
    data_x_val = data_x[split_index_train:split_index_val]
    data_x_test = data_x[split_index_val:]
    data_y_train = data_y[:split_index_train]
    data_y_val = data_y[split_index_train:split_index_val]
    data_y_test = data_y[split_index_val:]

    # plot_data_after_split(data_y_train, data_y_val, data_y_test, num_data_points, split_index_train, split_index_val, data_date)

    return data_x_train, data_x_val, data_x_test, data_y_train, data_y_val, data_y_test

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

def get_dataset_loaders(x_train, x_val, x_test, y_train, y_val, y_test):
    dataset_train = TimeSeriesDataset(x_train, y_train)
    dataset_val = TimeSeriesDataset(x_val, y_val)
    dataset_test = TimeSeriesDataset(x_test, y_test)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)
    print("Test data shape", dataset_test.x.shape, dataset_test.y.shape)

    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=config["training"]["batch_size"], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def run():
    data_date, data_close_price, num_data_points, display_rate_range = download_data(config)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset_train_val_test(data_date, data_close_price, num_data_points, display_rate_range)
    return get_dataset_loaders(x_train, x_val, x_test, y_train, y_val, y_test)

###====== Testing ======###
if __name__ == "__main__":
    data_date, data_close_price, num_data_points, display_rate_range = download_data(config)
    