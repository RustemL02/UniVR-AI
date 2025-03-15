import pandas as pd

from inc.graphics.plot   import *
from inc.utils.statistic import *

from src.machine_learning.artificial_neural_networks.deep_neural_networks import fit


def read_data(file_path):
    data = pd.read_csv(file_path, index_col = False, skipinitialspace = True)
    cols = data.columns.str.strip()

    plot_correlation_matrix(data)

    data_std = standardize_data(data, cols[1 : len(cols)])
    data_std = data_std.to_numpy()

    return data_std


def evaluation(values, labels, titles, function, legend_pos = "upper right"):
    series = []

    for i in range(len(values)):
        series.append({"x": range(len(values[i])), "y": values[i], "label": labels[i]})

    function(series, titles[0], titles[1], titles[2], legend_pos)


class Summarize:

    def __init__(self, model, history, test):
        self.model   = model
        self.history = history.history

        self.x_test, self.y_test = test

        print("\n")
        self.predictions = model.predict(self.x_test).reshape(1, -1)[0]
        print("\n")

        print("RMSE: {} \t NRMSE: {}".format(
            rmse(self.y_test, self.predictions), nrmse(self.y_test, self.predictions)
        ))


    def summarize_history(self):
        keys   = {
            "mae":     "mean_absolute_error",
            "los":     "loss",
            "val_mae": "val_mean_absolute_error",
            "val_los": "val_loss"
        }
        titles = {
            "mae": (
                "Model accuracy",
                "Epoch",
                "Accuracy"
            ),
            "los": (
                "Model loss",
                "Epoch",
                "Loss"
            )
        }

        mae_values = (self.history[keys["mae"]], self.history[keys["val_mae"]])
        los_values = (self.history[keys["los"]], self.history[keys["val_los"]])

        evaluation(mae_values, ("Train", "Test"), titles["mae"], Graphics.functions_map["plot"])
        evaluation(los_values, ("Train", "Test"), titles["los"], Graphics.functions_map["plot"])


    def summarize_outcome(self):
        titles = {
            "inference": (
                "Inference of log(NO2)",
                "Readings (row of file)",
                "log(NO2)"
            )
        }

        values = (self.y_test, self.predictions)
        labels = ("Real", "Inference")

        evaluation(values, labels, titles["inference"], Graphics.functions_map["scatter"])


    def summarize_abs_dif(self):
        titles = {
            "abs_diffr": (
                "Inference of log(NO2) - Absolute difference between inference and ground truth",
                "Absolute value of difference for log(NO2)",
                "log(NO2)"
            )
        }

        values = [abs(self.y_test - self.predictions)]
        labels = ["Absolute difference"]

        evaluation(values, labels, titles["abs_diffr"], Graphics.functions_map["bar"])


class Main:
    FILE_PATH = "data/dnn/traffic_meteorological_values_air_pollution.csv"

    if __name__ == "__main__":
        data = read_data(FILE_PATH)

        x = data[:, 1 : 7]
        y = data[:, 0]

        model, history, train, test = fit(x, y, [10, 30, 40, 1])

        summarize = Summarize(model, history, test)
        summarize.summarize_history()
        summarize.summarize_outcome()
        summarize.summarize_abs_dif()
