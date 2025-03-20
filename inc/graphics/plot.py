import numpy   as np
import seaborn as sns

from matplotlib import pyplot as plt


def plot_correlation_matrix(series, minimum = -1, maximum = 1, color = "coolwarm"):
    """
    Plots a correlation matrix for the given series.

    Args:
        series (pandas.core.frame.DataFrame): The data to be analyzed.
        minimum (float, optional): The minimum value for the color scale. Defaults to -1.
        maximum (float, optional): The maximum value for the color scale. Defaults to 1.
        color (str, optional): The color palette to be used. Defaults to “coolwarm”.

    Returns:
        `None`.

    """
    corr   = series.corr()
    matrix = np.triu(corr)

    sns.heatmap(
        corr, vmin = minimum, vmax = maximum, cmap = color, annot = True, annot_kws = {"size": 8}, mask = matrix
    )

    plt.show()


def create_graphics(series, title, xlabel, ylabel, function, legend_pos = "lower right"):
    """
    Create a graphics using the given series and plot function.

    Args:
        series (list of dict): A list of dictionaries, each containing:
            - “x” and “y” values for the data points;
            - “label” key for the legend.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        function (function): The plot function to be used to plot the data.
        legend_pos (str, optional): The position of the legend on the plot. Defaults to “lower right”.

    Returns:
        `None`.

    """
    plt.figure(figsize = (16, 9))

    for data in series:
        function(data["x"], data["y"], label = data["label"])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(loc = legend_pos)

    plt.show()


def bar(series, title, xlabel, ylabel, legend_pos = "lower right"):
    """
    Create a bar chart using the given series.

    Args:
        series (list of dict): A list of dictionaries, each containing:
            - “x” and “y” values for the data points;
            - “label” key for the legend.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        legend_pos (str, optional): The position of the legend on the plot. Defaults to “lower right”.

    Returns:
        `None`.

    """
    create_graphics(series, title, xlabel, ylabel, plt.bar, legend_pos)


def plot(series, title, xlabel, ylabel, legend_pos = "lower right"):
    """
    Create a plot using the given series.

    Args:
        series (list of dict): A list of dictionaries, each containing:
            - “x” and “y” values for the data points;
            - “label” key for the legend.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        legend_pos (str, optional): The position of the legend on the plot. Defaults to “lower right”.

    Returns:
        `None`.

    """
    create_graphics(series, title, xlabel, ylabel, plt.plot, legend_pos)


def scatter(series, title, xlabel, ylabel, legend_pos = "lower right"):
    """
    Create a scatter plot using the given series.

    Args:
        series (list of dict): A list of dictionaries, each containing:
            - “x” and “y” values for the data points;
            - “label” key for the legend.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        legend_pos (str, optional): The position of the legend on the plot. Defaults to “lower right”.

    Returns:
        `None`.

    """
    create_graphics(series, title, xlabel, ylabel, plt.scatter, legend_pos)


class Graphics:
    """
    A class to represent various plotting functions.

    Attributes:
        functions_map (dict): A mapping between the plotting function names and their implementations.

    """

    functions_map = {
        "bar":     bar,
        "plot":    plot,
        "scatter": scatter
    }
