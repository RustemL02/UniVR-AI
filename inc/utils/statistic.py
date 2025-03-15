import math

from sklearn.metrics       import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def rmse(real, pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        real (numpy.ndarray): Array of true values.
        pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Root Mean Squared Error (RMSE).

    """
    return math.sqrt(mean_squared_error(real, pred))


def nrmse(real, pred):
    """
    Calculates the Normalized Root Mean Squared Error (NRMSE) between true and predicted values.

    The NRMSE is normalized by the range of the true values (max - min).

    Args:
        real (numpy.ndarray): Array of true values.
        pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Normalized Root Mean Squared Error (NRMSE).

    """
    return math.sqrt(rmse(real, pred) / (max(real) - min(real)))


def standardize_data(series, columns):
    """
    Standardizes specified columns in a Pandas DataFrame using `MinMaxScaler`.

    This function scales the values of the given columns to a range between -1 and 1.
    It utilizes the `MinMaxScaler` from scikit-learn to perform the standardization.

    Args:
        series (pandas.core.frame.DataFrame): The data to be standardized.
        columns (list of string): A list of column names that should be standardized.

    Returns:
        pandas.core.frame.DataFrame: The standardized data.

    """
    scaler = MinMaxScaler(feature_range = (-1, 1))

    for column in columns:
        data = series[column]

        series[column] = scaler.fit_transform(data.values.reshape(-1, 1))

    return series
