import numpy as np

from keras            import metrics
from keras.src.layers import Dense
from keras.src.models import Sequential

from sklearn.model_selection import train_test_split

from tensorflow.python.keras import Input


def prepare_data(x_early, y_early, test_size = 0.25, vald_size = 0.3, seed = 1):
    """
    Prepare data for training, testing and validation.

    Args:
        x_early (numpy.array): Input data.
        y_early (numpy.array): Output data.
        test_size (float, optional): Test size. Defaults to 0.25.
        vald_size (float, optional): Validation size. Defaults to 0.3.
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - np.array: Input data for training;
            - np.array: Input data for testing;
            - np.array: Input data for validation;
            - np.array: Output data for training;
            - np.array: Output data for testing;
            - np.array: Output data for validation.

    """
    np.random.seed(seed)

    x_train, x_test, y_train, y_test = train_test_split(x_early, y_early, test_size = test_size, random_state = seed)
    x_train, x_vald, y_train, y_vald = train_test_split(x_train, y_train, test_size = vald_size, random_state = seed)

    return x_train, x_test, x_vald, y_train, y_test, y_vald


def units_to_layers(units, activation = "relu"):
    """
    Convert a list of units to a list of Keras Dense layers.

    Args:
        units (list of int): List of units for each layer.
        activation (str, optional): Activation function. Defaults to “relu”.

    Returns:
        list of keras.src.layers.core.dense.Dense: List of Keras Dense layers.

    """
    layers = []

    if len(units) >= 2:
        for unit in units[0 : -1]:
            layers.append(Dense(unit, activation = activation))

        layers.append(Dense(units[-1]))

    return layers


def create_model(layers, input_dim, optimizer = "adam", loss = "mean_squared_error"):
    """
    Create a Keras model.

    Args:
        layers (list of keras.src.layers.core.dense.Dense): List of Keras Dense layers.
        input_dim (signedinteger): Input dimension.
        optimizer (str, optional): Optimizer. Defaults to “adam”.
        loss (str, optional): Loss function. Defaults to “mean_squared_error”.

    Returns:
        keras.src.models.sequential.Sequential: Keras model.

    """
    model = Sequential()

    Input(input_dim)

    for layer in layers:
        model.add(layer)

    model.compile(
        optimizer = optimizer, loss = loss, metrics = [metrics.MeanAbsoluteError()]
    )

    return model
