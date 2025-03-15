from inc.ai.machine_learning import prepare_data, units_to_layers, create_model


def fit(x_early, y_early, units, test_size = 0.25, vald_size = 0.3, seed = 1):
    x_train, x_test, x_vald, y_train, y_test, y_vald = (
        prepare_data(x_early, y_early, test_size, vald_size, seed)
    )

    model = create_model(units_to_layers(units), (x_train.shape[1], 0))

    history = model.fit(
        x_train, y_train, validation_data = (x_vald, y_vald), epochs = 150, batch_size = 32
    )

    return model, history, (x_train, y_train), (x_test, y_test)
