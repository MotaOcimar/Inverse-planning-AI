from tensorflow.keras import layers, losses, optimizers, activations, metrics, regularizers
from tensorflow.keras.models import Sequential


def inverse_planning_model(height, width, output_len):
    """
    Create a Keras model for inverse planning

    :return: Keras model
    :rtype: Keras model
    """
    activation_func = activations.tanh
    lambda_l2 = 0.00

    model = Sequential()
    # 1
    model.add(layers.Dense(units=168, kernel_regularizer=regularizers.l2(lambda_l2), activation=activation_func,
                           input_shape=(height, width)))

    # 2
    model.add(layers.Dense(units=84, activation=activation_func, kernel_regularizer=regularizers.l2(lambda_l2)))

    # 3
    model.add(layers.Dense(units=84, activation=activation_func, kernel_regularizer=regularizers.l2(lambda_l2)))
    model.add(layers.Flatten())

    # 4
    model.add(layers.Dense(units=output_len, activation=activations.softmax,
                           kernel_regularizer=regularizers.l2(lambda_l2)))

    return model
