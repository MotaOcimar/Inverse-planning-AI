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


def make_lenet5(input_shape, output_len):
    model = Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                            input_shape=input_shape))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=120, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=14, activation=activations.tanh))
    model.add(layers.Dense(units=output_len, activation=activations.softmax))

    return model
