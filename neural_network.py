from tensorflow.keras import layers, losses, optimizers, activations, metrics, regularizers
from tensorflow.keras.models import Sequential


def inverse_planning_model(height, width, output_len):
    """
    Create a Keras model for inverse planning

    :return: Keras model
    :rtype: Keras model
    """
    model = Sequential()
    
    # Todo: implement LeNet-5 model
    activation_func = activations.tanh
    lambda_l2 = 0

    # 1
    
    # 2
    model.add(layers.Dense(units=84,kernel_regularizer=regularizers.l2(lambda_l2), activation=activation_func,  input_shape=( height, width)))

    #3
    model.add(layers.Dense(units=84, activation=activation_func, kernel_regularizer=regularizers.l2(lambda_l2)))

    model.add(layers.Flatten())
    # 4
    model.add(layers.Dense(units=output_len, activation=activations.softmax, kernel_regularizer=regularizers.l2(lambda_l2)))

    # Para criar covolucao 2D
    # model.add(layers.Conv2D(filters=nf, kernel_size=(fx, fy), strides=(sx, sy), activation=activations.fun))

    # Para criar average pooling
    # model.add(layers.AveragePooling2D(pool_size=(px, py), strides=(sx,sy)))
    # Para definir camada de transição entre as camadas convolucionais e as densas.
    # model.add(layers.Flatten())
    # model.add(layers.Dense(units=num_neurons,activation=activations.fun))
    return model


def inverse_planning_model_simpleLenet5(width, height, output_len):
    """
    Create a Keras model for inverse planning

    :return: Keras model
    :rtype: Keras model
    """
    model = Sequential()
    
    # Todo: implement LeNet-5 model
    activation_func = activations.tanh
    lambda_l2 = 0

    # 1
    model.add(layers.Conv2D( kernel_size=(5, 5), kernel_regularizer=regularizers.l2(lambda_l2),
                            strides=(1, 1), activation=activation_func,  input_shape=(height, width)))
    # 2
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3
    model.add(layers.Flatten())

    model.add(layers.Dense(units=84, activation=activation_func, kernel_regularizer=regularizers.l2(lambda_l2)))
    model.add(layers.Dense(units=84, activation=activation_func, kernel_regularizer=regularizers.l2(lambda_l2)))

    # 4
    model.add(layers.Dense(units=output_len, activation=activations.softmax), kernel_regularizer=regularizers.l2(lambda_l2))

    # Para criar covolucao 2D
    # model.add(layers.Conv2D(filters=nf, kernel_size=(fx, fy), strides=(sx, sy), activation=activations.fun))

    # Para criar average pooling
    # model.add(layers.AveragePooling2D(pool_size=(px, py), strides=(sx,sy)))
    # Para definir camada de transição entre as camadas convolucionais e as densas.
    # model.add(layers.Flatten())
    # model.add(layers.Dense(units=num_neurons,activation=activations.fun))
    return model


def inverse_planning_model_Lenet5(width, height, output_len):
    """
    Create a Keras model for inverse planning

    :return: Keras model
    :rtype: Keras model
    """
    model = Sequential()
    
    # Todo: implement LeNet-5 model
    activation_func = activations.tanh
    lambda_l2 = 0

    # 1
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(lambda_l2),
                            strides=(1, 1), activation=activation_func,  input_shape=( height, width)))
    # 2
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(lambda_l2),
                            strides=(1, 1), activation=activation_func))

    # 4
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5
    model.add(layers.Conv2D(filters=120, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(lambda_l2),
                            strides=(1, 1), activation=activation_func))

    # 6
    model.add(layers.Flatten())
    model.add(layers.Dense(units=84, activation=activation_func, kernel_regularizer=regularizers.l2(lambda_l2)))

    # 7
    model.add(layers.Dense(units=output_len, activation=activations.softmax, kernel_regularizer=regularizers.l2(lambda_l2)))
    # Para criar covolucao 2D
    # model.add(layers.Conv2D(filters=nf, kernel_size=(fx, fy), strides=(sx, sy), activation=activations.fun))

    # Para criar average pooling
    # model.add(layers.AveragePooling2D(pool_size=(px, py), strides=(sx,sy)))
    # Para definir camada de transição entre as camadas convolucionais e as densas.
# model.add(layers.Flatten())
# model.add(layers.Dense(units=num_neurons,activation=activations.fun))
    return model


def make_lenet5(input_shape, output_len):
    model = Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh,
                            input_shape=input_shape))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=14, activation=activations.tanh))
    model.add(layers.Dense(units=output_len, activation=activations.softmax))

    return model

