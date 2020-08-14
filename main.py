import os
from time import time

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from lenet5 import make_lenet5
from utils import read_mnist, save_model_to_json

from monte_carlo import generate_input_data
from neural_network import inverse_planning_model
import matplotlib.pyplot as plt
from data_generator import DataGenerator
# from neural_network import inverse_planning_model


num_iterations = 1
remaining = 0.8
data_generator = DataGenerator()
nn_input, nn_output = data_generator.generate_data(num_iterations, remaining)

for i in range(num_iterations*3):
    plt.matshow(nn_input[i])
    plt.plot(nn_output[i][1], nn_output[i][0], 'rx', markersize=8)
    plt.show()

# model = inverse_planning_model()
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# model.fit(inputs, expected_outputs, batch_size, epochs)


def train():
    # treina a NN usando os dados  gerados por Monte Carlo
    pass


def evaluate():
    # Avalia o resultado obtido pela NN comparando com os objetivos reais dos agentes
    pass
