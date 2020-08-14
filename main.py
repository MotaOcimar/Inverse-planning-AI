import os
from time import time

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from utils import save_model_to_json
import numpy as np

from neural_network import *
import matplotlib.pyplot as plt
from data_generator import DataGenerator
# from neural_network import inverse_planning_model

num_epochs = 10000
num_iterations = 1
remaining = 0.8
num_alternatives = 4
data_generator = DataGenerator()
nn_input, nn_output = data_generator.generate_data(num_iterations, remaining, num_alternatives)
nn_input = np.array(nn_input)
nn_output = np.array(nn_output)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# for i in range(num_iterations*3):
#     plt.matshow(nn_input[i])
#     plt.show()
#     print(nn_output[i])

# model = inverse_planning_model()
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# model.fit(inputs, expected_outputs, batch_size, epochs)


def train():
    # treina a NN usando os dados  gerados por Monte Carlo
    height = len(nn_input[0])
    width = len(nn_input[0][0])
    output_len = len(nn_output)
    model = inverse_planning_model( width = width, height = height, output_len = output_len)
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    history = model.fit(nn_input, nn_output,
                    batch_size=(len(nn_input)), epochs=num_epochs)
    save_model_to_json(model, 'inverse_planning_model')


def evaluate():
    # Avalia o resultado obtido pela NN comparando com os objetivos reais dos agentes
    pass


if __name__ == "__main__":
    # train()
    pass
