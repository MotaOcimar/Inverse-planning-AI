import os
from time import time
import numpy as np
import random

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from utils import save_model_to_json, load_model_from_json

from neural_network import *
import matplotlib.pyplot as plt
from data_generator import DataGenerator
# from neural_network import inverse_planning_model



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# for i in range(num_iterations*3):
#     plt.matshow(nn_input[i])
#     plt.show()
#     print(nn_output[i])

# model = inverse_planning_model()
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# model.fit(inputs, expected_outputs, batch_size, epochs)

remaining = 0.8
num_alternatives = 4

def train():
    # treina a NN usando os dados  gerados por Monte Carlo
    num_epochs = 50
    num_iterations = 1000
    random.seed(1)
    data_generator = DataGenerator()
    nn_input, nn_output = data_generator.generate_data(num_iterations, remaining, num_alternatives)
    nn_input = np.array(nn_input)
    nn_output = np.array(nn_output)
    height = len(nn_input[0])
    width = len(nn_input[0][0])
    output_len = len(nn_output[0])
    model = inverse_planning_model( width = width, height = height, output_len = output_len)

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    # model.summary()
    
    history = model.fit(nn_input, nn_output,
                    batch_size=(len(nn_input)), epochs=num_epochs)
    save_model_to_json(model, 'inverse_planning_model')


def evaluate():
    # Avalia o resultado obtido pela NN comparando com os objetivos reais dos agentes
    model = load_model_from_json('inverse_planning_model')
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    num_iterations = 100
    random.seed(2)
    test_features = DataGenerator()
    test_nn_input, expected_nn_output = test_features.generate_data(num_iterations, remaining, num_alternatives)
    test_nn_input = np.array(test_nn_input)
    expected_nn_output = np.array(expected_nn_output)
    predicted_labels = model.predict(test_nn_input)
    model.summary()
    score = model.evaluate(test_nn_input, expected_nn_output)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    for i in range(20):
        index = random.randint(0, test_nn_input.shape[0])
        print('Example: {}. Expected Label: {}. Predicted Label: {}.'.format(index, expected_nn_output[index], greatest_equal_one(predicted_labels[index])))


def greatest_equal_one (vec):
    '''
    :param vec: base vector to transformation 
    :type vec: numpy vector.
    '''
  
    ret_vec = []
    for el in vec:
        if el == np.max(vec):
            ret_vec.append(1)
        else:
            ret_vec.append(0)
    return ret_vec 

if __name__ == "__main__":
    # print(nn_input.shape)
    train()
    # evaluate()
