import os
from time import time
import numpy as np
import random
import pickle

from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from utils import save_model_to_json, load_model_from_json, greatest_equal_one

from neural_network import *
from data_generator import DataGenerator

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_set_size = 9999  # Will be approximated to the nearest multiple of 3 (floor)
eval_set_size = 100  # Will be approximated to the nearest multiple of 3 (floor)
remaining = 0.8  # How much of the path will remain
num_alternatives = 4  # Number of possibilities of goals for the network to choose
one_map = True  # If True, just one map will be created

# Turn true to generate a new data set to train and evaluate
# If there is none, will be generated anyway
generate_new_data = False


def train():
    # treina a NN usando os dados  gerados
    nn_input, expected_output = load_data('train')

    num_epochs = 100
    # batch_size = 800  # Choose a value that your RAM can handle
    batch_size = len(nn_input)//2

    input_shape = nn_input[0].shape
    output_len = len(expected_output[0])

    model = make_lenet5(input_shape, output_len)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    tensorboard = TensorBoard(log_dir=os.path.join("logs", "{}".format(time())))
    model.fit(nn_input, expected_output, batch_size=batch_size, epochs=num_epochs, callbacks=[tensorboard])

    save_model_to_json(model, 'inverse_planning_model')


def evaluate():
    # Avalia o resultado obtido pela NN comparando com os objetivos reais dos agentes
    model = load_model_from_json('inverse_planning_model')
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    test_nn_input, expected_nn_output = load_data('eval')

    predicted_labels = model.predict(test_nn_input)
    model.summary()
    score = model.evaluate(test_nn_input, expected_nn_output)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    for i in range(20):
        index = random.randint(0, test_nn_input.shape[0])
        print('Example: {}. Expected Label: {}. Predicted Label: {}.'.format(index, expected_nn_output[index], greatest_equal_one(predicted_labels[index])))


def load_data(data_type):
    if data_type == 'train':
        set_size = train_set_size
        random.seed(1)
    elif data_type == 'eval':
        set_size = eval_set_size
        random.seed(2)
    else:
        raise Exception("Passe o tipo de dado a ser carregado ('train' ou 'eval')")

    if one_map:
        data_file_name = data_type + '-one_map'
    else:
        data_file_name = data_type + '-mult_map'

    data_file_name = data_file_name + '-' + str(set_size)
    data_file_name = data_file_name + '-' + str(remaining)
    data_file_name = data_file_name + '-' + str(num_alternatives)
    data_file_name = data_file_name + '.dat'

    if generate_new_data or not os.path.exists('./' + data_file_name):
        print('Generating ' + data_type + ' data...', end='')
        # Run and save train or evaluate set
        data_generator = DataGenerator()
        nn_input, expected_output = data_generator.generate_data(set_size//3, remaining, num_alternatives, one_map=True)
        nn_input = np.array(nn_input)
        expected_output = np.array(expected_output)

        print('Done')

        with open(data_file_name, 'wb') as file:
            pickle.dump([nn_input, expected_output], file)

        return nn_input, expected_output

    else:
        print('Loading ' + data_type + ' data from file...', end='')
        with open(data_file_name, 'rb') as file:
            [nn_input, expected_output] = pickle.load(file)

        print('Done')

        return nn_input, expected_output


if __name__ == "__main__":
    # train()
    evaluate()
