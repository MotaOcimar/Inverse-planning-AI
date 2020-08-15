import gzip
import numpy as np
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt


def display_image(image, title):
    image = image.squeeze()
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=plt.cm.gray_r)


def save_model_to_json(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + '.h5')


def load_model_from_json(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    return loaded_model


def greatest_equal_one(vec):
    """
    :param vec: base vector to transformation
    :type vec: numpy vector.
    """

    ret_vec = []
    for el in vec:
        if el == np.max(vec):
            ret_vec.append(1)
        else:
            ret_vec.append(0)
    return ret_vec
