from utils.feature_extraction import WordEmbed as w2v
from classifier import nn
import numpy as np
from utils.dataset import Data
from keras.utils import np_utils
import os

if __name__ == '__main__':
    dats = Data()
    data_root = os.path.join('utils')
    data_path = os.path.join(data_root, 'token-vectorized.bin')

    import pickle as pkl
    with open(data_path, 'rb') as file:
        datas = pkl.load(file)
    dats.load_data('utils/data.csv')

    targets = dats.y
    num_target = len(set(targets))
    targets = np_utils.to_categorical(targets, num_classes = num_target)

    model = nn.keras_model(datas, targets, nb_classes=num_target, hidden_units = 50)
    model.fit(datas, targets, batch_size = 300, epochs=10, verbose=1, metrics = ['mse', 'accuracy'])
    import pickle as pkl
    with open("beta-2.mdl", 'wb') as file:
        pkl.dump(model, file)
    # inp = np.array([docs[0]])
    # print(model.predict_classes(inp, batch_size = 1, verbose = 1))
