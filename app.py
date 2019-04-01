from utils.feature_extraction import Word_embed as w2v
from classifier import nn
import numpy as np
from keras.utils import np_utils

def sen2vec(sentence, words = 20, length = 150, vectors = None):
    res = []
    x = 0
    zeros = np.zeros(150)
    if vectors != None:
        while x < 20 and x < len(sentence):
            word = sentence[x]
            try:
                vector = vectors[word]
            except KeyError:
                vector = zeros
            res.append(vector)
            x += 1
    gap = words - len(res)
    pad =  [zeros] * gap
    res.extend(pad)
    return np.array(res)

if __name__ == '__main__':
    pool_dummy = [
        'aku makan tidur mandi mabal',
        'aku tidur',
        'teza makan',
        'teza tidur'
    ]

    pool_dummy_target = [1, 0, 1, 1]
    pool_dummy_target = np_utils.to_categorical(pool_dummy_target, 2)
    print("converting data to vectors..")
    dummy_tokens = []
    for x in pool_dummy:
        val = x.split(" ")
        dummy_tokens.append(val)

    vectors = w2v(dummy_tokens).get_feature()
    docs = np.array([sen2vec(x, vectors = vectors) for x in dummy_tokens])
    docs = docs.reshape(docs.shape[0], -1, 1)
    print(docs.shape)
    model = nn.keras_model(docs, pool_dummy_target, nb_classes=2, hidden_units = 50)
    model.fit(docs, pool_dummy_target, batch_size = 10, epochs=1, verbose=1)
    import pickle as pkl
    with open("model.abal", 'wb') as file:
        pkl.dump(model, file)
    inp = np.array([docs[0]])
    print(model.predict_classes(inp, batch_size = 1, verbose = 1))