from utils.feature_extraction import WordEmbed as w2v
# from classifier import nn
# import numpy as np
# from keras.utils import np_utils

if __name__ == '__main__':
    pool_dummy = [
        'aku makan tidur mandi mabal',
        'aku tidur',
        'teza makan',
        'teza tidur'
    ]

    # pool_dummy_target = [1, 0, 1, 1]
    # pool_dummy_target = np_utils.to_categorical(pool_dummy_target, 2)
    # print("converting data to vectors..")
    # dummy_tokens = []
    # for x in pool_dummy:
    #     val = x.split(" ")
    #     dummy_tokens.append(val)

    # vectors = w2v(dummy_tokens).get_feature()

    # docs = np.array([sen2vec(x, vectors = vectors) for x in dummy_tokens])
    # docs = docs.reshape(docs.shape[0], -1, 1)
    
    # print(docs.shape)
    # model = nn.keras_model(docs, pool_dummy_target, nb_classes=2, hidden_units = 50)
    # model.fit(docs, pool_dummy_target, batch_size = 10, epochs=1, verbose=1)
    # import pickle as pkl
    # with open("model.abal", 'wb') as file:
    #     pkl.dump(model, file)
    # inp = np.array([docs[0]])
    # print(model.predict_classes(inp, batch_size = 1, verbose = 1))

    import os
    # Load Google's pre-trained Word2Vec model.
    path = os.path.join('utils', 'w2v_models', 'GoogleNews-vectors-negative300.bin')
    model =  w2v()
    model.load_vectors(path)
    print(model.model['i'])
    print(model.model.most_similiar(positive = "i"))