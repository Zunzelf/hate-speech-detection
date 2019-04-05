from utils.feature_extraction import WordEmbed as w2v
from classifier import nn
import numpy as np
from utils.dataset import Data
from keras.utils import np_utils
import os
from utils import preprocess as prepro
import pickle as pkl

class driver():
    def load_model(self, path):
        with open(path, 'rb') as file:
            self.model = pkl.load(file)

    def load_word_model(self, path):
        with open(path, 'rb') as file:
            self.word_model = pkl.load(file)

    def predict(self, inp):
        tesx = inp
        tesx = prepro.clean(tesx)[:20]
        test = w2v().sen2vec_2(tesx, self.word_model)
        res = self.model.predict_classes(test)
        return res
    
    def train_model(self, x, y):
        pass
        
if __name__ == '__main__':
    dats = Data()
    data_root = os.path.join('utils')
    data_path = os.path.join(data_root, 'token-checked.bin')

    with open(data_path, 'rb') as file:
        datas = pkl.load(file)
    dats.load_data('utils/labeled_data.csv')
    word_model = w2v().load_model(path = 'utils/sen2vec.mdl')
    pretrained_weights = word_model.wv.syn0
    
    # vocab_size, emdedding_size = pretrained_weights.shape
    # targets = dats.y

    # num_target = len(set(targets))
    # targets = np_utils.to_categorical(targets, num_classes = num_target)

    # def idx2word(idx):
    #     return word_model.wv.index2word[idx]
    
    # datas = [x[:20] for x in datas]
    # train_x = np.zeros([len(datas), 20], dtype=np.int32)
    # train_y = np.zeros([len(datas), num_target], dtype=np.int32)
    # for i, datas in enumerate(datas):
    #     for t, word in enumerate(datas[:-1]):
    #         train_x[i, t] = word2idx(word)
    #     train_y[i, 0] = targets[i, 0]
    #     train_y[i, 1] = targets[i, 1]
    #     train_y[i, 2] = targets[i, 2]
    # print('train_x shape:', train_x.shape)
    # print('train_y shape:', train_y.shape)


    # model = nn.BiLSTM().create_model(vocab_size, 20, hidden_units = 50)
    # model.fit(train_x, train_y, batch_size = 64, epochs=100, verbose=1)
    # import pickle as pkl
    # with open("beta-3.mdl", 'wb') as file:
    #     pkl.dump(model, file)
    # # inp = np.array([docs[0]])
    # # print(model.predict_classes(inp, batch_size = 1, verbose = 1))

    from tqdm import tqdm
    clsfr = driver()
    # load classifier model
    clsfr.load_model('beta-3.mdl')
    # load classifier model
    clsfr.load_word_model('utils/sen2vec.mdl')

    cnt = 0
    for x,i in tqdm(enumerate(dats.x[:20])):
        pnt = 221
        txt = dats.x[pnt]
        # to predict text ---> text, not list of strings
        pred = clsfr.predict(txt)
        # just for checking accuracy
        orig = dats.y[pnt]
        if pred[0] == orig:
            cnt += 1
    print('>>>>>', cnt)
        

