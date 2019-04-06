import os, sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
try:
    from utils.feature_extraction import WordEmbed
    from utils.feature_extraction import word2idx, idx2word    
    from utils.dataset import Data, tokenize_doc, tokenize_sen
    from classifier.nn import BiLSTM
    from utils.dataset import Data
    from keras.utils import np_utils
except ModuleNotFoundError:
    sys.path.append('..')
    from utils.feature_extraction import WordEmbed
    from utils.feature_extraction import word2idx, idx2word    
    from utils.dataset import Data, tokenize_doc, tokenize_sen
    from classifier.nn import BiLSTM
    from utils.dataset import Data
    from keras.utils import np_utils

from utils import preprocess as prepro
import pickle as pkl
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split as tts
# sys.stderr = stderr
dats = Data()

class driver():
    def load_model(self, path):
        print("loading model: ")
        with open(path, 'rb') as file:
            self.model = pkl.load(file)
        print("model loaded")

    def load_word_model(self, path):
        print("loading model: ")
        with open(path, 'rb') as file:
            self.word_model = pkl.load(file)
        print("model loaded")

    def predict(self, inp):
        tesx = inp
        tesx = prepro.clean(tesx)[:20]
        test = WordEmbed().sen2vec_2(tesx, self.word_model)
        res = self.model.predict_classes(test)
        return res[0]

    def train_model(self, X, Y, X_test = None, Y_test = None, save_path = 'model_classifier.mdl', epochs = 100, model):        
        pretrained_weights = self.word_model.wv.syn0
        datas = X
        targets = Y

        vocab_size, emdedding_size = pretrained_weights.shape

        num_target = len(set(targets))
        targets = np_utils.to_categorical(targets, num_classes = num_target)
        test_targets = np_utils.to_categorical(Y_test, num_classes = num_target)
        datas = [x[:20] for x in datas]
        test_data = [x[:20] for x in X_test]
        train_x = np.zeros([len(datas), 20], dtype=np.int32)
        train_y = np.zeros([len(datas), num_target], dtype=np.int32)
        test_x = np.zeros([len(X_test), 20], dtype=np.int32)
        test_y = np.zeros([len(Y_test), num_target], dtype=np.int32)
        
        for i, data in enumerate(datas):
            for t, word in enumerate(data[:-1]):
                train_x[i, t] = word2idx(word, self.word_model)
            for x in range(num_target):
                train_y[i, x] = targets[i, x]     

        for i, data in enumerate(test_data):
            for t, word in enumerate(data[:-1]):
                test_x[i, t] = word2idx(word, self.word_model)
            for x in range(num_target):
                test_y[i, x] = test_targets[i, x]

        self.model = BiLSTM.create_model(vocab_size, 20, hidden_units = 50)
        self.model.fit(train_x, train_y, batch_size = 64, epochs = epochs, verbose=1, validation_data = [test_x, test_y])

        with open(save_path, 'wb') as file:
            pkl.dump(self.model, file)
    
    def train_word_model(self, sentences, save_path = 'model_word_vector.mdl', ndim = 100, window_size = 10, min_count = 2):
        # the format for input -> [['word-1-sentence-1',..,'word-n-sentence-1'],..,['word-1-sentence-n',..,'word-n-sentence-n']]
        datas = sentences
        datas.extend([['$$'], ['$$']]) # 'padding' to handle <unknown> vocab
        w2v = WordEmbed()
        self.word_model = w2v.create_model(datas, size = ndim, window = window_size, min_count = min_count)
        w2v.save_model(self.word_model, path = save_path)

    def clean_dataset(self, dat, save_path = 'token-checked-2.bin', spell_check = True):
        print("Preparing Parallelism...", end = '')
        pool = mp.Pool(int(mp.cpu_count()))
        dats = dat.x
        print("complete! cpu ready : %i cpus" % int(mp.cpu_count()))

        print("tokenize docs...")
        tkn = list(tqdm(pool.imap(tokenize_sen, dats), total = len(dats)))
        pool.close
        print("tokenize docs...complete!")
        
        if spell_check:
            print("spell checking tokens...")
            pool = mp.Pool(int(mp.cpu_count()))    
            res = list(tqdm(pool.imap(prepro.spell_check, tkn), total = len(tkn)))
            pool.close
            print("spell checking tokens...complete!")
        elif not spell_check:
            res = tkn

        with open(save_path, 'wb') as file:
            print('saving tokens...')
            pkl.dump(res, file)
        
        return res

    def generate_models(self, data, save_path, w2v = None, test = 0.3, spell_check = True, evaluate = True, epochs = 100, model = 'bi'):
        if type(data) == str:
            dat = Data()
            print("loading file...")
            dat.load_data(data)
            print("loading file...complete!")
            cleaned_data = self.clean_dataset(dat, spell_check = spell_check)
            targets = dat.y
        elif type(data) != str:
            cleaned_data = data[0]
            targets = data[1]
        
        train_x, test_x, train_y, test_y = tts(cleaned_data, targets, test_size = test)
        
        if w2v == None:
            self.train_word_model(cleaned_data, save_path = 'generated_word_model.mdl')
            print('Processed : %i vocabs'% len(self.word_model.wv.vocab))
        elif type(w2v) == str:
            self.load_word_model(w2v)

        if type(data) == str:
            # freed some memory
            del cleaned_data
            del dat

        self.train_model(train_x, train_y, test_x, test_y, save_path = save_path, epochs = epochs, model = model)
        if evaluate:
            self.evaluate(test_x, test_y)

    def evaluate(self, test_x, test_y):
            preds = []
            orig = []
            txts = []
            for x in tqdm(range(len(test_x))):
                txts = " ".join(test_x[x])
                preds.append(self.predict(txts))
                # just for checking accuracy
                orig.append(test_y[x])       
            score = f1_score(orig, preds, average = 'micro')   
            print(score)
        
if __name__ == "__main__":
    data_path = os.path.join("..", 'models', 'token-checked-2.bin')
    w2v_path = os.path.join("..", 'models', 'glove-twitter-100.txt')
    with open(data_path, 'rb') as file:
        data = pkl.load(file)
    # w2v = WordEmbed()
    # model = w2v.load_vectors(w2v_path, False)
    # # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # # into our TensorFlow and Keras models
    # embedding_matrix = np.zeros((len(model.wv.vocab), 100))
    # for i in tqdm(range(len(model.wv.vocab))):
    #     embedding_vector = model.wv[model.wv.index2word[i]]
    #     if embedding_vector is not None:
    #         embedding_matrix[i] = embedding_vector
    # print(embedding_matrix)
    drv = driver()
    drv.load_word_model(os.path.join('..', 'models', 'sen2vec.mdl'))
    trgt = Data()
    trgt.load_data(os.path.join('..', 'utils', 'labeled_data.csv'))
    trgt = trgt.y
    drv.generate_models([data, trgt], 'generated_model.mdl', os.path.join('..', 'models', 'sen2vec.mdl'))