import os, csv
try :
    from utils import preprocess
    from utils.feature_extraction import WordEmbed as w2v
except ModuleNotFoundError:
    import preprocess
    from feature_extraction import WordEmbed as w2v
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
# import nltk
# nltk.download('punkt')
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Data():
    def __init__(self):
        self.y = []      #   for data with class. Training, [class]
        self.x = []      #   for training feature [tweets]
        
    def load_data(self, path):
        with open(path, mode ='r', encoding="utf-8") as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            next(csv_file)
            for line in tqdm(data):
                self.x.append(line[2])
                self.y.append(int(line[1]))
            csv_file.close()

# Support Vector
class Support_Vector:
    def __init__(self):
        self.acc = 0
        self.model = None
        self.classifier = None
        self.pred = None
        self.vectorized = CountVectorizer()
        self.tf_idf = TfidfTransformer()
        self.data = None
        self.filenameModel = 'SupportVector_model.sav'
        self.filenameVector = 'SupportVector_vector.sav'
        self.filenameTFIDF = 'SupportVector_tfidf.sav'

    def load_data(self):
        print("load model from file .........................")
        self.classifier = pickle.load(open('Model/'+self.filenameModel, 'rb'))
        self.vectorized = pickle.load(open('Model/'+self.filenameVector, 'rb'))
        self.tf_idf = pickle.load(open('Model/'+self.filenameTFIDF, 'rb'))
    
    def train(self,data):
        self.data = data

        X_train_counts = self.vectorized.fit_transform(self.data.X_train)

        # feature extraction       
        X_train_tfidf = self.tf_idf.fit_transform(X_train_counts)

        # train
        self.classifier = SVC(kernel="linear")  
        self.classifier.fit(X_train_tfidf, self.data.y_train)

        # save to pickle
        print("Save model to file ...........................")
        pickle.dump(self.classifier, open('Model/'+self.filenameModel, 'wb'))
        pickle.dump(self.vectorized, open('Model/'+self.filenameVector, 'wb'))
        pickle.dump(self.tf_idf, open('Model/'+self.filenameTFIDF, 'wb'))

        # testing
        X_test = self.vectorized.transform(self.data.X_test)
        X_test_tfidf = self.tf_idf.transform(X_test)
        self.y_pred = self.classifier.predict(X_test_tfidf) 
        self.acc = accuracy_score(self.data.y_test, self.y_pred)
        print("SVM accuration : ",self.acc)  

    def classify(self,docs):
        vectorized_docs = self.vectorized.transform([docs])
        tfidf_docs = self.tf_idf.transform(vectorized_docs)
        pred = self.classifier.predict(tfidf_docs)
        print("Classification result : ",docs)
        if (pred[0] == 0):
            print('Hate Speech')
        elif (pred[0] == 1):
            print('Offensive Language')
        else:
            print('Neither')
        
# Ekstraksi fitur TFIDF
class Data_Processing:
    def __init__(self,tweet,kelas):
        self.tweet = tweet
        self.kelas = kelas
        self.X_train = self.X_test = self.y_pred = self.y_train = self.y_test = None

    def set_all(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tweet, self.kelas, test_size=0.2)

    def cek_algorithm_result(self,y_test,y_pred):
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))

def tokenize_doc(docs, prepro = False):
    tokens = []
    for line in docs:
        if prepro:
            tmp = preprocess.tweet_prepro(line)
        elif not prepro:
            tmp = line.split(" ")
        tokens.append(tmp)
    return tokens

def tokenize_sen(doc): # for parallel process
    tmp = preprocess.tweet_prepro(doc)
    return tmp

if __name__ == "__main__":
    dat = Data()

    print("loading file...")
    dat.load_data('labeled_data.csv')
    print("loading file...complete!")

    # print("Preparing Parallelism...", end = '')
    # pool = mp.Pool(int(mp.cpu_count()))
    # dats = dat.x
    # print("complete! cpu ready : %i cpus" % int(mp.cpu_count()))

    # print("tokenize docs...")
    # tkn = list(tqdm(pool.imap(tokenize_sen, dats), total = len(dats)))
    # pool.close
    # print("tokenize docs...complete!")

    # print("spell checking tokens...")
    # pool = mp.Pool(int(mp.cpu_count()))    
    # res = list(tqdm(pool.imap(preprocess.spell_check, tkn), total = len(tkn)))
    # pool.close
    # print("spell checking tokens...complete!")


    # # Word2Vec
    # word2v = w2v()
    # model =  word2v.load_model()
    # vectorized = []
    # for dat in tqdm(dats):
    #     vectorized.append(word2v.sen2vec(dats[0], vectors = model))

    # vectorized = np.array(vectorized)
    # vectorized = vectorized.reshape(vectorized.shape[0], -1, 1)  

    # print(vectorized.shape)
    # with open('token-vectorized.bin', 'wb') as file:
    #     pickle.dump(vectorized, file)


    # TF-IDF with Support Vector
    sv = Support_Vector()
    
    if not os.path.exists('./Model/'):
        os.makedirs('./Model/')

    if (os.path.exists('./Model/'+sv.filenameModel)):
        sv.load_data()
        words = input("Insert sentence to classify : ")
        token = tokenize_sen(words)
        checked = ' '.join(preprocess.spell_check(token))
        sv.classify(checked)
    else:
        # if not then train

        print("Preparing Parallelism...", end = '')
        pool = mp.Pool(int(mp.cpu_count()))
        dats = dat.x
        print("complete! cpu ready : %i cpus" % int(mp.cpu_count()))

        print("tokenize docs...")
        token = list(tqdm(pool.imap(tokenize_sen, dats), total = len(dats)))
        pool.close
        print("tokenize docs...complete!")

        print("spell checking tokens...")
        spellChecked = []
        for w in token:
            spellChecked.append(' '.join(preprocess.spell_check(w)))
        print("spell checking tokens...complete!")

        tweets = spellChecked
        label = dat.y
        data = Data_Processing(tweets, label)

        # set all feature
        data.set_all()
            
        # classify (cl stand for classifier)
        sv.train(data)
    
  