# from classifier import nn
from classifier.driver import driver
from utils.feature_extraction import WordEmbed
# from utils.dataset import Data
import os


from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)
drv = driver()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["GET","POST"])
def classify():
    pred = ''
    txt = ""
    check = True
    if request.method == "POST":
        txt = request.form["inference"]
        print("input          :",txt)
        print("type           :",type(txt))
        pred = drv.predict(txt)
        if pred == '':
            check = False
        elif pred == 0:
            pred = "kebencian"
        elif pred == 1:
            pred = "ofensif"
        elif pred == 2:
            pred = "netral"
        print('predicted      :', pred)
    return render_template("classify.html",txt = txt, pred = pred, check = check)

if __name__ == '__main__':
    # load classifier model
    drv.load_model(os.path.join('models', 'generated_model_2.mdl'))
    drv.model._make_predict_function()
    w2v_path = os.path.join('models', 'glove-twitter-100.txt')

    w2v = WordEmbed() 
    drv.word_model = w2v.load_vectors(w2v_path, False)

    app.run()      