# from classifier import nn
from classifier.driver import driver
# from utils.dataset import Data
import os


from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)

clsfr = driver()
# load classifier model
clsfr.load_word_model('models/sen2vec.mdl')
# load classifier model
clsfr.load_model('models/beta-3.mdl')
clsfr.model._make_predict_function()



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
        print("input: ",txt)
        print("type: ",type(txt))
        pred = clsfr.predict(txt)
        if pred == '':
            check = False
        elif pred == 0:
            pred = "kebencian"
        elif pred == 1:
            pred = "ofensif"
        elif pred == 2:
            pred = "netral"
        print(txt, '--->', pred)
    return render_template("classify.html",txt = txt, pred = pred, check = check)

if __name__ == '__main__':

    app.run()
    # print(clssfr.predict('i i i i'))        