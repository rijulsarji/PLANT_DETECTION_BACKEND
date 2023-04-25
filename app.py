from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import keras.utils as image
from flask_cors import CORS
import os

model = load_model('model.h5')

classnames = ['moneyPlant', 'peepalTree', 'spiderPlant']


def pridict(j):
    i = image.load_img(j, target_size=(150,150, 3))
    i = image.img_to_array(i)
    i = i.reshape(1, 150,150,3)
    ans = model.predict(i)
    ans_index = tf.math.argmax(ans[0])
    if(ans[0][ans_index] > 0.5):
        return classnames[ans_index]
    else:
        return "This does not exist in our database"



app = Flask(__name__)
CORS(app)
@app.route("/")
def home():
    return "Hello, World!"

@app.route("/about")
def about():
    return {"a": 1}

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['file']

        img_path = os.getcwd() + "//static//" + img.filename	
        img.save(img_path)

        a = pridict(img_path)
        ans = {"a": a}
        print(ans)
        os.remove(img_path)
    return ans


if __name__ == "__main__":
    app.run()