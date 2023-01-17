from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf

from keras.models import load_model
import warnings

#All warnings are ignored
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'models/best_model.hdf5'

# Load model
model = load_model(MODEL_PATH)


def predictOR(img_path, model):

    #print("f1")
    # recuperer l'image et ajuster
    image = tf.keras.utils.load_img(img_path, target_size=(155, 155))

    #print("avant : ")
    #print(image)
    # convert image to numpy array
    image = tf.keras.utils.img_to_array(image)

    # ajouter un nouvel axe au tableau.
    image = np.expand_dims(image, axis=0)
    print("apres : ")
    #print(image)
    result = model.predict(image / 255)

    return result


@app.route('/', methods=['GET'])
def Result():
    print("f2")
    return render_template('result.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("f3")
    if request.method == 'POST':
        # recuperer l'image
        f = request.files['file']

        print("fichier : "+__file__)
        # Recuperer le rep du projet
        basepath = os.path.dirname(__file__)
        # Enregister f DIR
        file_path = os.path.join(
            basepath, 'DIR', secure_filename(f.filename))
        f.save(file_path)
        print("allo")
        preds = predictOR(file_path, model)
        print("-------------- " + format(preds[0][1] * 100, '.2f'))
        print("-------------- " + format(preds[0][0] * 100, '.2f'))

        # first one organic
        # second one recyclable
        # ----------------------------- QUEEEESTTTT
        print(preds)
        if preds[0][0] < preds[0][1]:
            return format(preds[0][1] * 100, '.1f') + "% Recyclable"
        else:
            return format(preds[0][0] * 100, '.1f') + "% Organic"

    return None


if __name__ == '__main__':
    app.run(debug=True)

