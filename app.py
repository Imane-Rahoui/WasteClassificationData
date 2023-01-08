from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf

from keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/best_model.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img1 = tf.keras.utils.load_img(img_path, target_size=(155, 155))
    img1 = tf.keras.utils.img_to_array(img1)
    img1 = np.expand_dims(img1, axis=0)

    result = model.predict(img1 / 255)

    return result


@app.route('/', methods=['GET'])
def Result():
    # Main page
    return render_template('result.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'DIR', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print("-------------- " + format(preds[0][1] * 100, '.2f'))
        print("-------------- " + format(preds[0][0] * 100, '.2f'))

        # Process your result for human
        if preds[0][0] < preds[0][1]:  #
            return format(preds[0][1] * 100, '.1f') + "% Recyclable"
        else:
            return format(preds[0][0] * 100, '.1f') + "% Organic"

    return None


if __name__ == '__main__':
    app.run(debug=True)

