from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import static.models.cityscapes as cityscapes
#import static.models.unet_xception as mux
import static.models.deeplab_v3plus as mdv3
import numpy as np
import base64
import os
import io
from PIL import Image

import json

app = Flask(__name__)

STATIC_FOLDER = 'static/'
MODEL_FOLDER = os.path.join(STATIC_FOLDER, 'models/')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER,'uploads/')
RESULT_FOLDER = os.path.join(STATIC_FOLDER,'results/')

@app.before_first_request
def load__model():
    """
    Load model
    :return: model (global variable)
    """
    print('[INFO] Model Loading ........')
    global model, resize
    
    #backbone="mobilenetv2" #"xception"
    model_name = "deeplab_v3plus_512_augment"

    resize = int(model_name.replace("_augment", "").split("_")[-1])
    model = mdv3.get_model(
                    weights="cityscapes",
                    input_tensor=None,
                    input_shape=(resize, resize, 3),
                    classes=8,
                    backbone="mobilenetv2", #"xception",
                    OS=36,
                    alpha=1.0,
                    activation="softmax",
                    model_name=model_name,
                )
    print(model.name)

    model.load_weights(MODEL_FOLDER + model_name + ".h5")


def prediction(fullpath_image):
    #input_img = tf.keras.preprocessing.image.load_img(fullpath_image, target_size=(resize, resize))
    resize = 512
    input_img = Image.open(fullpath_image).resize((resize, resize))

    # Prediction:
    result = Image.fromarray(
        cityscapes.cityscapes_category_ids_to_category_colors(
            np.squeeze(
                np.argmax(
                    model.predict(np.expand_dims(input_img, 0)),
                    axis=-1,
                )
            )
        )
    )
    return result
    

@app.route('/api/predict/', methods=['POST'])
def predict_segmentation():

    if request.method == 'POST':
        image_file = request.files.get('image')
        
        # Do something with the image file
        image_data = image_file.read()
        image_resized = Image.open(io.BytesIO(image_data)).resize((resize, resize))
        fullname = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_resized.save(fullname)
        
        result = prediction(fullname)
        result.save(os.path.join(RESULT_FOLDER,"result_img.png"))
        fullpath_res = os.path.join(RESULT_FOLDER,"result_img.png")
        with open(fullpath_res, "rb") as imag_file:
            img_enc = base64.b64encode(imag_file.read()).decode()
    
        return jsonify({"image":img_enc})

        
# Home Page
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        image_input_file = request.files['image']

        # Do something with the image file
        image_data = image_input_file.read()
        image_resized = Image.open(io.BytesIO(image_data)).resize((resize, resize))
        fullpath_image_input  = os.path.join(UPLOAD_FOLDER, image_input_file.filename)
        image_resized.save(fullpath_image_input )

        # Prediction
        result = prediction(fullpath_image_input)
        result_fname = "result_"+image_input_file.filename
        fullpath_res = os.path.join(UPLOAD_FOLDER, result_fname)
        result.save(fullpath_res)

        return render_template('index.html', predict=True, image_file_name=image_input_file.filename, 
                               result_fname=result_fname)
    else:
        return render_template('index.html', predict=False)

@app.route('/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/<result_filename>')
def result_file(result_filename):
    return send_from_directory(UPLOAD_FOLDER, result_filename)

"""
@app.route('/<filename>')
def ground_truth_seg_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
""" 

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)