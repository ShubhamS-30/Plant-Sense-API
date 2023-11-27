from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from flask import abort

# Initialize the Flask application
app = Flask(__name__)

# model path
model_path = 'model.tflite'


# labels path
labels_path = 'labels.txt'

# number of classes with highest confidence
pred_num = 1

diseases = {
    "Tomato" : ["Tomato Septoria Leaf Spot", "Tomato Bacterial Spot", "Tomato Blight", "Tomato Spider Mite", "Tomato Leaf Mold", "Tomato Yellow Leaf Curl Virus"],
    "Cabbage":["Cabbage Black Rot"],
    "Maize":["Maize Ravi Corn Rust", "Maize Grey Leaf Spot", "Maize Lethal Necrosis"],
    "Soy":["Soy Frogeye Leaf Spot", "Soy Downy Mildew"],
}

# returns labels as a map
def get_labels():
    with open(labels_path, 'r') as f:
        labels = list(map(str.strip, f.readlines()))
    return labels

# returns predictions of the uploaded image
# function to inference the model
def predictions(img):

    # loading the model
    interpreter = tf.lite.Interpreter(model_path)

    # Obtain input size of image from input details of the model
    input_details = interpreter.get_input_details()

    # Obtain input size of image from output details of the model
    output_details = interpreter.get_output_details()

    # resizing the image to model's input shape
    img1 = cv2.resize(img, (300, 300))

    # converting to tensor
    img_to_tensor = tf.convert_to_tensor(img1)
    input_data = np.array([img_to_tensor], dtype=np.uint8)

    # allocate tensors to use the set_tensor() method to feed the processed_image
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # model Inference
    interpreter.invoke()

    # results
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # selecting top pred_num values
    pred_num_indices = np.argsort(output_data[0])[::-1][0:pred_num]

    # getting labels
    labs = get_labels()

    # stores confidence
    scores = []

    # stores labels
    lbls = []

    for i in range(pred_num):

        # converting score into percentage
        score = output_data[0][pred_num_indices[i]]/255

        # getting corresponding label
        lbl = labs[pred_num_indices[i]]

        # adding confidence to scores
        scores.append(str(score))

        # adding to label to lbls
        lbls.append(lbl)

    return lbls, scores

# route http posts to this method

# send and recieve image classification result to user
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # processing
    labels, scores = predictions(img)
    # build a response dict to send back to client
    res = {
        "message": "Image recieved!!",
        "image shape": [img.shape[0], img.shape[1]],
        "labels": labels,
        "score": scores,
    }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(res)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# function to return info about a disease
@app.route('/api/info/<name>', methods=['GET'])
def information(name):

    path = "./static/Diseases/" + name + ".txt"

    file1 = open(path, "r+")

    info = file1.read()

    file1.close()

    res = {
        "message": "Request recieved!!",
        "Information": info,
    }

    response_pickled = jsonpickle.encode(res)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# function to send the names of species in database to user.
@app.route('/api/dict/species', methods=['GET'])
def get_species():
    
    res = {
        "message": "Request recieved!!",
        "Species": list(diseases.keys()),
    }
    response_pickled = jsonpickle.encode(res)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# function that returns diseases in the database related to particular specie
@app.route('/api/dict/species/<species>', methods=['GET'])
def species_diseases(species):

    if species in diseases.keys():

        diseases_list = diseases[species]

        res = {
            "message": "Request recieved!!",
            "Diseases": diseases_list,
        }

        response_pickled = jsonpickle.encode(res)

        return Response(response=response_pickled, status=200, mimetype="application/json")
    else:
        abort(404)


# start flask app
app.run(host="0.0.0.0", port=5000)


# command to run - > python -m flask --app app run
# or
# python -m flask run
# or
# python app.py
