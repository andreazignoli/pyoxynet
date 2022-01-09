import flask
import os
from flask import Flask, request
from werkzeug.utils import secure_filename

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))

@app.route("/")
def hello_world():
    return "Hello World"

@app.route("/print_filename", methods=['POST','PUT'])
def print_filename():
    file = request.files['file']
    filename=secure_filename(file.filename)   
    return filename

@app.route('/predict', methods=['POST'])
def predict():

    features = flask.request.get_json(force=True)['features']
    # prediction = model.predict([features])[0,0]
    response = {'prediction': 0}

    return flask.jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)