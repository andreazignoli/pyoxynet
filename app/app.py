import flask
import os
from flask import Flask, request
from werkzeug.utils import secure_filename
import requests
from pyoxynet import *
import numpy as np
import pandas as pd

tfl_model = load_tf_model()
input_details = tfl_model.get_input_details()
output_details = tfl_model.get_output_details()

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))

@app.route('/read_json', methods=['POST'])
def read_json():
    request_data = request.get_json(force=True)
    df = pd.DataFrame.from_dict(request_data)
    XN = normalize(df)

    time_series_len = input_details[0]['shape'][1]
    p_md = []
    p_hv = []
    p_sv = []

    for i in np.arange(len(XN) - time_series_len):
        XN_array = np.asarray(XN[i:(i + time_series_len)])
        input_data = np.reshape(XN_array, input_details[0]['shape'])
        input_data = input_data.astype(np.float32)

        tfl_model.allocate_tensors()
        tfl_model.set_tensor(input_details[0]['index'], input_data)
        tfl_model.invoke()
        output_data = tfl_model.get_tensor(output_details[0]['index'])
        p_md.append(output_data[0][2])
        p_sv.append(output_data[0][1])
        p_hv.append(output_data[0][0])

    results = {}
    results['p_md'] = str(p_md)
    results['p_hv'] = str(p_hv)
    results['p_sv'] = str(p_sv)

    return flask.jsonify(results)

@app.route("/")
def hello_world():
    return "Hello World"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)