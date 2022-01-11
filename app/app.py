import flask
import os
from flask import Flask, request
from pyoxynet import *
import numpy as np
import pandas as pd

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))

@app.route('/search', methods=['GET', 'POST'])
def search():
    args = request.args
    model = args.get('model')

    results = {}
    results['model'] = model

    return flask.jsonify(results)

@app.route('/read_json', methods=['GET', 'POST'])
def read_json():

    args = request.args
    n_inputs = args.get('n_inputs')

    # loading the model
    if n_inputs == '5':
        tfl_model = load_tf_model(n_inputs=int(n_inputs),
                                  past_points=40)
    else:
        tfl_model = load_tf_model()

    input_details = tfl_model.get_input_details()
    output_details = tfl_model.get_output_details()

    request_data = request.get_json(force=True)
    df = pd.DataFrame.from_dict(request_data)

    if n_inputs == '7':
        X = df[['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I']]
    if n_inputs == '5':
        print(df.columns)
        X = df[['VO2_I', 'VE_I', 'PetO2_I', 'RF_I', 'VEVO2_I']]

    XN = normalize(X)

    time_series_len = input_details[0]['shape'][1]
    p_1 = []
    p_2 = []
    p_3 = []

    for i in np.arange(len(XN) - time_series_len):
        XN_array = np.asarray(XN[i:(i + time_series_len)])
        input_data = np.reshape(XN_array, input_details[0]['shape'])
        input_data = input_data.astype(np.float32)

        tfl_model.allocate_tensors()
        tfl_model.set_tensor(input_details[0]['index'], input_data)
        tfl_model.invoke()
        output_data = tfl_model.get_tensor(output_details[0]['index'])
        p_1.append(output_data[0][0])
        p_2.append(output_data[0][1])
        p_3.append(output_data[0][2])

    results = {}
    results['p_1'] = str(p_1)
    results['p_2'] = str(p_2)
    results['p_3'] = str(p_3)

    return flask.jsonify(results)

@app.route("/")
def hello_world():
    return "Hello World from the updated version"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)