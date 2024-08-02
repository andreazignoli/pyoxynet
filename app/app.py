import flask
import os
from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from flasgger import Swagger, swag_from
import pyoxynet
import numpy as np
import pandas as pd
from pandas import read_csv
import plotly.graph_objs as go
import plotly
from faker import Faker
import pandas as pd
import tensorflow as tf

app = flask.Flask(__name__)
Swagger(app)
port = int(os.getenv("PORT", 9098))
app.secret_key = "super secret key"

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to store pre-loaded TFLite models
models = {
    'model_1': tf.lite.Interpreter('tf_lite_models/tfl_model.tflite')
}

def CPET_var_plot_vs_CO2(df, var_list=[]):
    import json
    import plotly.express as px

    labels_dict = {}

    for lab_ in var_list:
        labels_dict[lab_] = lab_.replace('_', ' ').replace('I', '')

    fig = px.scatter(df.iloc[np.arange(0, len(df))],
                     x="VCO2_I",
                     y=var_list, color_discrete_sequence=['white', 'gray'])
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))

    fig.update_layout(
        xaxis=dict(
            title='VCO2',
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title='',
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(82, 82, 82)',
            ),
        ),
        autosize=True,
        showlegend=True,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.for_each_trace(lambda t: t.update(name=labels_dict[t.name],
                                          legendgroup=labels_dict[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, labels_dict[t.name])
                                          ))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def CPET_var_plot_vs_O2(df, var_list=[], VT=[0, 0, 0, 0]):
    import json
    import plotly.express as px

    VT1 = VT[0]
    VT2 = VT[1]
    VT1_oxynet = VT[2]
    VT2_oxynet = VT[3]

    labels_dict = {}

    for lab_ in var_list:
        labels_dict[lab_] = lab_.replace('_', ' ').replace('I', '')

    fig = px.scatter(df.iloc[np.arange(0, len(df))], x="VO2_I", y=var_list,
                     color_discrete_sequence=['white', 'gray'])
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black'), color='#51a1ff', opacity=0.7))

    if VT1 > 0:
        fig.add_vline(x=VT1, line_width=3, line_dash="dash", line_color="dodgerblue", annotation_text="VT1")
    if VT2 > 0:
        fig.add_vline(x=VT2, line_width=3, line_dash="dash", line_color="red", annotation_text="VT2")

    fig.add_vline(x=VT1_oxynet, line_width=1, line_color="dodgerblue")
    fig.add_vline(x=VT2_oxynet, line_width=1, line_color="red")

    fig.update_layout(
        xaxis=dict(
            title='VO2',
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=14,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title='',
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            tickfont=dict(
                family='Arial',
                size=14,
                color='rgb(82, 82, 82)',
            ),
        ),
        autosize=True,
        showlegend=True,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.for_each_trace(lambda t: t.update(name=labels_dict[t.name],
                                          legendgroup=labels_dict[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, labels_dict[t.name])
                                          ))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def CPET_var_plot(df, var_list=[], VT=[300, 400]):
    import json
    import plotly.express as px

    VT1 = VT[0]
    VT2 = VT[1]
    VT1_oxynet = VT[2]
    VT2_oxynet = VT[3]

    labels_dict = {}

    for lab_ in var_list:
        labels_dict[lab_] = lab_.replace('_', ' ').replace('I', '')

    fig = px.line(df.iloc[np.arange(0, len(df))], x="time", y=var_list,
                  color_discrete_sequence=['white', 'gray', 'black'])
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
    fig.add_vline(x=VT1, line_width=3, line_dash="dash", line_color="dodgerblue", annotation_text="VT1")
    fig.add_vline(x=VT2, line_width=3, line_dash="dash", line_color="red", annotation_text="VT2")
    fig.add_vline(x=VT1_oxynet, line_width=1, line_color="dodgerblue")
    fig.add_vline(x=VT2_oxynet, line_width=1, line_color="red")

    fig.update_layout(
        xaxis=dict(
            title='Time',
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=14,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title='',
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            tickfont=dict(
                family='Arial',
                size=14,
                color='rgb(82, 82, 82)',
            ),
        ),
        autosize=True,
        showlegend=True,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.for_each_trace(lambda t: t.update(name=labels_dict[t.name],
                                          legendgroup=labels_dict[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, labels_dict[t.name])
                                          ))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def test_tf_lite_model(interpreter):
    """Test if the model is running correctly

    Parameters:
        interpreter (loaded tf.lite.Interpreter) : Loaded interpreter TFLite model

    Returns:
        x (array) : Model output example

    """
    import numpy as np

    # Allocate tensors.
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])

def tf_lite_model_inference(tf_lite_model=[], input_df=[], past_points=40, n_inputs=5, inference_stride=1):
    """Runs the pyoxynet inference

    Parameters:
        tf_model (TF model) : Tf lite model
        inference_stride (int) : Stride inference for NN - speed up computation

    Returns:
        x (array) : Model output example

    """

    df = input_df

    model_id = 'model_1'
    tf_lite_model = models.get(model_id)

    # retrieve interpreter details
    input_details = tf_lite_model.get_input_details()
    output_details = tf_lite_model.get_output_details()

    # some adjustments to input df
    # TODO: create dedicated function for this
    df = df.drop_duplicates('time')
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('timestamp')
    df = df.resample('1S').mean()
    df = df.interpolate()
    df['VO2_20s'] = df.VO2_I.rolling(20, win_type='triang', center=True).mean().fillna(method='bfill').fillna(
        method='ffill')
    df = df.reset_index()
    df = df.drop('timestamp', axis=1)

    if 'VCO2VO2_I' not in df.columns:
        df['VCO2VO2_I'] = df['VCO2_I'].values/df['VO2_I'].values
    filter_vars = ['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I']
    XN = df.copy()
    XN['VO2_I'] = (XN['VO2_I'] - XN['VO2_I'].min()) / (
            XN['VO2_I'].max() - XN['VO2_I'].min())
    XN['VCO2_I'] = (XN['VCO2_I'] - XN['VCO2_I'].min()) / (
            XN['VCO2_I'].max() - XN['VCO2_I'].min())
    XN['VE_I'] = (XN['VE_I'] - XN['VE_I'].min()) / (
            XN['VE_I'].max() - XN['VE_I'].min())
    XN['PetO2_I'] = (XN['PetO2_I'] - XN['PetO2_I'].min()) / (
            XN['PetO2_I'].max() - XN['PetO2_I'].min())
    XN['PetCO2_I'] = (XN['PetCO2_I'] - XN['PetCO2_I'].min()) / (
            XN['PetCO2_I'].max() - XN['PetCO2_I'].min())
    XN = XN.filter(filter_vars, axis=1)

    p_1 = []
    p_2 = []
    p_3 = []
    time = []
    VO2 = []
    VCO2 = []
    VE = []
    PetO2 = []
    PetCO2 = []

    for i in np.arange(1, len(XN) - past_points, inference_stride):
        XN_array = np.asarray(XN[i:i+int(past_points)])
        input_data = np.reshape(XN_array, input_details[0]['shape'])
        input_data = input_data.astype(np.float32)
        tf_lite_model.allocate_tensors()
        tf_lite_model.set_tensor(input_details[0]['index'], input_data)
        tf_lite_model.invoke()
        output_data = tf_lite_model.get_tensor(output_details[0]['index'])
        p_1.append(output_data[0][0])
        p_2.append(output_data[0][1])
        p_3.append(output_data[0][2])
        time.append(df.time[i] + past_points)

        # ['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'domain']
        VO2.append(np.mean(XN_array[-1, 0]) * (df['VO2_I'].max() - df['VO2_I'].min()) + df['VO2_I'].min())
        VCO2.append(np.mean(XN_array[-1, 1]) * (df['VCO2_I'].max() - df['VCO2_I'].min()) + df['VCO2_I'].min())
        VE.append(np.mean(XN_array[-1, 2]) * (df['VE_I'].max() - df['VE_I'].min()) + df['VE_I'].min())
        PetO2.append(np.mean(XN_array[-1, 3]) * (df['PetO2_I'].max() - df['PetO2_I'].min()) + df['PetO2_I'].min())
        PetCO2.append(np.mean(XN_array[-1, 4]) * (df['PetCO2_I'].max() - df['PetCO2_I'].min()) + df['PetCO2_I'].min())

    tmp_df = pd.DataFrame()
    tmp_df['time'] = time
    tmp_df['p_md'] = pyoxynet.utilities.optimal_filter(np.asarray(time), np.asarray(p_1), 100)
    tmp_df['p_hv'] = pyoxynet.utilities.optimal_filter(np.asarray(time), np.asarray(p_2), 100)
    tmp_df['p_sv'] = pyoxynet.utilities.optimal_filter(np.asarray(time), np.asarray(p_3), 100)

    # compute the normalised probabilities
    tmp_df['p_md_N'] = np.asarray(p_1) / (np.asarray(p_1) + np.asarray(p_2) + np.asarray(p_3))
    tmp_df['p_hv_N'] = np.asarray(p_2) / (np.asarray(p_1) + np.asarray(p_2) + np.asarray(p_3))
    tmp_df['p_sv_N'] = np.asarray(p_3) / (np.asarray(p_1) + np.asarray(p_2) + np.asarray(p_3))

    tmp_df.loc[tmp_df['p_md_N'] < 0, 'p_md_N'] = 0
    tmp_df.loc[tmp_df['p_hv_N'] < 0, 'p_hv_N'] = 0
    tmp_df.loc[tmp_df['p_sv_N'] < 0, 'p_sv_N'] = 0

    mod_col = tmp_df[['p_md', 'p_hv', 'p_sv']].iloc[:5].mean().idxmax()
    sev_col = tmp_df[['p_md', 'p_hv', 'p_sv']].iloc[-5:].mean().idxmax()
    for labels_ in ['p_md', 'p_hv', 'p_sv']:
        if labels_ not in [mod_col, sev_col]:
            hv_col = labels_

    out_df = pd.DataFrame()
    out_df['time'] = time
    out_df['p_md'] = tmp_df[mod_col]
    out_df['p_hv'] = tmp_df[hv_col]
    out_df['p_sv'] = tmp_df[sev_col]
    out_df['VO2'] = VO2
    out_df['VCO2'] = VCO2
    out_df['VE'] = VE
    out_df['PetO2'] = PetO2
    out_df['PetCO2'] = PetCO2
    out_df['VO2_F'] = pyoxynet.utilities.optimal_filter(np.asarray(time), np.asarray(VO2), 100)

    out_dict = {}
    out_dict['VT1'] = {}
    out_dict['VT2'] = {}
    out_dict['VT1']['time'] = {}
    out_dict['VT2']['time'] = {}

    # FIXME: hard coded
    VT1_index = int(out_df[(out_df['p_hv'] >= out_df['p_md'])].index[0] - int(past_points / inference_stride))
    VT2_index = int(out_df[(out_df['p_sv'] <= out_df['p_hv'])].index[-1] - int(past_points / inference_stride))

    VT1_time = int(out_df.iloc[VT1_index]['time'])
    VT2_time = int(out_df.iloc[VT2_index]['time'])

    out_dict['VT1']['time'] = VT1_time
    out_dict['VT2']['time'] = VT2_time

    out_dict['VT1']['HR'] = df.iloc[VT1_index]['HR_I']
    out_dict['VT2']['HR'] = df.iloc[VT2_index]['HR_I']

    out_dict['VT1']['VE'] = out_df.iloc[VT1_index]['VE']
    out_dict['VT2']['VE'] = out_df.iloc[VT2_index]['VE']

    out_dict['VT1']['VO2'] = out_df.iloc[VT1_index]['VO2_F']
    out_dict['VT2']['VO2'] = out_df.iloc[VT2_index]['VO2_F']

    return out_df, out_dict

@app.route('/search', methods=['GET', 'POST'])
def search():
    args = request.args
    model = args.get('model')

    results = {}
    results['model'] = model

    return flask.jsonify(results)

@app.route('/read_json', methods=['GET', 'POST'])
def read_json():
    # Reads from json in the Oxynet recommended format
    args = request.args

    try:
        request_data = request.get_json(force=True)
        print(isinstance(request_data, dict))
        df = pd.DataFrame.from_dict(request_data)

        model_id = 'model_1'
        tf_lite_model = models.get(model_id)

        df_estimates, dict_estimates = tf_lite_model_inference(tf_lite_model=tf_lite_model,
                                                               input_df=df,
                                                               inference_stride=2)
    except:
        dict_estimates = {}

    return flask.jsonify(dict_estimates)

@app.route('/read_json_ET', methods=['POST', 'GET'])
@swag_from('swagger/read_json_ET.yml')
def read_json_ET():
    # Reads data from JSON in ET formats
    try:
        if request.method == 'POST':
            request_data = request.get_json(force=True)
        elif request.method == 'GET':
            request_data = request.args.get('data')

        if request_data is None:
            return 'No JSON data provided', 400

        df = pyoxynet.utilities.load_exercise_threshold_app_data(data_dict=request_data)

        model_id = 'model_1'
        tf_lite_model = models.get(model_id)

        df_estimates, dict_estimates = tf_lite_model_inference(tf_lite_model=tf_lite_model,
                                                               input_df=df,
                                                               inference_stride=2)
        return flask.jsonify(dict_estimates), 200
    except Exception as e:
        return f'Error processing JSON data: {str(e)}', 500

@app.route('/curl_csv', methods=['POST'])
@swag_from('swagger/curl_csv.yml')
def curl_csv():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    # If the user does not select a file, the browser may submit an empty file without a filename
    if file.filename == '':
        return 'No selected file', 400

    # If the file is valid, process it
    if file:
        # Read and process the contents of the CSV file
        try:
            # Example: Read the CSV file using pandas
            df = pd.read_csv(file)
            t = pyoxynet.Test('idle')
            t.set_data_extension('.csv')
            t.infer_metabolimeter(optional_data=df)
            t.load_file()
            t.create_data_frame()
            t.create_raw_data_frame()

            model_id = 'model_1'
            tf_lite_model = models.get(model_id)

            df_estimates, dict_estimates = tf_lite_model_inference(tf_lite_model=tf_lite_model, input_df=t.data_frame, inference_stride=2)

            return flask.jsonify(dict_estimates), 200
        except Exception as e:
            return f'Error processing file: {str(e)}', 500
    else:
        return 'Invalid file', 400

@app.route('/read_csv_app', methods=['GET', 'POST'])
def read_csv_app():

    # Reads from csv and uses the pyoxynet parser
    args = request.args

    try:

        file = request.files['file']
        filename, file_extension = os.path.splitext(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        file.save(file.filename)

        t = pyoxynet.Test(filename)
        t.set_data_extension(file_extension)

        if file_extension == '.csv' or file_extension == '.CSV':
            try:
                df = read_csv(file.filename)
            except:
                try:
                    df = read_csv(file.filename)
                except:
                    try:
                        df = pd.read_csv(file.filename, delimiter=';')
                    except:
                        pass
            print('Just reading a csv file')
        if file_extension == '.txt':
            df = read_csv(file.filename, sep="\t", header=None, skiprows=3)
            print('Just reading a txt file')
            t.metabolimeter = 'vyiare'
        if file_extension == '.xlsx' or file_extension == '.xls':
            print('Attempting to read an Excel file')
            df = pd.read_excel(file.filename)

        os.remove(file.filename)
        t.infer_metabolimeter(optional_data=df)
        t.load_file()
        t.create_data_frame()
        t.create_raw_data_frame()

        model_id = 'model_1'
        tf_lite_model = models.get(model_id)

        # df_estimates, dict_estimates = pyoxynet.utilities.test_pyoxynet(input_df=t.data_frame, model = 'murias_lab')
        df_estimates, dict_estimates = tf_lite_model_inference(tf_lite_model=tf_lite_model, input_df=t.data_frame, inference_stride=2)

        VT1 = 0
        VT2 = 0
        VO2VT1 = 0
        VO2VT2 = 0

        dict_estimates['VT1']['time'] = int(dict_estimates['VT1']['time'])
        dict_estimates['VT2']['time'] = int(dict_estimates['VT2']['time'])
        dict_estimates['VT1']['VO2'] = int(dict_estimates['VT1']['VO2'])
        dict_estimates['VT2']['VO2'] = int(dict_estimates['VT2']['VO2'])

        dict_estimates['VT1']['perc_VO2'] = np.round(dict_estimates['VT1']['VO2']/t.data_frame['VO2_I'].max() * 100, 1)
        dict_estimates['VT2']['perc_VO2'] = np.round(dict_estimates['VT2']['VO2']/t.data_frame['VO2_I'].max() * 100, 1)

        VT1_oxynet = dict_estimates['VT1']['time']
        VT2_oxynet = dict_estimates['VT2']['time']
        VO2VT1_oxynet = dict_estimates['VT1']['VO2']
        VO2VT2_oxynet = dict_estimates['VT2']['VO2']

        # Trim at VE max
        max_ve_index = t.raw_data_frame['VE'].idxmax()
        t.raw_data_frame = t.raw_data_frame.loc[:max_ve_index]

        # Rename ONLY for viz purposes
        # TODO: fix this sh*t
        t.raw_data_frame = t.raw_data_frame.rename(columns={'VO2': 'VO2_I'})
        t.raw_data_frame = t.raw_data_frame.rename(columns={'VCO2': 'VCO2_I'})
        t.raw_data_frame = t.raw_data_frame.rename(columns={'VE': 'VE_I'})
        t.raw_data_frame = t.raw_data_frame.rename(columns={'PetO2': 'PetO2_I'})
        t.raw_data_frame = t.raw_data_frame.rename(columns={'PetCO2': 'PetCO2_I'})
        t.raw_data_frame = t.raw_data_frame.rename(columns={'VEVCO2': 'VEVCO2_I'})
        t.raw_data_frame = t.raw_data_frame.rename(columns={'VEVO2': 'VEVO2_I'})

        plot_VEvsVO2 = CPET_var_plot_vs_O2(t.raw_data_frame, var_list=['VE_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
        plot_VCO2vsVO2 = CPET_var_plot_vs_O2(t.raw_data_frame, var_list=['VCO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
        plot_PetO2 = CPET_var_plot_vs_O2(t.raw_data_frame, var_list=['PetO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
        plot_PetCO2 = CPET_var_plot_vs_O2(t.raw_data_frame, var_list=['PetCO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
        plot_VEVO2 = CPET_var_plot_vs_O2(t.raw_data_frame, var_list=['VEVO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
        plot_VEVCO2 = CPET_var_plot_vs_O2(t.raw_data_frame, var_list=['VEVCO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])

        return render_template('plot_interpretation.html',
                                       VCO2vsVO2=plot_VCO2vsVO2,
                                       VEvsVO2=plot_VEvsVO2,
                                       PetO2=plot_PetO2,
                                       PetCO2=plot_PetCO2,
                                       VEVO2=plot_VEVO2,
                                       VEVCO2=plot_VEVCO2,
                                       CPET_data=dict_estimates)
    except:
        if 'file' not in request.files:
            dict_estimates = 'No file part'
        dict_estimates = {}
        return flask.jsonify('We are sorry to report that something went wrong with your file :-(')

@app.route('/CPET_generation', methods=['GET', 'POST'])
def CPET_generation():

    args = request.args
    fitness_group = args.get("fitness_group", default=None, type=int)
    df, gen_dict = pyoxynet.utilities.generate_CPET(generator, plot=False, fitness_group=fitness_group)

    return flask.jsonify(df.to_dict())

@app.route('/CPET_plot', methods=['GET', 'POST'])
def CPET_plot():

    if request.method == 'POST':
        if request.form.get('action1') == session['test_type'] or request.form.get('action2') == session['test_type']:
            session['correct'] = session['correct'] + 1
            session['tot_test'] = session['tot_test'] + 1
            reply = 'Your answer was CORRECT 😀 \n Total tests: ' + str(session['tot_test']) + ' (' + str(np.round(session['correct']/session['tot_test']*100, 2)) + '% correct)'
            return render_template('response.html', value=reply)
        else:
            if request.form.get('play') == 'PLAY' or request.form.get('start_over') == 'AGAIN':
                import random
                args = request.args
                fitness_group = args.get("fitness_group", default=None, type=int)

                if random.randint(0, 1) == 1:
                    generator = pyoxynet.utilities.load_tf_generator()
                    df, CPET_data = pyoxynet.utilities.generate_CPET(generator, plot=False, fitness_group=fitness_group, noise_factor=None)
                    print('Test was FAKE')
                    session['test_type'] = 'FAKE'
                else:
                    df, CPET_data = pyoxynet.utilities.draw_real_test()
                    print('Test was REAL')
                    session['test_type'] = 'REAL'

                df_oxynet, out_dict = pyoxynet.utilities.test_pyoxynet(input_df=df,
                                                                       model = 'murias_lab')

                VT1 = int(float(CPET_data['VT1']))
                VT2 = int(float(CPET_data['VT2']))
                VO2VT1 = int(float(CPET_data['VO2VT1']))
                VO2VT2 = int(float(CPET_data['VO2VT2']))

                VT1_oxynet = out_dict['VT1']['time']
                VT2_oxynet = out_dict['VT2']['time']
                VO2VT1_oxynet = out_dict['VT1']['VO2']
                VO2VT2_oxynet = out_dict['VT2']['VO2']

                plot_VEvsVCO2 = CPET_var_plot_vs_CO2(df, var_list=['VE_I'])
                plot_VCO2vsVO2 = CPET_var_plot_vs_O2(df, var_list=['VCO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
                plot_PetO2 = CPET_var_plot_vs_O2(df, var_list=['PetO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
                plot_PetCO2 = CPET_var_plot_vs_O2(df, var_list=['PetCO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
                plot_VEVO2 = CPET_var_plot_vs_O2(df, var_list=['VEVO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
                plot_VEVCO2 = CPET_var_plot_vs_O2(df, var_list=['VEVCO2_I'], VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])
                plot_oxynet = CPET_var_plot(df_oxynet, var_list=['p_md', 'p_hv', 'p_sv'], VT=[VT1, VT2, VT1_oxynet, VT2_oxynet])

                fake = Faker()
                fake_address = fake.address()
                fake_name = fake.name()

                data = [
                    {
                        'name': fake_name.split(' ')[0][0] + '. ' + fake_name.split(' ')[1],
                        'address': fake_address.replace('\n', ', ')
                    }
                ]

                return render_template('index.html',
                                       VCO2vsVO2=plot_VCO2vsVO2,
                                       VEvsVCO2=plot_VEvsVCO2,
                                       PetO2=plot_PetO2,
                                       PetCO2=plot_PetCO2,
                                       VEVO2=plot_VEVO2,
                                       VEVCO2=plot_VEVCO2,
                                       oxynet=plot_oxynet,
                                       data=data,
                                       CPET_data=CPET_data)
            else:
                session['wrong'] = session['wrong'] + 1
                session['tot_test'] = session['tot_test'] + 1
                reply = 'Your answer was WRONG 🙈 \n Total tests: ' + str(session['tot_test']) + ' (' + str(np.round(session['correct']/session['tot_test']*100, 2)) + '% correct)'
                return render_template('response.html', value=reply)

@app.route("/", methods=['GET', 'POST'])
@swag_from('swagger/homepage.yml')
def HelloWorld():

    session['test_type'] = 'NONE'

    if 'tot_test' not in session.keys() or not session['tot_test'] > 0:
        session['tot_test'] = 0
        session['correct'] = 0
        session['wrong'] = 0

    if request.method == 'POST':
        if request.form.get('play') == 'PLAY':
            return redirect(url_for('CPET_plot'))
        else:
            pass
    else:
        pass

    return render_template('homepage.html')

@app.route('/say_hello')
@swag_from('swagger/say_hello.yml')
def say_hello():
    # A very polite end point that says hello!
    return 'Hello World!'

def start_over():

    if request.method == 'POST':
        if request.form.get('start_over') == 'AGAIN':
            return redirect(url_for('CPET_plot'))
        else:
            pass
    else:
        pass

    return ''

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=port)
