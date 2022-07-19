import flask
import os
from flask import Flask, request, render_template, session, redirect, url_for
from pyoxynet import *
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly
from faker import Faker

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9098))
app.secret_key = "super secret key"

def CPET_var_plot_vs_CO2(df, var_list=[]):
    import json
    import plotly.express as px

    fig = px.scatter(df.iloc[np.arange(0, len(df), 5)],
                     x="VCO2_I",
                     y=var_list, color_discrete_sequence=['white', 'gray'])
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))

    fig.update_layout(
        xaxis=dict(
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
        template='ggplot2',
        width=800, height=800
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def CPET_var_plot_vs_O2(df, var_list=[], VT=[0, 0]):
    import json
    import plotly.express as px

    VT1 = VT[0]
    VT2 = VT[1]

    fig = px.scatter(df.iloc[np.arange(0, len(df), 5)], x="VO2_I", y=var_list,
                     color_discrete_sequence=['white', 'gray'])
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
    fig.add_vline(x=VT1, line_width=3, line_dash="dash", line_color="green", annotation_text="VT1")
    fig.add_vline(x=VT2, line_width=3, line_dash="dash", line_color="red", annotation_text="VT2")

    fig.update_layout(
        xaxis=dict(
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
        template='ggplot2',
        width=800, height=800
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def CPET_var_plot(df, var_list=[], VT=[300, 400]):
    import json
    import plotly.express as px

    VT1 = VT[0]
    VT2 = VT[1]

    fig = px.line(df.iloc[np.arange(0, len(df), 5)], x="time", y=var_list,
                  color_discrete_sequence=['white', 'gray', 'black'])
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
    fig.add_vline(x=VT1, line_width=3, line_dash="dash", line_color="green", annotation_text="VT1")
    fig.add_vline(x=VT2, line_width=3, line_dash="dash", line_color="red", annotation_text="VT2")

    fig.update_layout(
        xaxis=dict(
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
        template='ggplot2',
        width=800, height=400
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

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

@app.route('/CPET_generation', methods=['GET', 'POST'])
def CPET_generation():

    args = request.args
    fitness_group = args.get("fitness_group", default=None, type=int)
    generator = load_tf_generator()
    df = generate_CPET(generator, plot=False, fitness_group=fitness_group)

    return flask.jsonify(df.to_dict())

@app.route('/CPET_plot', methods=['GET', 'POST'])
def CPET_plot():

    if request.method == 'POST':
        if request.form.get('action1') == session['test_type'] or request.form.get('action2') == session['test_type']:
            reply = 'Very good: your answer was CORRECT :)'
            return render_template('response.html', value=reply)
        else:
            if request.form.get('play') == 'PLAY' or request.form.get('start_over') == 'AGAIN':
                import random
                args = request.args
                fitness_group = args.get("fitness_group", default=None, type=int)

                if random.randint(0, 1) == 1:
                    generator = load_tf_generator()
                    df, CPET_data = generate_CPET(generator, plot=False, fitness_group=fitness_group, noise_factor=None)
                    print('Test was FAKE')
                    session['test_type'] = 'FAKE'
                else:
                    df, CPET_data = draw_real_test()
                    print('Test was REAL')
                    session['test_type'] = 'REAL'

                df_oxynet, out_dict = test_pyoxynet(input_df=df)
                VT1 = out_dict['VT1']['time']
                VT2 = out_dict['VT2']['time']
                VO2VT1 = out_dict['VT1']['VO2']
                VO2VT2 = out_dict['VT2']['VO2']

                plot_VO2VCO2 = CPET_var_plot_vs_O2(df, var_list=['VO2_I', 'VCO2_I'], VT=[VO2VT1, VO2VT2])
                plot_Pet = CPET_var_plot_vs_O2(df, var_list=['PetO2_I', 'PetCO2_I'], VT=[VO2VT1, VO2VT2])
                plot_VEVO2 = CPET_var_plot_vs_O2(df, var_list=['VEVO2_I', 'VEVCO2_I'], VT=[VO2VT1, VO2VT2])
                plot_oxynet = CPET_var_plot(df_oxynet, var_list=['p_md', 'p_hv', 'p_sv'], VT=[VT1, VT2])
                plot_VCO2vsVO2 = CPET_var_plot_vs_O2(df, var_list=['VCO2_I'], VT=[VO2VT1, VO2VT2])
                plot_VEvsVCO2 = CPET_var_plot_vs_CO2(df, var_list=['VE_I'])

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
                                       VO2VCO2=plot_VO2VCO2,
                                       Pet=plot_Pet,
                                       VEVO2=plot_VEVO2,
                                       oxynet=plot_oxynet,
                                       data=data,
                                       CPET_data=CPET_data)
            else:
                reply = 'Very bad: your answer was WRONG :('
                return render_template('response.html', value=reply)

@app.route("/", methods=['GET', 'POST'])
def HelloWorld():

    session['test_type'] = 'NONE'

    if request.method == 'POST':
        if request.form.get('play') == 'PLAY':
            return redirect(url_for('CPET_plot'))
        else:
            pass
    else:
        pass

    return render_template('homepage.html')

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
