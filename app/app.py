import flask
import os
from flask import Flask, request, render_template, session, redirect, url_for
import pyoxynet
# import pyoxynet import utilities
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly
from faker import Faker

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9098))
app.secret_key = "super secret key"

generator = pyoxynet.utilities.load_tf_generator()

def CPET_var_plot_vs_CO2(df, var_list=[]):
    import json
    import plotly.express as px

    labels_dict = {}

    for lab_ in var_list:
        labels_dict[lab_] = lab_.replace('_', ' ').replace('I', '')

    fig = px.scatter(df.iloc[np.arange(0, len(df), 5)],
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
        template='ggplot2',
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

    fig = px.scatter(df.iloc[np.arange(0, len(df), 5)], x="VO2_I", y=var_list,
                     color_discrete_sequence=['white', 'gray'])
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
    fig.add_vline(x=VT1, line_width=3, line_dash="dash", line_color="dodgerblue", annotation_text="VT1")
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
        template='ggplot2',
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

    fig = px.line(df.iloc[np.arange(0, len(df), 5)], x="time", y=var_list,
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
        template='ggplot2',
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
        df = pd.DataFrame.from_dict(request_data)
        df_estimates, dict_estimates = pyoxynet.utilities.test_pyoxynet(input_df=df)
    except:
        dict_estimates = {}

    return flask.jsonify(dict_estimates)

@app.route('/read_json_ET', methods=['GET', 'POST'])
def read_json_ET():
    # Reads from json in ET formats
    args = request.args

    try:
        request_data = request.get_json(force=True)
        df = pyoxynet.utilities.load_exercise_threshold_app_data(data_dict=request_data)
        df_estimates, dict_estimates = pyoxynet.utilities.test_pyoxynet(input_df=df)
    except:
        dict_estimates = {}

    return flask.jsonify(dict_estimates)

@app.route('/read_csv', methods=['GET', 'POST'])
def read_csv():
    # Reads from csv and uses the pyoxynet parser
    args = request.args

    try:
        file = request.files['file']
        data = pd.read_csv(file)
        t = pyoxynet.Test('')
        t.set_data_extension('.csv')
        t.infer_metabolimeter(optional_data=data)
        t.load_file()
        t.create_data_frame()
        df_estimates, dict_estimates = pyoxynet.test_pyoxynet(input_df=t.data_frame)
    except:
        dict_estimates = {}

    return flask.jsonify(dict_estimates)

@app.route('/CPET_generation', methods=['GET', 'POST'])
def CPET_generation():

    args = request.args
    fitness_group = args.get("fitness_group", default=None, type=int)
    df, gen_dict = utilities.generate_CPET(generator, plot=False, fitness_group=fitness_group)

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
                    generator = utilities.load_tf_generator()
                    df, CPET_data = utilities.generate_CPET(generator, plot=False, fitness_group=fitness_group, noise_factor=None)
                    print('Test was FAKE')
                    session['test_type'] = 'FAKE'
                else:
                    df, CPET_data = utilities.draw_real_test()
                    print('Test was REAL')
                    session['test_type'] = 'REAL'

                df_oxynet, out_dict = utilities.test_pyoxynet(input_df=df)

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
