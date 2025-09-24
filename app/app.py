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
import tflite_runtime.interpreter as tflite
import warnings

# Suppress divide by zero warnings from numpy
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')

app = flask.Flask(__name__)
Swagger(app)
port = int(os.getenv("PORT", 9098))
app.secret_key = "super secret key"

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to store pre-loaded TFLite models
# Use absolute path that works both locally and in Docker
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'tf_lite_models', 'tfl_model.tflite')

print(f"Current directory: {current_dir}")
print(f"Looking for model at: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")

if not os.path.exists(model_path):
    # Try parent directory
    parent_model_path = os.path.join(current_dir, '..', 'tf_lite_models', 'tfl_model.tflite')
    print(f"Trying parent directory: {parent_model_path}")
    print(f"Parent model file exists: {os.path.exists(parent_model_path)}")
    if os.path.exists(parent_model_path):
        model_path = parent_model_path

models = {
    'model_1': tflite.Interpreter(model_path=model_path)
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

    print(var_list)
    print(df)

    labels_dict = {}

    for lab_ in var_list:
        labels_dict[lab_] = lab_.replace('_', ' ').replace('I', '')

    if "VO2_I" in df.columns:
        indip_var_ = "VO2_I"
    else:
        indip_var_ = "VO2_F"
    
    fig = px.scatter(df.iloc[np.arange(0, len(df))], x=indip_var_, y=var_list)
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

    fig = px.line(df.iloc[np.arange(0, len(df))], x="time", y=var_list)
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
    # fig.add_vline(x=VT1, line_width=3, line_dash="dash", line_color="dodgerblue", annotation_text="VT1")
    # fig.add_vline(x=VT2, line_width=3, line_dash="dash", line_color="red", annotation_text="VT2")
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

def create_fat_oxidation_plot(df):
    """
    Create fat oxidation rate plot based on load change points
    
    Parameters:
        df (DataFrame): Raw CPET data with load, VO2_I, VCO2_I columns
        
    Returns:
        str: JSON string for plotly plot
    """
    import json
    import plotly.express as px
    import numpy as np
    
    # Find load change points (where load increases)
    load_diff = df['load'].diff()
    change_points = df[load_diff > 0].index.tolist()
    
    # Add the end point to capture the last steady state
    if len(df) - 1 not in change_points:
        change_points.append(len(df) - 1)
    
    avg_loads = []
    fat_oxidation_rates = []
    cho_consumption_rates = []
    eefat_rates = []
    eecho_rates = []
    perc_ee_fat = []
    perc_ee_cho = []
    
    for point in change_points:
        # Get 60 samples before this change point (or all available)
        start_idx = max(0, point - 60)
        end_idx = point
        
        if end_idx - start_idx < 10:  # Skip if too few samples
            continue
            
        # Get the data slice
        slice_data = df.iloc[start_idx:end_idx]
        
        # Calculate averages
        avg_load = slice_data['load'].mean()
        avg_vo2 = slice_data['VO2_I'].mean() * 0.001  # Convert to L/min
        avg_vco2 = slice_data['VCO2_I'].mean() * 0.001  # Convert to L/min
        
        # Calculate fat oxidation rate: 1.695*VO2 - 1.701*VCO2 (g/min)
        fat_rate = 1.695 * avg_vo2 - 1.701 * avg_vco2
        
        # Calculate CHO consumption rate: 4.585*VCO2 - 3.226*VO2 (g/min)
        cho_rate = 4.585 * avg_vco2 - 3.226 * avg_vo2
        
        # Calculate energy expenditure rates using caloric equivalents
        # EEFAT: fat oxidation rate × 9.75 kcal/g (caloric equivalent from fatty acids)
        eefat_rate = fat_rate * 9.75
        # EECHO: CHO consumption rate × 4.18 kcal/g (caloric equivalent from glucose)
        eecho_rate = cho_rate * 4.18
        
        # Calculate percentage of energy expenditure from each substrate
        total_ee = eefat_rate + eecho_rate
        if total_ee > 0:
            perc_fat = (eefat_rate / total_ee) * 100
            perc_cho = (eecho_rate / total_ee) * 100
        else:
            perc_fat = 0
            perc_cho = 0
        
        avg_loads.append(avg_load)
        fat_oxidation_rates.append(fat_rate)
        cho_consumption_rates.append(cho_rate)
        eefat_rates.append(eefat_rate)
        eecho_rates.append(eecho_rate)
        perc_ee_fat.append(perc_fat)
        perc_ee_cho.append(perc_cho)
    
    if not avg_loads:  # No valid points found
        return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create polynomial fits (3rd degree with y-intercept = 0)
    # For polynomial with y-intercept = 0: y = ax + bx² + cx³
    avg_loads_array = np.array(avg_loads)
    fat_rates_array = np.array(fat_oxidation_rates)
    cho_rates_array = np.array(cho_consumption_rates)
    
    # Create design matrix for polynomial without constant term (forces y-intercept = 0)
    X = np.column_stack([avg_loads_array, avg_loads_array**2, avg_loads_array**3])
    
    # Fit polynomials using least squares
    fat_coeffs = np.linalg.lstsq(X, fat_rates_array, rcond=None)[0]
    cho_coeffs = np.linalg.lstsq(X, cho_rates_array, rcond=None)[0]
    
    # Fit polynomials for percentage energy expenditure
    perc_ee_fat_array = np.array(perc_ee_fat)
    perc_ee_cho_array = np.array(perc_ee_cho)
    perc_fat_coeffs = np.linalg.lstsq(X, perc_ee_fat_array, rcond=None)[0]
    perc_cho_coeffs = np.linalg.lstsq(X, perc_ee_cho_array, rcond=None)[0]
    
    # Find FAT MAX point by finding maximum of polynomial fit
    # For polynomial y = ax + bx² + cx³, derivative is dy/dx = a + 2bx + 3cx²
    # Set derivative to zero and solve: a + 2bx + 3cx² = 0
    # This is a quadratic equation: 3cx² + 2bx + a = 0
    a, b, c = fat_coeffs[0], fat_coeffs[1], fat_coeffs[2]
    
    fatmax_load = None
    fatmax_rate = None
    
    if abs(c) > 1e-10:  # Avoid division by zero
        # Solve quadratic equation: 3cx² + 2bx + a = 0
        discriminant = (2*b)**2 - 4*(3*c)*a
        if discriminant >= 0:
            x1 = (-2*b + np.sqrt(discriminant)) / (2*3*c)
            x2 = (-2*b - np.sqrt(discriminant)) / (2*3*c)
            
            # Check which solution is in our data range and gives maximum
            valid_solutions = []
            for x in [x1, x2]:
                if min(avg_loads) <= x <= max(avg_loads):
                    # Check second derivative to confirm it's a maximum (d²y/dx² = 2b + 6cx < 0)
                    second_derivative = 2*b + 6*c*x
                    if second_derivative < 0:  # Maximum (concave down)
                        y = a*x + b*x**2 + c*x**3
                        valid_solutions.append((x, y))
            
            if valid_solutions:
                # Take the solution with highest fat oxidation rate if multiple valid solutions
                fatmax_load, fatmax_rate = max(valid_solutions, key=lambda sol: sol[1])

    # Generate smooth curves for plotting
    x_smooth = np.linspace(min(avg_loads), max(avg_loads), 100)
    fat_fitted = fat_coeffs[0]*x_smooth + fat_coeffs[1]*x_smooth**2 + fat_coeffs[2]*x_smooth**3
    cho_fitted = cho_coeffs[0]*x_smooth + cho_coeffs[1]*x_smooth**2 + cho_coeffs[2]*x_smooth**3
    
    # Generate smooth curves for percentage energy expenditure
    perc_ee_fat_smooth = perc_fat_coeffs[0]*x_smooth + perc_fat_coeffs[1]*x_smooth**2 + perc_fat_coeffs[2]*x_smooth**3
    perc_ee_cho_smooth = perc_cho_coeffs[0]*x_smooth + perc_cho_coeffs[1]*x_smooth**2 + perc_cho_coeffs[2]*x_smooth**3
    
    # Filter to only consider workloads > 50W for crossover detection
    valid_load_indices = x_smooth > 50
    
    # Find crossover point where CHO% becomes > FAT%
    crossover_load = None
    crossover_fat_rate = None
    crossover_cho_rate = None
    
    crossover_indices = np.where((perc_ee_cho_smooth > perc_ee_fat_smooth) & valid_load_indices)[0]
    if len(crossover_indices) > 0:
        crossover_idx = crossover_indices[0]  # First index where CHO > FAT at >50W
        crossover_load = x_smooth[crossover_idx]
        crossover_fat_rate = fat_fitted[crossover_idx]
        crossover_cho_rate = cho_fitted[crossover_idx]
    
    # Create subplot with secondary y-axis
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add fat oxidation data points on primary y-axis
    fig.add_trace(
        go.Scatter(x=avg_loads, y=fat_oxidation_rates, name="Fat Oxidation (Data)", 
                  mode='markers', marker=dict(size=8, color='#ff6b35'),
                  showlegend=True),
        secondary_y=False,
    )
    
    # Add fat oxidation fitted curve on primary y-axis (subtle)
    fig.add_trace(
        go.Scatter(x=x_smooth, y=fat_fitted, name="Fat Oxidation (Fit)", 
                  mode='lines', line=dict(color='rgba(255, 107, 53, 0.4)', width=2.5, dash='solid'),
                  showlegend=True),
        secondary_y=False,
    )
    
    # Add CHO consumption data points on secondary y-axis
    fig.add_trace(
        go.Scatter(x=avg_loads, y=cho_consumption_rates, name="CHO Consumption (Data)", 
                  mode='markers', marker=dict(size=8, color='#2ECC71'),
                  showlegend=True),
        secondary_y=True,
    )
    
    # Add CHO consumption fitted curve on secondary y-axis (subtle)
    fig.add_trace(
        go.Scatter(x=x_smooth, y=cho_fitted, name="CHO Consumption (Fit)", 
                  mode='lines', line=dict(color='rgba(46, 204, 113, 0.4)', width=2.5, dash='solid'),
                  showlegend=True),
        secondary_y=True,
    )
    
    # Add FAT MAX point if found
    if fatmax_load is not None and fatmax_rate is not None:
        fig.add_trace(
            go.Scatter(x=[fatmax_load], y=[fatmax_rate], name="FAT MAX", 
                      mode='markers+text', 
                      marker=dict(size=12, color='red', symbol='star', line=dict(color='darkred', width=2)),
                      text=["FAT MAX"], textposition="top center",
                      textfont=dict(size=12, color='red'),
                      showlegend=True),
            secondary_y=False,
        )
    
    
    # Set x-axis title with grid styling
    fig.update_xaxes(
        title_text="Average Load (Watts)", 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    
    # Set y-axes titles with grid styling
    fig.update_yaxes(
        title_text="Fat Oxidation Rate (g/min)", 
        secondary_y=False, 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    fig.update_yaxes(
        title_text="CHO Consumption Rate (g/min)", 
        secondary_y=True, 
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    
    # Update layout
    fig.update_layout(
        title="Substrate Utilization vs Load with Polynomial Fits",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        annotations=[
            dict(
                text=(f"<b>FAT MAX:</b><br>" +
                      f"• Load: {fatmax_load:.0f}W<br>" +
                      f"• FAT: {fatmax_rate:.3f} g/min<br>" +
                      f"• CHO: {np.polyval(np.append(cho_coeffs[::-1], 0), fatmax_load):.3f} g/min"
                      if fatmax_load is not None and fatmax_rate is not None 
                      else "No FAT MAX found"),
                xref="paper", yref="paper",
                x=0.85, y=0.02,
                xanchor='right', yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color='#666'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd',
                borderwidth=1
            )
        ]
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_substrate_vs_vo2max_plot(df):
    """
    Create substrate utilization plot vs %VO2max
    
    Parameters:
        df (DataFrame): Raw CPET data with load, VO2_I, VCO2_I columns
        
    Returns:
        str: JSON string for plotly plot
    """
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Calculate VO2max as the maximum of 20-second rolling average
    # Assuming data is sampled every second, 20-second window
    rolling_vo2 = df['VO2_I'].rolling(window=20, min_periods=10).mean()
    vo2_max = rolling_vo2.max()
    
    # Find load change points (where load increases)
    load_diff = df['load'].diff()
    change_points = df[load_diff > 0].index.tolist()
    
    # Add the end point to capture the last steady state
    if len(df) - 1 not in change_points:
        change_points.append(len(df) - 1)
    
    percent_vo2max = []
    fat_oxidation_rates = []
    cho_consumption_rates = []
    eefat_rates = []
    eecho_rates = []
    perc_ee_fat = []
    perc_ee_cho = []
    
    for point in change_points:
        # Get 60 samples before this change point (or all available)
        start_idx = max(0, point - 60)
        end_idx = point
        
        if end_idx - start_idx < 10:  # Skip if too few samples
            continue
            
        # Get the data slice
        slice_data = df.iloc[start_idx:end_idx]
        
        # Calculate averages
        avg_vo2 = slice_data['VO2_I'].mean() * 0.001  # Convert to L/min
        avg_vco2 = slice_data['VCO2_I'].mean() * 0.001  # Convert to L/min
        
        # Calculate %VO2max
        percent_vo2 = (avg_vo2 * 1000 / vo2_max) * 100  # Convert back to ml/min for %
        
        # Calculate fat oxidation rate: 1.695*VO2 - 1.701*VCO2 (g/min)
        fat_rate = 1.695 * avg_vo2 - 1.701 * avg_vco2
        
        # Calculate CHO consumption rate: 4.585*VCO2 - 3.226*VO2 (g/min)
        cho_rate = 4.585 * avg_vco2 - 3.226 * avg_vo2
        
        # Calculate energy expenditure rates using caloric equivalents
        # EEFAT: fat oxidation rate × 9.75 kcal/g (caloric equivalent from fatty acids)
        eefat_rate = fat_rate * 9.75
        # EECHO: CHO consumption rate × 4.18 kcal/g (caloric equivalent from glucose)
        eecho_rate = cho_rate * 4.18
        
        # Calculate percentage of energy expenditure from each substrate
        total_ee = eefat_rate + eecho_rate
        if total_ee > 0:
            perc_fat = (eefat_rate / total_ee) * 100
            perc_cho = (eecho_rate / total_ee) * 100
        else:
            perc_fat = 0
            perc_cho = 0
        
        percent_vo2max.append(percent_vo2)
        fat_oxidation_rates.append(fat_rate)
        cho_consumption_rates.append(cho_rate)
        eefat_rates.append(eefat_rate)
        eecho_rates.append(eecho_rate)
        perc_ee_fat.append(perc_fat)
        perc_ee_cho.append(perc_cho)
    
    if not percent_vo2max:  # No valid points found
        return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create polynomial fits (3rd degree with y-intercept = 0)
    # For polynomial with y-intercept = 0: y = ax + bx² + cx³
    percent_vo2max_array = np.array(percent_vo2max)
    fat_rates_array = np.array(fat_oxidation_rates)
    cho_rates_array = np.array(cho_consumption_rates)
    
    # Create design matrix for polynomial without constant term (forces y-intercept = 0)
    X = np.column_stack([percent_vo2max_array, percent_vo2max_array**2, percent_vo2max_array**3])
    
    # Fit polynomials using least squares
    fat_coeffs = np.linalg.lstsq(X, fat_rates_array, rcond=None)[0]
    cho_coeffs = np.linalg.lstsq(X, cho_rates_array, rcond=None)[0]
    
    # Fit polynomials for percentage energy expenditure
    perc_ee_fat_array = np.array(perc_ee_fat)
    perc_ee_cho_array = np.array(perc_ee_cho)
    perc_fat_coeffs = np.linalg.lstsq(X, perc_ee_fat_array, rcond=None)[0]
    perc_cho_coeffs = np.linalg.lstsq(X, perc_ee_cho_array, rcond=None)[0]
    
    # Find FAT MAX point by finding maximum of polynomial fit
    # For polynomial y = ax + bx² + cx³, derivative is dy/dx = a + 2bx + 3cx²
    # Set derivative to zero and solve: a + 2bx + 3cx² = 0
    # This is a quadratic equation: 3cx² + 2bx + a = 0
    a, b, c = fat_coeffs[0], fat_coeffs[1], fat_coeffs[2]
    
    fatmax_percent_vo2 = None
    fatmax_rate = None
    
    if abs(c) > 1e-10:  # Avoid division by zero
        # Solve quadratic equation: 3cx² + 2bx + a = 0
        discriminant = (2*b)**2 - 4*(3*c)*a
        if discriminant >= 0:
            x1 = (-2*b + np.sqrt(discriminant)) / (2*3*c)
            x2 = (-2*b - np.sqrt(discriminant)) / (2*3*c)
            
            # Check which solution is in our data range and gives maximum
            valid_solutions = []
            for x in [x1, x2]:
                if min(percent_vo2max) <= x <= max(percent_vo2max):
                    # Check second derivative to confirm it's a maximum (d²y/dx² = 2b + 6cx < 0)
                    second_derivative = 2*b + 6*c*x
                    if second_derivative < 0:  # Maximum (concave down)
                        y = a*x + b*x**2 + c*x**3
                        valid_solutions.append((x, y))
            
            if valid_solutions:
                # Take the solution with highest fat oxidation rate if multiple valid solutions
                fatmax_percent_vo2, fatmax_rate = max(valid_solutions, key=lambda sol: sol[1])
    
    # Get corresponding loads from original change points
    loads_at_change_points = []
    if len(percent_vo2max) > 1:
        for point in change_points[:len(percent_vo2max)]:
            loads_at_change_points.append(df.iloc[point]['load'])
    
    # Calculate corresponding load at FAT MAX
    fatmax_load = None
    if fatmax_percent_vo2 is not None:
        # Find the load corresponding to this %VO2max
        # We need to interpolate from our original data points
        if len(loads_at_change_points) > 1:
            # Use numpy interpolation instead of scipy
            try:
                fatmax_load = float(np.interp(fatmax_percent_vo2, percent_vo2max, loads_at_change_points))
            except:
                # Fallback: find closest point
                closest_idx = np.argmin(np.abs(np.array(percent_vo2max) - fatmax_percent_vo2))
                fatmax_load = loads_at_change_points[closest_idx]

    # Generate smooth curves for plotting
    x_smooth = np.linspace(min(percent_vo2max), max(percent_vo2max), 100)
    fat_fitted = fat_coeffs[0]*x_smooth + fat_coeffs[1]*x_smooth**2 + fat_coeffs[2]*x_smooth**3
    cho_fitted = cho_coeffs[0]*x_smooth + cho_coeffs[1]*x_smooth**2 + cho_coeffs[2]*x_smooth**3
    
    # Generate smooth curves for percentage energy expenditure
    perc_ee_fat_smooth = perc_fat_coeffs[0]*x_smooth + perc_fat_coeffs[1]*x_smooth**2 + perc_fat_coeffs[2]*x_smooth**3
    perc_ee_cho_smooth = perc_cho_coeffs[0]*x_smooth + perc_cho_coeffs[1]*x_smooth**2 + perc_cho_coeffs[2]*x_smooth**3
    
    # Filter to only consider workloads > 50W for crossover detection
    # Convert each %VO2max point to corresponding load
    valid_load_indices = np.zeros_like(x_smooth, dtype=bool)
    if len(loads_at_change_points) > 1:
        for i, vo2_percent in enumerate(x_smooth):
            try:
                corresponding_load = float(np.interp(vo2_percent, percent_vo2max, loads_at_change_points))
                valid_load_indices[i] = corresponding_load > 50
            except:
                valid_load_indices[i] = False
    
    # Find crossover point where CHO% becomes > FAT%
    crossover_point = None
    crossover_load = None
    crossover_fat_rate = None
    crossover_cho_rate = None
    
    crossover_indices = np.where((perc_ee_cho_smooth > perc_ee_fat_smooth) & valid_load_indices)[0]
    if len(crossover_indices) > 0:
        crossover_idx = crossover_indices[0]  # First index where CHO > FAT at >50W
        crossover_point = x_smooth[crossover_idx]
        crossover_fat_rate = fat_fitted[crossover_idx]
        crossover_cho_rate = cho_fitted[crossover_idx]
        
        # Find corresponding load at crossover %VO2max
        if len(loads_at_change_points) > 1:
            try:
                crossover_load = float(np.interp(crossover_point, percent_vo2max, loads_at_change_points))
            except:
                closest_idx = np.argmin(np.abs(np.array(percent_vo2max) - crossover_point))
                crossover_load = loads_at_change_points[closest_idx]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add fat oxidation data points on primary y-axis
    fig.add_trace(
        go.Scatter(x=percent_vo2max, y=fat_oxidation_rates, name="Fat Oxidation (Data)", 
                  mode='markers', marker=dict(size=8, color='#ff6b35'),
                  showlegend=True),
        secondary_y=False,
    )
    
    # Add fat oxidation fitted curve on primary y-axis (subtle)
    fig.add_trace(
        go.Scatter(x=x_smooth, y=fat_fitted, name="Fat Oxidation (Fit)", 
                  mode='lines', line=dict(color='rgba(255, 107, 53, 0.4)', width=2.5, dash='solid'),
                  showlegend=True),
        secondary_y=False,
    )
    
    # Add CHO consumption data points on secondary y-axis
    fig.add_trace(
        go.Scatter(x=percent_vo2max, y=cho_consumption_rates, name="CHO Consumption (Data)", 
                  mode='markers', marker=dict(size=8, color='#2ECC71'),
                  showlegend=True),
        secondary_y=True,
    )
    
    # Add CHO consumption fitted curve on secondary y-axis (subtle)
    fig.add_trace(
        go.Scatter(x=x_smooth, y=cho_fitted, name="CHO Consumption (Fit)", 
                  mode='lines', line=dict(color='rgba(46, 204, 113, 0.4)', width=2.5, dash='solid'),
                  showlegend=True),
        secondary_y=True,
    )
    
    # Add FAT MAX point if found
    if fatmax_percent_vo2 is not None and fatmax_rate is not None:
        fig.add_trace(
            go.Scatter(x=[fatmax_percent_vo2], y=[fatmax_rate], name="FAT MAX", 
                      mode='markers+text', 
                      marker=dict(size=12, color='red', symbol='star', line=dict(color='darkred', width=2)),
                      text=["FAT MAX"], textposition="top center",
                      textfont=dict(size=12, color='red'),
                      showlegend=True),
            secondary_y=False,
        )
    
    
    # Set x-axis title with grid styling
    fig.update_xaxes(
        title_text="% VO₂max", 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    
    # Set y-axes titles with grid styling
    fig.update_yaxes(
        title_text="Fat Oxidation Rate (g/min)", 
        secondary_y=False, 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    fig.update_yaxes(
        title_text="CHO Consumption Rate (g/min)", 
        secondary_y=True, 
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    
    # Update layout
    fig.update_layout(
        title="Substrate Utilization vs % VO₂max with Polynomial Fits",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        annotations=[
            dict(
                text=(f"<b>FAT MAX:</b><br>" +
                      f"• {fatmax_percent_vo2:.1f}% VO₂max<br>" +
                      f"• FAT: {fatmax_rate:.3f} g/min<br>" +
                      f"• CHO: {np.polyval(np.append(cho_coeffs[::-1], 0), fatmax_percent_vo2):.3f} g/min<br>" +
                      (f"• Load: {fatmax_load:.0f}W" if fatmax_load is not None else "• Load: N/A")
                      if fatmax_percent_vo2 is not None else "No FAT MAX found"),
                xref="paper", yref="paper",
                x=0.85, y=0.02,
                xanchor='right', yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color='#666'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd',
                borderwidth=1
            )
        ]
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_energy_contribution_vs_load_plot(df):
    """
    Create energy contribution percentage plot vs load
    
    Parameters:
        df (DataFrame): Raw CPET data with load, VO2_I, VCO2_I columns
        
    Returns:
        str: JSON string for plotly plot
    """
    import json
    import plotly.graph_objects as go
    import numpy as np
    
    # Find load change points (where load increases)
    load_diff = df['load'].diff()
    change_points = df[load_diff > 0].index.tolist()
    
    # Add the end point to capture the last steady state
    if len(df) - 1 not in change_points:
        change_points.append(len(df) - 1)
    
    avg_loads = []
    perc_ee_fat = []
    perc_ee_cho = []
    
    for point in change_points:
        # Get 60 samples before this change point (or all available)
        start_idx = max(0, point - 60)
        end_idx = point
        
        if end_idx - start_idx < 10:  # Skip if too few samples
            continue
            
        # Get the data slice
        slice_data = df.iloc[start_idx:end_idx]
        
        # Calculate averages
        avg_load = slice_data['load'].mean()
        avg_vo2 = slice_data['VO2_I'].mean() * 0.001  # Convert to L/min
        avg_vco2 = slice_data['VCO2_I'].mean() * 0.001  # Convert to L/min
        
        # Calculate fat oxidation rate: 1.695*VO2 - 1.701*VCO2 (g/min)
        fat_rate = 1.695 * avg_vo2 - 1.701 * avg_vco2
        
        # Calculate CHO consumption rate: 4.585*VCO2 - 3.226*VO2 (g/min)
        cho_rate = 4.585 * avg_vco2 - 3.226 * avg_vo2
        
        # Calculate energy expenditure rates using caloric equivalents
        # EEFAT: fat oxidation rate × 9.75 kcal/g (caloric equivalent from fatty acids)
        eefat_rate = fat_rate * 9.75
        # EECHO: CHO consumption rate × 4.18 kcal/g (caloric equivalent from glucose)
        eecho_rate = cho_rate * 4.18
        
        # Calculate percentage of energy expenditure from each substrate
        total_ee = eefat_rate + eecho_rate
        if total_ee > 0:
            perc_fat = (eefat_rate / total_ee) * 100
            perc_cho = (eecho_rate / total_ee) * 100
        else:
            perc_fat = 0
            perc_cho = 0
        
        avg_loads.append(avg_load)
        perc_ee_fat.append(perc_fat)
        perc_ee_cho.append(perc_cho)
    
    if not avg_loads:  # No valid points found
        return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create plot
    fig = go.Figure()
    
    # Convert to numpy arrays for fitting
    x_data = np.array(avg_loads)
    fat_data = np.array(perc_ee_fat)
    cho_data = np.array(perc_ee_cho)
    
    # Add FAT percentage data points
    fig.add_trace(go.Scatter(
        x=avg_loads, 
        y=perc_ee_fat, 
        name="FAT % (data)", 
        mode='markers',
        marker=dict(size=8, color='#ff6b35'),
        showlegend=True
    ))
    
    # Fit 3rd degree polynomial to FAT percentage data
    if len(x_data) >= 4:  # Need at least 4 points for 3rd degree polynomial
        fat_coeffs = np.polyfit(x_data, fat_data, 3)
        # Generate smooth curve
        x_smooth = np.linspace(min(x_data), max(x_data), 100)
        fat_fit = np.polyval(fat_coeffs, x_smooth)
        
        # Add fitted curve for FAT
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=fat_fit,
            name="FAT % (fit)",
            mode='lines',
            line=dict(width=3, color='rgba(255, 107, 53, 0.6)', dash='solid'),
            showlegend=True
        ))
    
    # Add CHO percentage data points
    fig.add_trace(go.Scatter(
        x=avg_loads, 
        y=perc_ee_cho, 
        name="CHO % (data)", 
        mode='markers',
        marker=dict(size=8, color='#2ECC71'),
        showlegend=True
    ))
    
    # Fit 3rd degree polynomial to CHO percentage data
    if len(x_data) >= 4:  # Need at least 4 points for 3rd degree polynomial
        cho_coeffs = np.polyfit(x_data, cho_data, 3)
        # Generate smooth curve
        cho_fit = np.polyval(cho_coeffs, x_smooth)
        
        # Add fitted curve for CHO
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=cho_fit,
            name="CHO % (fit)",
            mode='lines',
            line=dict(width=3, color='rgba(46, 204, 113, 0.6)', dash='solid'),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Energy Contribution vs Load",
        xaxis_title="Load (Watts)",
        yaxis_title="Energy Contribution (%)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    
    # Add grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_vo2_vs_load_plot(df):
    """
    Create VO2 vs load scatter plot
    
    Parameters:
        df (DataFrame): Raw CPET data with load, VO2_I columns
        
    Returns:
        str: JSON string for plotly plot    
    """
    import json
    import plotly.graph_objects as go
    import numpy as np
    
    # Find load change points (where load increases)
    load_diff = df['load'].diff()
    change_points = df[load_diff > 0].index.tolist()
    
    # Add the end point to capture the last steady state
    if len(df) - 1 not in change_points:
        change_points.append(len(df) - 1)
    
    avg_loads = []
    avg_vo2_values = []
    ge_values = []
    
    for point in change_points:
        # Get 60 samples before this change point (or all available)
        start_idx = max(0, point - 60)
        end_idx = point
        
        if end_idx - start_idx < 10:  # Skip if too few samples
            continue
            
        # Get the data slice
        slice_data = df.iloc[start_idx:end_idx]
        
        # Calculate averages
        avg_load = slice_data['load'].mean()
        avg_vo2 = slice_data['VO2_I'].mean() * 0.001  # Convert to L/min
        avg_vco2 = slice_data['VCO2_I'].mean() * 0.001  # Convert to L/min
        
        # Calculate GE (Gross Efficiency)
        # EEm = 3.941*VO2/1000+1.106*VCO2/1000 (energy expenditure metabolic)
        EEm = 3.941 * avg_vo2 + 1.106 * avg_vco2  # Already in L/min
        
        # Emecc = 60 * average_load / 1000 (mechanical energy)
        Emecc = 60 * avg_load / 1000
        
        # GE_perc = Emecc/(EEm*4.186) * 100 (gross efficiency percentage)
        if EEm > 0:
            GE_perc = Emecc / (EEm * 4.186) * 100
        else:
            GE_perc = 0
        
        avg_loads.append(avg_load)
        avg_vo2_values.append(avg_vo2)
        ge_values.append(GE_perc)
    
    if not avg_loads:  # No valid points found
        return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create plot
    fig = go.Figure()
    
    # Convert to numpy arrays for fitting
    x_data = np.array(avg_loads)
    y_data = np.array(avg_vo2_values)
    
    # Add VO2 data points
    fig.add_trace(go.Scatter(
        x=avg_loads, 
        y=avg_vo2_values, 
        name="VO2 (data)", 
        mode='markers',
        marker=dict(size=8, color='#3498db'),
        showlegend=True
    ))
    
    # Add GE text annotations above each point (skip first point)
    for i, (x, y, ge) in enumerate(zip(avg_loads, avg_vo2_values, ge_values)):
        if i == 0:  # Skip the first point
            continue
        fig.add_annotation(
            x=x,
            y=y,
            text=f"GE: {ge:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#2c3e50",
            ax=0,
            ay=-30,  # Position above the point
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#2c3e50",
            borderwidth=1,
            font=dict(size=10, color="#2c3e50")
        )
    
    # Fit linear regression to VO2 vs load data
    if len(x_data) >= 2:  # Need at least 2 points for linear fit
        coeffs = np.polyfit(x_data, y_data, 1)
        slope = coeffs[0]  # mlO2/min per Watt -> multiply by 1000 to get mlO2/W
        intercept = coeffs[1]  # L/min -> multiply by 1000 to get mlO2/min
        
        # Calculate R² (coefficient of determination)
        y_pred = np.polyval(coeffs, x_data)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Convert to appropriate units for display
        vo2_resting_ml = intercept * 1000  # Convert L/min to mlO2/min
        efficiency_ml_per_w = slope * 1000  # Convert L/min/W to mlO2/min/W
        
        # Generate smooth curve
        x_smooth = np.linspace(min(x_data), max(x_data), 100)
        y_fit = np.polyval(coeffs, x_smooth)
        
        # Add fitted curve with equation in legend
        equation_text = f"VO2 (fit)<br>VO2 resting: {vo2_resting_ml:.0f} mlO2/min<br>Efficiency: {efficiency_ml_per_w:.1f} mlO2/W<br>R² = {r_squared:.3f}"
        
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_fit,
            name=equation_text,
            mode='lines',
            line=dict(width=3, color='rgba(52, 152, 219, 0.6)', dash='solid'),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="VO2 vs Load",
        xaxis_title="Load (Watts)",
        yaxis_title="VO2 (L/min)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )
    
    # Add grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_energy_contribution_vs_vo2max_plot(df):
    """
    Create energy contribution percentage plot vs %VO2max
    
    Parameters:
        df (DataFrame): Raw CPET data with load, VO2_I, VCO2_I columns
        
    Returns:
        str: JSON string for plotly plot
    """
    import json
    import plotly.graph_objects as go
    import numpy as np
    
    # Calculate VO2max as the maximum of 20-second rolling average
    rolling_vo2 = df['VO2_I'].rolling(window=20, min_periods=10).mean()
    vo2_max = rolling_vo2.max()
    
    # Find load change points (where load increases)
    load_diff = df['load'].diff()
    change_points = df[load_diff > 0].index.tolist()
    
    # Add the end point to capture the last steady state
    if len(df) - 1 not in change_points:
        change_points.append(len(df) - 1)
    
    percent_vo2max = []
    perc_ee_fat = []
    perc_ee_cho = []
    
    for point in change_points:
        # Get 60 samples before this change point (or all available)
        start_idx = max(0, point - 60)
        end_idx = point
        
        if end_idx - start_idx < 10:  # Skip if too few samples
            continue
            
        # Get the data slice
        slice_data = df.iloc[start_idx:end_idx]
        
        # Calculate averages
        avg_vo2 = slice_data['VO2_I'].mean() * 0.001  # Convert to L/min
        avg_vco2 = slice_data['VCO2_I'].mean() * 0.001  # Convert to L/min
        
        # Calculate %VO2max
        percent_vo2 = (avg_vo2 * 1000 / vo2_max) * 100  # Convert back to ml/min for %
        
        # Calculate fat oxidation rate: 1.695*VO2 - 1.701*VCO2 (g/min)
        fat_rate = 1.695 * avg_vo2 - 1.701 * avg_vco2
        
        # Calculate CHO consumption rate: 4.585*VCO2 - 3.226*VO2 (g/min)
        cho_rate = 4.585 * avg_vco2 - 3.226 * avg_vo2
        
        # Calculate energy expenditure rates using caloric equivalents
        # EEFAT: fat oxidation rate × 9.75 kcal/g (caloric equivalent from fatty acids)
        eefat_rate = fat_rate * 9.75
        # EECHO: CHO consumption rate × 4.18 kcal/g (caloric equivalent from glucose)
        eecho_rate = cho_rate * 4.18
        
        # Calculate percentage of energy expenditure from each substrate
        total_ee = eefat_rate + eecho_rate
        if total_ee > 0:
            perc_fat = (eefat_rate / total_ee) * 100
            perc_cho = (eecho_rate / total_ee) * 100
        else:
            perc_fat = 0
            perc_cho = 0
        
        percent_vo2max.append(percent_vo2)
        perc_ee_fat.append(perc_fat)
        perc_ee_cho.append(perc_cho)
    
    if not percent_vo2max:  # No valid points found
        return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create plot
    fig = go.Figure()
    
    # Convert to numpy arrays for fitting
    x_data = np.array(percent_vo2max)
    fat_data = np.array(perc_ee_fat)
    cho_data = np.array(perc_ee_cho)
    
    # Add FAT percentage data points
    fig.add_trace(go.Scatter(
        x=percent_vo2max, 
        y=perc_ee_fat, 
        name="FAT % (data)", 
        mode='markers',
        marker=dict(size=8, color='#ff6b35'),
        showlegend=True
    ))
    
    # Fit 3rd degree polynomial to FAT percentage data
    if len(x_data) >= 4:  # Need at least 4 points for 3rd degree polynomial
        fat_coeffs = np.polyfit(x_data, fat_data, 3)
        # Generate smooth curve
        x_smooth = np.linspace(min(x_data), max(x_data), 100)
        fat_fit = np.polyval(fat_coeffs, x_smooth)
        
        # Add fitted curve for FAT
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=fat_fit,
            name="FAT % (fit)",
            mode='lines',
            line=dict(width=3, color='rgba(255, 107, 53, 0.6)', dash='solid'),
            showlegend=True
        ))
    
    # Add CHO percentage data points
    fig.add_trace(go.Scatter(
        x=percent_vo2max, 
        y=perc_ee_cho, 
        name="CHO % (data)", 
        mode='markers',
        marker=dict(size=8, color='#2ECC71'),
        showlegend=True
    ))
    
    # Fit 3rd degree polynomial to CHO percentage data
    if len(x_data) >= 4:  # Need at least 4 points for 3rd degree polynomial
        cho_coeffs = np.polyfit(x_data, cho_data, 3)
        # Generate smooth curve
        cho_fit = np.polyval(cho_coeffs, x_smooth)
        
        # Add fitted curve for CHO
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=cho_fit,
            name="CHO % (fit)",
            mode='lines',
            line=dict(width=3, color='rgba(46, 204, 113, 0.6)', dash='solid'),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Energy Contribution vs % VO₂max",
        xaxis_title="% VO₂max",
        yaxis_title="Energy Contribution (%)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        yaxis=dict(range=[0, 100])
    )
    
    # Add grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_load_with_gas_exchange_plot(df, VT=[300, 400]):
    """
    Create load vs time plot with VO2 and VCO2 on secondary y-axis
    
    Parameters:
        df (DataFrame): Raw CPET data with time, load, VO2_I, VCO2_I columns
        VT (list): Ventilatory thresholds [VT1, VT2, VT1_oxynet, VT2_oxynet]
        
    Returns:
        str: JSON string for plotly plot
    """
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    VT1, VT2, VT1_oxynet, VT2_oxynet = VT
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add VO2 and VCO2 traces on secondary y-axis first (behind load) with softer colors
    if 'VO2_I' in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["VO2_I"], name="VO₂", 
                      line=dict(color='rgba(70, 130, 180, 0.7)', width=2)),  # Softer steel blue with alpha
            secondary_y=True,
        )
    
    if 'VCO2_I' in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["VCO2_I"], name="VCO₂", 
                      line=dict(color='rgba(205, 92, 92, 0.7)', width=2)),  # Softer indian red with alpha
            secondary_y=True,
        )
    
    # Add load trace on primary y-axis last (on top)
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["load"], name="Load", 
                  line=dict(color='#ff6b35', width=3)),
        secondary_y=False,
    )
    
    # Add vertical lines for thresholds
    fig.add_vline(x=VT1_oxynet, line_width=2, line_color="dodgerblue", 
                  annotation_text="VT1", annotation_position="top")
    fig.add_vline(x=VT2_oxynet, line_width=2, line_color="red", 
                  annotation_text="VT2", annotation_position="top")
    
    # Set x-axis title with grid styling
    fig.update_xaxes(
        title_text="Time (min)", 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    
    # Set y-axes titles with grid styling
    fig.update_yaxes(
        title_text="Load (Watts)", 
        secondary_y=False, 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    fig.update_yaxes(
        title_text="Gas Exchange (ml/min)", 
        secondary_y=True, 
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    
    # Update layout
    fig.update_layout(
        title="Load vs Time with Gas Exchange",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_domain_probabilities_time_plot(df_estimates, VT=[300, 400]):
    """Create domain probabilities vs time plot with filled areas
    
    Parameters:
        df_estimates (DataFrame): DataFrame containing time, p_md, p_hv, p_sv columns
        VT (list): Ventilatory thresholds [VT1, VT2, VT1_oxynet, VT2_oxynet]
    
    Returns:
        JSON string of Plotly figure
    """
    import json
    import plotly.graph_objects as go
    
    VT1, VT2, VT1_oxynet, VT2_oxynet = VT
    
    fig = go.Figure()
    
    # Add filled area plots for each domain probability
    fig.add_trace(go.Scatter(
        x=df_estimates['time'],
        y=df_estimates['p_md'],
        fill='tonexty',
        mode='lines',
        name='Moderate Domain (p_md)',
        line=dict(width=0),
        fillcolor='rgba(46, 204, 113, 0.6)',  # Green
        hovertemplate='<b>Moderate Domain</b><br>Time: %{x}s<br>Probability: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_estimates['time'],
        y=df_estimates['p_md'] + df_estimates['p_hv'],
        fill='tonexty',
        mode='lines',
        name='Heavy Domain (p_hv)',
        line=dict(width=0),
        fillcolor='rgba(241, 196, 15, 0.6)',  # Yellow/Gold
        hovertemplate='<b>Heavy Domain</b><br>Time: %{x}s<br>Probability: %{customdata:.3f}<extra></extra>',
        customdata=df_estimates['p_hv']
    ))
    
    fig.add_trace(go.Scatter(
        x=df_estimates['time'],
        y=df_estimates['p_md'] + df_estimates['p_hv'] + df_estimates['p_sv'],
        fill='tonexty',
        mode='lines',
        name='Severe Domain (p_sv)',
        line=dict(width=0),
        fillcolor='rgba(231, 76, 60, 0.6)',  # Red/Orange
        hovertemplate='<b>Severe Domain</b><br>Time: %{x}s<br>Probability: %{customdata:.3f}<extra></extra>',
        customdata=df_estimates['p_sv']
    ))
    
    # Add ventilatory threshold lines
    if VT1_oxynet > 0:
        fig.add_vline(x=VT1_oxynet, line_width=2, line_color="blue", 
                     annotation_text="VT1", annotation_position="top")
    if VT2_oxynet > 0:
        fig.add_vline(x=VT2_oxynet, line_width=2, line_color="red", 
                     annotation_text="VT2", annotation_position="top")
    
    # Update layout
    fig.update_layout(
        title="Exercise Domain Probabilities vs Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Probability",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified',
        yaxis=dict(range=[0, 1])
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_domain_probabilities_vo2_plot(df_estimates, VT=[0, 0, 0, 0]):
    """Create VO2 vs time scatter plot colored by dominant exercise domain
    
    Parameters:
        df_estimates (DataFrame): DataFrame containing time, VO2, p_md, p_hv, p_sv columns
        VT (list): Ventilatory thresholds [VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet]
    
    Returns:
        JSON string of Plotly figure
    """
    import json
    import plotly.graph_objects as go
    import numpy as np
    
    VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet = VT
    
    # Determine dominant domain for each point
    probabilities = df_estimates[['p_md', 'p_hv', 'p_sv']].values
    dominant_domain = np.argmax(probabilities, axis=1)
    
    # Define colors and domain names
    domain_colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Green, Yellow, Red
    domain_names = ['Moderate', 'Heavy', 'Severe']
    
    fig = go.Figure()
    
    # Determine which VO2 column to use
    if 'VO2' in df_estimates.columns:
        vo2_column = 'VO2'
    elif 'VO2_I' in df_estimates.columns:
        vo2_column = 'VO2_I'
    elif 'VO2_F' in df_estimates.columns:
        vo2_column = 'VO2_F'
    else:
        # If no VO2 column found, create a dummy one
        df_estimates = df_estimates.copy()
        df_estimates['VO2'] = range(len(df_estimates))
        vo2_column = 'VO2'
    
    # Add scatter traces for each domain
    for domain_idx, (color, name) in enumerate(zip(domain_colors, domain_names)):
        mask = dominant_domain == domain_idx
        if mask.any():
            max_prob = probabilities[mask, domain_idx]
            fig.add_trace(go.Scatter(
                x=df_estimates.loc[mask, 'time'],
                y=df_estimates.loc[mask, vo2_column],
                mode='markers',
                name=f'{name} Domain',
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.8
                ),
                hovertemplate=f'<b>{name} Domain</b><br>Time: %{{x}}s<br>VO₂: %{{y:.0f}} ml/min<br>Probability: %{{customdata:.3f}}<extra></extra>',
                customdata=max_prob
            ))
    
    # Add ventilatory threshold lines
    if VO2VT1_oxynet > 0:
        fig.add_hline(y=VO2VT1_oxynet, line_width=2, line_color="blue", line_dash="dash",
                     annotation_text="VT1", annotation_position="right")
    if VO2VT2_oxynet > 0:
        fig.add_hline(y=VO2VT2_oxynet, line_width=2, line_color="red", line_dash="dash",
                     annotation_text="VT2", annotation_position="right")
    
    # Update layout
    fig.update_layout(
        title="Exercise Domain Classification: VO₂ vs Time",
        xaxis_title="Time (seconds)",
        yaxis_title="VO₂ (ml/min)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.02, y=0.98),
        hovermode='closest'
    )
    
    # Add grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
    df = df.resample('1s').mean()
    df = df.interpolate()
    df['VO2_20s'] = df.VO2_I.rolling(20, win_type='triang', center=True).mean().bfill().ffill()
    df = df.reset_index()
    df = df.drop('timestamp', axis=1)

    if 'VCO2VO2_I' not in df.columns:
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            df['VCO2VO2_I'] = np.where(df['VO2_I'] != 0, 
                                      df['VCO2_I'].values/df['VO2_I'].values, 
                                      0)
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
    try:
        VT1_condition = out_df['p_hv'] >= out_df['p_md']
        VT2_condition = out_df['p_sv'] <= out_df['p_hv']
        
        VT1_matches = out_df[VT1_condition].index
        VT2_matches = out_df[VT2_condition].index
        
        if len(VT1_matches) == 0 or len(VT2_matches) == 0:
            # Return default values when inference fails
            out_dict['VT1']['time'] = 0
            out_dict['VT2']['time'] = 0
            out_dict['VT1']['HR'] = 0
            out_dict['VT2']['HR'] = 0
            out_dict['VT1']['VE'] = 0
            out_dict['VT2']['VE'] = 0
            out_dict['VT1']['VO2'] = 0
            out_dict['VT2']['VO2'] = 0
            return out_df, out_dict
        
        VT1_index = int(VT1_matches[0] - int(past_points / inference_stride))
        VT2_index = int(VT2_matches[-1] - int(past_points / inference_stride))

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
        
    except:
        # Handle any remaining index errors gracefully
        out_dict['VT1']['time'] = 0
        out_dict['VT2']['time'] = 0
        out_dict['VT1']['HR'] = 0
        out_dict['VT2']['HR'] = 0
        out_dict['VT1']['VE'] = 0
        out_dict['VT2']['VE'] = 0
        out_dict['VT1']['VO2'] = 0
        out_dict['VT2']['VO2'] = 0

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
        
        # Create probability plots using dedicated functions
        plot_probabilities_time = create_domain_probabilities_time_plot(df_estimates, VT=[VT1, VT2, VT1_oxynet, VT2_oxynet])
        plot_probabilities_vo2 = create_domain_probabilities_vo2_plot(df_estimates, VT=[VO2VT1, VO2VT2, VO2VT1_oxynet, VO2VT2_oxynet])

        # Check if 'load' column exists and create load vs time plot
        plot_load = None
        plot_fat_oxidation = None
        plot_substrate_vo2max = None
        plot_energy_contribution_load = None
        plot_energy_contribution_vo2max = None
        show_load_plot = False
        show_fat_plot = False
        show_energy_plot = False
        
        if 'load' in t.raw_data_frame.columns:
            plot_load = create_load_with_gas_exchange_plot(t.raw_data_frame, VT=[VT1, VT2, VT1_oxynet, VT2_oxynet])
            show_load_plot = True
            
            # Create fat oxidation analysis if we have the required columns and load data is not all zeros
            if ('VO2_I' in t.raw_data_frame.columns and 'VCO2_I' in t.raw_data_frame.columns and 
                not (t.raw_data_frame['load'] == 0).all()):
                plot_fat_oxidation = create_fat_oxidation_plot(t.raw_data_frame)
                plot_substrate_vo2max = create_substrate_vs_vo2max_plot(t.raw_data_frame)
                plot_energy_contribution_load = create_energy_contribution_vs_load_plot(t.raw_data_frame)
                plot_energy_contribution_vo2max = create_energy_contribution_vs_vo2max_plot(t.raw_data_frame)
                plot_vo2_vs_load = create_vo2_vs_load_plot(t.raw_data_frame)
                show_fat_plot = True
                show_energy_plot = True

        return render_template('plot_interpretation.html',
                                       VCO2vsVO2=plot_VCO2vsVO2,
                                       VEvsVO2=plot_VEvsVO2,
                                       PetO2=plot_PetO2,
                                       PetCO2=plot_PetCO2,
                                       VEVO2=plot_VEVO2,
                                       VEVCO2=plot_VEVCO2,
                                       CPET_data=dict_estimates,
                                       load_plot=plot_load,
                                       show_load_plot=show_load_plot,
                                       fat_oxidation_plot=plot_fat_oxidation,
                                       substrate_vo2max_plot=plot_substrate_vo2max,
                                       show_fat_plot=show_fat_plot,
                                       energy_contribution_load_plot=plot_energy_contribution_load,
                                       energy_contribution_vo2max_plot=plot_energy_contribution_vo2max,
                                       vo2_vs_load_plot=plot_vo2_vs_load,
                                       show_energy_plot=show_energy_plot,
                                       probabilities_time_plot=plot_probabilities_time,
                                       probabilities_vo2_plot=plot_probabilities_vo2)
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
