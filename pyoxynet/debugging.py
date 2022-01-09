import pickle

from pyoxynet import *

tfl_model = load_tf_model()
df = load_csv_data()

X = df[['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I']]
XN = normalize(X)

import numpy as np

input_details = tfl_model.get_input_details()
output_details = tfl_model.get_output_details()

time_series_len = input_details[0]['shape'][1]
p_md = []
p_hv = []
p_sv = []
time = []

for i in np.arange(len(XN)-time_series_len):
    XN_array = np.asarray(XN[i:(i+time_series_len)])
    input_data = np.reshape(XN_array, input_details[0]['shape'])
    input_data = input_data.astype(np.float32)

    tfl_model.allocate_tensors()
    tfl_model.set_tensor(input_details[0]['index'], input_data)
    tfl_model.invoke()
    output_data = tfl_model.get_tensor(output_details[0]['index'])
    p_md.append(output_data[0][2])
    p_sv.append(output_data[0][1])
    p_hv.append(output_data[0][0])
    time.append(df.time[i])

import pandas as pd
df = pd.DataFrame()
df['time'] = time
df['p_md'] = p_md
df['p_hv'] = p_hv
df['p_sv'] = p_sv

from uniplot import plot

plot([p_md, p_sv, p_hv], title="Probabilities", color=True, legend_labels=['Moderate', 'Heavy', 'Severe'])

here =0