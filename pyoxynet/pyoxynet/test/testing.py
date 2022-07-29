import json
import matplotlib.pyplot as plt
import pyoxynet

from pyoxynet import *
# df_out, results_dict = test_pyoxynet()
# Opening JSON file
# file_name = '../test/exercise_threshold_app_test.json'
file_name = '/Users/andreazignoli/oxynet-interpreter-tf2/generated/gen_0.json'

with open(file_name) as json_file:
    data = json.load(json_file)

df_in = utilities.load_exercise_threshold_app_data(data_dict=data)
df_out, results_dict = utilities.test_pyoxynet(df_in)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(df_in.VCO2_I, df_in.VO2_I, df_in.VE_I)
ax.set_xlabel('VCO2')
ax.set_ylabel('VO2')
ax.set_zlabel('VE')


here=0