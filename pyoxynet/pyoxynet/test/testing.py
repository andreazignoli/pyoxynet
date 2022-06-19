from pyoxynet import *
import json
import matplotlib.pyplot as plt

# Opening JSON file
with open('../test/exercise_threshold_app_test.json') as json_file:
    data = json.load(json_file)

df_in = load_exercise_threshold_app_data(data_dict=data)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(df_in.VCO2_I, df_in.VO2_I, df_in.VE_I)
ax.set_xlabel('VCO2')
ax.set_ylabel('VO2')
ax.set_zlabel('VE')

df_out, results_dict = test_pyoxynet(input_df=df_in)

here=0