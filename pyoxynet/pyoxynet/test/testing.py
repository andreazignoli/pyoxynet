from pyoxynet import *
import json

# Opening JSON file
with open('../test/exercise_threshold_app_test.json') as json_file:
    data = json.load(json_file)

df_in = load_exercise_threshold_app_data(data_dict=data)

df_out, results_dict = test_pyoxynet(input_df=df_in)

here=0