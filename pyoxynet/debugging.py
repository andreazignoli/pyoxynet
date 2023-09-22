import pickle

from pyoxynet import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import os
from pyoxynet import testing
from pyoxynet import utilities
from bs4 import BeautifulSoup

##################

# file_path = '/Users/andreazignoli/oxynet-interpreter-tf2/training_data/ARC_VO2_FOL.csv'
# df = pd.read_csv(file_path)
# df_est, dict_est = test_pyoxynet(df, model='TCN')


##################

# generator = load_tf_generator()
# df_fake, data_dict_fake = generate_CPET(generator, noise_factor=None)
# CPET_data = pd.DataFrame.from_dict(data_dict_fake['data'], orient='columns')
# df_est_fake, dict_est_fake = test_pyoxynet(df_fake)

here=0

#test_file = '/Users/andreazignoli/Downloads/CPET_files_try_me/4.xls'
#filename, file_extension = os.path.splitext(test_file)

file_dir = '/Users/andreazignoli/oxynet-interpreter-tf2/generated/json/'
json_list = []
for files in os.listdir(file_dir):
    if files.endswith(".json"):
        json_list.append(files)

VO2VT1_TDE = []
VO2VT2_TDE = []
VO2VT1_NN = []
VO2VT2_NN = []

for json_file in json_list:
    with open(file_dir + json_file, 'r') as f:
        CPET_data = json.load(f)
    # new_dict_CPET = {}
    # new_dict_CPET[0] = {'data': []}
    # new_dict_CPET[0]['data'] = CPET_data['data']

    try:
        df = utilities.load_exercise_threshold_app_data(data_dict=CPET_data)
        df_fake = pd.DataFrame.from_dict(CPET_data[0]['data'], orient='columns')
        df_estimates, dict_estimates = utilities.test_pyoxynet(input_df=df, n_inputs=5, model='TCN')

        VO2VT1_TDE.append(int(CPET_data[0]['VO2VT1']))
        VO2VT2_TDE.append(int(CPET_data[0]['VO2VT2']))

        VO2VT1_NN.append(int(dict_estimates['VT1']['VO2']))
        VO2VT2_NN.append(int(dict_estimates['VT2']['VO2']))

    except:
        print('Could not process ', json_file)
        pass

    here = 0

test_file = '/Users/andreazignoli/Downloads/CPET_files_try_me/4.xls'
filename, file_extension = os.path.splitext(test_file)
t = testing.Test(filename)
t.set_data_extension(file_extension)
t.infer_metabolimeter()
t.load_file()
t.create_data_frame()
#
df_estimates, dict_estimates = utilities.test_pyoxynet(input_df=t.data_frame, model='CNN')
pass
here=0

if False:
    # test_file = '/Users/andreazignoli/Downloads/Standard_Test_1_2022.12.05_16.54.00_.xml'
    #
    # with open(test_file, 'r') as f:
    #     data = f.read()
    # Bs_data = BeautifulSoup(data, "xml")
    #
    # rows = Bs_data.find_all('Row')
    #
    # # Create an empty list to store the row data
    # data = []
    # # Iterate over the rows
    # for row in rows:
    #     row_data = []
    #     # Find all the 'Cell' elements in the row
    #     cells = row.find_all('Cell')
    #     # Iterate over the cells
    #     for cell in cells:
    #         # Get the value from the 'Data' element within the 'Cell'
    #         value = cell.find('Data').text
    #         row_data.append(value)
    #     data.append(row_data)
    #
    # df = pd.DataFrame()
    # flag_ON = False
    # for n, cells in enumerate(data):
    #     try:
    #         if cells[0] == 't':
    #             flag_ON = True
    #         if flag_ON:
    #             df = pd.concat([df, pd.DataFrame(cells)])
    #     except:
    #         pass
    pass

#
# model = load_tf_model(n_inputs=6, past_points=40, model='CNN')
#
# t = Test(filename)
# t.set_data_extension(file_extension)
# t.infer_metabolimeter()
# t.load_file()
# t.clear_file_name()
# t.create_data_frame()
#
# t.generate_csv()

generator = load_tf_generator()
df_fake, data_dict_fake = generate_CPET(generator, noise_factor=None)



for i_ in np.arange(50):

    print('--------------------')
    print('--------------------')
    print('GENERATING =>>', str(i_))
    print('--------------------')
    print('--------------------')

    # GENERATE THE FAKE ONES
    df_fake, data_dict_fake = generate_CPET(generator,
                                            noise_factor=None,
                                            resting=0)
    df_breath_fake = pd.DataFrame.from_dict(data_dict_fake['data'])
    df_est_fake, dict_est_fake = test_pyoxynet(df_fake)

    data_dict_fake['oxynet'] = {}
    data_dict_fake['oxynet']['LT'] = dict_est_fake['VT1']['VO2']
    data_dict_fake['oxynet']['RCP'] = dict_est_fake['VT2']['VO2']

    file_id = 'fake_#' + str(i_).zfill(3)
    data_dict_fake['id'] = file_id
    df_fake.to_csv('/Users/andreazignoli/oxynet-interpreter-tf2/test_4_paper/csv/fake/' + file_id + '.csv')
    json_object_fake = json.dumps([data_dict_fake])
    # Writing to sample.json
    with open('/Users/andreazignoli/oxynet-interpreter-tf2/test_4_paper/json/fake/' + file_id + '.json', "w") as outfile:
        outfile.write(json_object_fake)

    # # GENERATE THE REAL ONES
    # df_real, data_dict_real = draw_real_test(resting=False)
    # df_breath_real = pd.DataFrame.from_dict(data_dict_real['data'])
    # df_est_real, dict_est_real = test_pyoxynet(df_real)
    #
    # data_dict_real['oxynet'] = {}
    # data_dict_real['oxynet']['LT'] = dict_est_real['VT1']['VO2']
    # data_dict_real['oxynet']['RCP'] = dict_est_real['VT2']['VO2']
    #
    # file_id = 'real_#' + str(i_).zfill(3)
    # data_dict_real['id'] = file_id
    # df_real.to_csv('/Users/andreazignoli/oxynet-interpreter-tf2/test_4_paper/csv/real/' + file_id + '.csv')
    # json_object_real = json.dumps([data_dict_real])
    # # Writing to sample.json
    # with open('/Users/andreazignoli/oxynet-interpreter-tf2/test_4_paper/json/real/' + file_id + '.json', "w") as outfile:
    #     outfile.write(json_object_real)

    here=0

# my_model = load_tf_model(n_inputs=6, past_points=40, model='CNN')
# generator = load_tf_generator()

# df_real, data_dict_real = utilities.draw_real_test()

# df_fake, data_dict_fake = generate_CPET(my_generator)

# explainer = load_explainer(my_model)
# df_real, data_dict_real = utilities.draw_real_test()
# df_real.to_csv('../oxynet-interpreter-tf2/output/df_CPET.csv')
#
# out_df_CNN, out_dict_CNN = utilities.test_pyoxynet(input_df=df_real, model='CNN', plot=True)
# out_df_CNN.to_csv('../oxynet-interpreter-tf2/output/df_CNN.csv')
#
# # Serializing json
# json_object = json.dumps(out_dict_CNN, indent=4)
#
# # Writing to sample.json
# with open("../oxynet-interpreter-tf2/output/out_dict_CNN.json", "w") as outfile:
#     outfile.write(json_object)
#
# shap_values, df_shap = compute_shap(explainer, df_real, shap_stride=2)
# df_shap.to_csv('../oxynet-interpreter-tf2/output/df_shap.csv')
#
# df_all = pd.read_csv('pyoxynet/pyoxynet/data_test/database_statistics_resting.csv')
# df_all.to_csv('../oxynet-interpreter-tf2/output/df_all.csv')
#
# generator = utilities.load_tf_generator()
plotting = True

for i_ in np.arange(10):
    resting = random.randint(0, 1)
    df_fake, data_dict_fake = generate_CPET(generator,
                                                      noise_factor=None,
                                                      plot=False,
                                                      resting=resting,
                                                      training=True,
                                                      normalization=False)
    # df_real, data_dict_real = utilities.draw_real_test()
    out_df_CNN, out_dict_CNN = test_pyoxynet(input_df=df_fake, model='CNN', plot=False)
    # out_df_transf, out_dict_transf = test_pyoxynet(input_df=df_fake, model='transformer', plot=False)
    # print(out_dict_CNN)
    # print(out_dict_transf)
    # file_id = 'generated_#' + str(i_).zfill(3)
    # data_dict_fake['id'] = file_id
    # df.to_csv('/Users/andreazignoli/oxynet-interpreter-tf2/generated/csv/' + file_id + '.csv')
    # json_object = json.dumps([data_dict_fake])
    # # Writing to sample.json
    # with open('/Users/andreazignoli/oxynet-interpreter-tf2/generated/json/' + file_id + '.json', "w") as outfile:
    #     outfile.write(json_object)
    if plotting:
        df = df_fake
        data_dict = data_dict_fake
        VT1 = int(data_dict['VT1'])
        VT2 = int(data_dict['VT2'])
        fig = plt.figure()
        plt.subplot(3, 2, 1)
        plt.scatter(df.VO2_I, df.VCO2_I, 2)
        plt.subplot(3, 2, 2)
        plt.plot(df.time, df.VE_I, 'g')
        plt.vlines(x=VT1, ymin=20, ymax=100)
        plt.vlines(x=VT2, ymin=20, ymax=100)
        plt.vlines(x=out_dict_CNN['VT1']['time'], ymin=20, ymax=100, colors='k')
        plt.vlines(x=out_dict_CNN['VT2']['time'], ymin=20, ymax=100, colors='k')
        # plt.vlines(x=out_dict_transf['VT1']['time'], ymin=20, ymax=100, colors='m')
        # plt.vlines(x=out_dict_transf['VT2']['time'], ymin=20, ymax=100, colors='m')
        plt.subplot(3, 2, 3)
        plt.plot(df.time, df.VEVO2_I, 'b')
        plt.plot(df.time, df.VEVCO2_I, 'r')
        plt.vlines(x=VT1, ymin=0.02, ymax=0.05)
        plt.vlines(x=VT2, ymin=0.02, ymax=0.05)
        plt.vlines(x=out_dict_CNN['VT1']['time'], ymin=0.02, ymax=0.05, colors='k')
        plt.vlines(x=out_dict_CNN['VT2']['time'], ymin=0.02, ymax=0.05, colors='k')
        # plt.vlines(x=out_dict_transf['VT1']['time'], ymin=0.02, ymax=0.05, colors='m')
        # plt.vlines(x=out_dict_transf['VT2']['time'], ymin=0.02, ymax=0.05, colors='m')
        plt.subplot(3, 2, 4)
        plt.plot(df.time, df.PetO2_I, 'b')
        plt.plot(df.time, df.PetCO2_I, 'r')
        plt.vlines(x=VT1, ymin=20, ymax=120)
        plt.vlines(x=VT2, ymin=20, ymax=120)
        plt.vlines(x=out_dict_CNN['VT1']['time'], ymin=20, ymax=120, colors='k')
        plt.vlines(x=out_dict_CNN['VT2']['time'], ymin=20, ymax=120, colors='k')
        # plt.vlines(x=out_dict_transf['VT1']['time'], ymin=20, ymax=120, colors='m')
        # plt.vlines(x=out_dict_transf['VT2']['time'], ymin=20, ymax=120, colors='m')
        plt.subplot(3, 2, 5)
        plt.plot(df.time, df.VO2_I, 'b')
        plt.plot(df.time, df.VCO2_I, 'r')
        plt.vlines(x=VT1, ymin=350, ymax=1600)
        plt.vlines(x=VT2, ymin=1200, ymax=3000)
        plt.vlines(x=out_dict_CNN['VT1']['time'], ymin=350, ymax=2600, colors='k')
        plt.vlines(x=out_dict_CNN['VT2']['time'], ymin=1200, ymax=3000, colors='k')
        # plt.vlines(x=out_dict_transf['VT1']['time'], ymin=350, ymax=2600, colors='m')
        # plt.vlines(x=out_dict_transf['VT2']['time'], ymin=1200, ymax=3000, colors='m')
        plt.subplot(3, 2, 6)
        plt.scatter(df.VCO2_I, df.VE_I, 2)
        # out_df, out_dict = utilities.test_pyoxynet(input_df=df)
        # plt.plot(out_df.time, out_df.p_sv)
        # plt.plot(out_df.time, out_df.p_md)
        # plt.plot(out_df.time, out_df.p_hv)
        # plt.plot(df.time, df.domain)
        # plt.plot(df.time, df.p_mF)
        # plt.plot(df.time, df.p_hF)
        # plt.plot(df.time, df.p_sF)
        # plt.vlines(x=int(data_dict_fake['VT1']), ymin=-1, ymax=1)
        # plt.vlines(x=int(data_dict_fake['VT2']), ymin=-1, ymax=1)
        # plt.vlines(x=out_dict['VT1']['time'], ymin=-1, ymax=1, linestyles='dashed')
        # plt.vlines(x=out_dict['VT2']['time'], ymin=-1, ymax=1, linestyles='dashed')
        here=0
        # plt.cla()

here =0

# df_out, results_dict = test_pyoxynet()
# Opening JSON file
with open('pyoxynet/pyoxynet/test/exercise_threshold_app_test.json') as json_file:
    data = json.load(json_file)

df_in = utilities.load_exercise_threshold_app_data(data_dict=data)
df_out, results_dict = utilities.test_pyoxynet(df_in)

# out_dict = test_pyoxynet()