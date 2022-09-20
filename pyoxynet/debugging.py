import pickle

from pyoxynet import utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random

generator = utilities.load_tf_generator()
plotting = True

for i_ in np.arange(10):
    resting = random.randint(0, 1)
    # df_fake, data_dict_fake = utilities.generate_CPET(generator,
    #                                                   noise_factor=None,
    #                                                   plot=False,
    #                                                   resting=resting,
    #                                                   training=True,
    #                                                   normalization=True)
    df_real, data_dict_real = utilities.draw_real_test()
    out_df_CNN, out_dict_CNN = utilities.test_pyoxynet(input_df=df_real, model='CNN', plot=False)
    out_df_transf, out_dict_transf = utilities.test_pyoxynet(input_df=df_real, model='transformer', plot=False)
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
        plt.vlines(x=out_dict_transf['VT1']['time'], ymin=20, ymax=100, colors='m')
        plt.vlines(x=out_dict_transf['VT2']['time'], ymin=20, ymax=100, colors='m')
        plt.subplot(3, 2, 3)
        plt.plot(df.time, df.VEVO2_I, 'b')
        plt.plot(df.time, df.VEVCO2_I, 'r')
        plt.vlines(x=VT1, ymin=0.02, ymax=0.05)
        plt.vlines(x=VT2, ymin=0.02, ymax=0.05)
        plt.vlines(x=out_dict_CNN['VT1']['time'], ymin=0.02, ymax=0.05, colors='k')
        plt.vlines(x=out_dict_CNN['VT2']['time'], ymin=0.02, ymax=0.05, colors='k')
        plt.vlines(x=out_dict_transf['VT1']['time'], ymin=0.02, ymax=0.05, colors='m')
        plt.vlines(x=out_dict_transf['VT2']['time'], ymin=0.02, ymax=0.05, colors='m')
        plt.subplot(3, 2, 4)
        plt.plot(df.time, df.PetO2_I, 'b')
        plt.plot(df.time, df.PetCO2_I, 'r')
        plt.vlines(x=VT1, ymin=20, ymax=120)
        plt.vlines(x=VT2, ymin=20, ymax=120)
        plt.vlines(x=out_dict_CNN['VT1']['time'], ymin=20, ymax=120, colors='k')
        plt.vlines(x=out_dict_CNN['VT2']['time'], ymin=20, ymax=120, colors='k')
        plt.vlines(x=out_dict_transf['VT1']['time'], ymin=20, ymax=120, colors='m')
        plt.vlines(x=out_dict_transf['VT2']['time'], ymin=20, ymax=120, colors='m')
        plt.subplot(3, 2, 5)
        plt.plot(df.time, df.VO2_I, 'b')
        plt.plot(df.time, df.VCO2_I, 'r')
        plt.vlines(x=VT1, ymin=350, ymax=1600)
        plt.vlines(x=VT2, ymin=1200, ymax=3000)
        plt.vlines(x=out_dict_CNN['VT1']['time'], ymin=350, ymax=2600, colors='k')
        plt.vlines(x=out_dict_CNN['VT2']['time'], ymin=1200, ymax=3000, colors='k')
        plt.vlines(x=out_dict_transf['VT1']['time'], ymin=350, ymax=2600, colors='m')
        plt.vlines(x=out_dict_transf['VT2']['time'], ymin=1200, ymax=3000, colors='m')
        plt.subplot(3, 2, 6)
        plt.scatter(df.VCO2_I, df.VE_I, 2)
        plt.cla()
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
        # here=0
        # plt.cla()

here =0

# df_out, results_dict = test_pyoxynet()
# Opening JSON file
with open('pyoxynet/pyoxynet/test/exercise_threshold_app_test.json') as json_file:
    data = json.load(json_file)

df_in = utilities.load_exercise_threshold_app_data(data_dict=data)
df_out, results_dict = utilities.test_pyoxynet(df_in)

# out_dict = test_pyoxynet()