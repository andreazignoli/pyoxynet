import pickle

from pyoxynet import utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# df_out, results_dict = test_pyoxynet()
# Opening JSON file
with open('pyoxynet/pyoxynet/test/exercise_threshold_app_test.json') as json_file:
    data = json.load(json_file)

df_in = utilities.load_exercise_threshold_app_data(data_dict=data)
df_out, results_dict = utilities.test_pyoxynet(df_in)

# out_dict = test_pyoxynet()

generator = utilities.load_tf_generator()
# real_df, data_dict_real = draw_real_test()

for i_ in np.arange(100):
    df, data_dict_fake = utilities.generate_CPET(generator)
    # df.to_csv('/Users/andreazignoli/oxynet-interpreter-tf2/generated/gen_' + str(i_) + '.csv')
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(df.VO2_I, df.VCO2_I)
    plt.subplot(2, 2, 2)
    plt.plot(df.time, df.VE_I)
    plt.vlines(x=int(data_dict_fake[0]['VT1']), ymin=0, ymax=100)
    plt.vlines(x=int(data_dict_fake[0]['VT2']), ymin=0, ymax=100)
    plt.subplot(2, 2, 3)
    plt.plot(df.time, df.VEVO2_I)
    plt.plot(df.time, df.VEVCO2_I)
    plt.vlines(x=int(data_dict_fake[0]['VT1']), ymin=0, ymax=0.05)
    plt.vlines(x=int(data_dict_fake[0]['VT2']), ymin=0, ymax=0.05)
    plt.subplot(2, 2, 4)
    plt.plot(df.time, df.PetO2_I, 'o')
    plt.vlines(x=int(data_dict_fake[0]['VT1']), ymin=100, ymax=120)
    plt.vlines(x=int(data_dict_fake[0]['VT2']), ymin=100, ymax=120)
    here=0
    plt.cla()
    out_df, out_dict = utilities.test_pyoxynet(input_df=df)
    plt.plot(out_df.time, out_df.p_sv)
    plt.plot(out_df.time, out_df.p_md)
    plt.plot(out_df.time, out_df.p_hv)
    plt.plot(df.time, df.domain)
    plt.plot(df.time, df.p_mF)
    plt.plot(df.time, df.p_hF)
    plt.plot(df.time, df.p_sF)
    plt.vlines(x=int(data_dict_fake[0]['VT1']), ymin=-1, ymax=1)
    plt.vlines(x=int(data_dict_fake[0]['VT2']), ymin=-1, ymax=1)
    plt.vlines(x=out_dict['VT1']['time'], ymin=-1, ymax=1, linestyles='dashed')
    plt.vlines(x=out_dict['VT2']['time'], ymin=-1, ymax=1, linestyles='dashed')
    here=0
    plt.cla()

here =0