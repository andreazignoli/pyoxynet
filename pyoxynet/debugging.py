import pickle

from pyoxynet import utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# out_dict = test_pyoxynet()

generator = utilities.load_tf_generator()

# real_df, data_dict_real = draw_real_test()

for i_ in np.arange(100):
    df, data_dict_fake = utilities.generate_CPET(generator)
    # df.to_csv('/Users/andreazignoli/oxynet-interpreter-tf2/generated/gen_' + str(i_) + '.csv')
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

here =0