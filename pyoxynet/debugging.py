import pickle

from pyoxynet import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

generator = load_tf_generator()

# real_df, data_dict_real = draw_real_test()

for i_ in np.arange(20):
    df, data_dict_fake = generate_CPET(generator)
    df.to_csv('/Users/andreazignoli/oxynet-interpreter-tf2/generated/gen_' + str(i_) + '.csv')
    # test_pyoxynet(input_df=real_df)
    plt.plot(df.VO2_I)
    here=0

here =0