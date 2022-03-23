import pickle

from pyoxynet import *

generator = load_tf_generator()

real_df, data_dict_real = draw_real_test()

df, data_dict_fake = generate_CPET(generator, plot=False, fitness_group=2)
test_pyoxynet(input_df=real_df)

here =0