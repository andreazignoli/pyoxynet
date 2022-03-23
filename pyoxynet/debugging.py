import pickle

from pyoxynet import *

generator = load_tf_generator()

real_df = draw_real_test()

df = generate_CPET(generator, plot=False, fitness_group=2)
test_pyoxynet(input_df=df)

here =0