import pickle

from pyoxynet import *

generator = load_tf_generator()
df = generate_CPET(generator, plot=True, duration=800, VT1=640, VT2=720)
test_pyoxynet(input_df=df)

here =0