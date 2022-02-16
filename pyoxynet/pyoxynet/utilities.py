def PrintHello(hello='hello'):
    """This function prints to screen.

    Args:
       name (str):  The name to use.

    Returns:
       none

    """
    
    print(hello)
    
    return

def normalize(df):
    """Pandas df normalisation

    Parameters:
        df (pd df) : input df

    Returns:
        result (pd df) : output df

    """

    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = 2 * (df[feature_name] - min_value) / (max_value - min_value) - 1
    result = result.fillna(0)

    return result

def optimal_filter(t, y, my_lambda):
    """A bad ass optimisation filter
    
    Parameters: 
        t : array
            Independent coord array
        y : array
            Dependent coord array
        my_lambda : float
            Smoothing factor

    Returns:
        x : array
            Filtered variable
    
    """

    import numpy as np

    # be robust for non-monotonic x variables (made for the time in CPET specifically)
    for i in np.arange(1, len(t)-1):
        if t[i+1] == t[i]:
            t[i+1] = t[i] + 0.1

    h  = 0.5 * np.concatenate([[t[1]-t[0]], t[2:] - t[0:-2], [t[-1] - t[-2]]])
    # Robustness
    m = np.median(h[h > 0])
    # Assign the median to the zero elements
    h[h == 0] = m

    dg = np.divide(my_lambda,h)
    # symmetric tri-diagonal system
    a = - dg[1:]
    b = np.diff(t) + dg[0:-1] + dg[1:]
    u = np.diff(y)
    # Solution of the system
    n = len(u)

    for j in np.arange(0, n - 1):
        mu = a[j] / b[j]
        b[j + 1] = b[j + 1] - mu * a[j]
        u[j + 1] = u[j + 1] - mu * u[j]

    u[n-1] = u[n-1] / b[n-1]

    for j in np.arange(n-2, -1, -1):
        u[j] = (u[j] - a[j] * u[j + 1]) / b[j]

    # Retrieving solution
    x = np.empty([len(y), ])
    x[0] = y[0] + my_lambda * u[0] / h[0]

    for i in np.arange(0, n):
        x[i + 1] = x[i] + (t[i + 1] - t[i]) * u[i]

    return x

def load_tf_model(n_inputs=7, past_points=40):
    """This function loads the saved tflite models.

    Args:
       n_inputs (int):  Number of input variables.
       past_points (int):  Number of past inputs in the time-series.

    Returns:
       interpreter (tflite interpreter) : handle on the TFLite interpreter

    """

    import importlib_resources
    import pickle
    from io import BytesIO
    import pyoxynet.models

    # get the model
    pip_install_tflite()
    import tflite_runtime.interpreter as tflite

    if n_inputs==7 and past_points==40:
        # load the classic Oxynet model configuration
        print('Classic Oxynet configuration model uploaded')
        tfl_model_binaries = importlib_resources.read_binary(pyoxynet.models, 'tfl_model.pickle')
    if n_inputs==5 and past_points==40:
        # load the 5 input model configuration (e.g. in this case when on CO2 info is included)
        print('Specific configuration model uploaded (no VCO2 available)')
        tfl_model_binaries = importlib_resources.read_binary(pyoxynet.models, 'tfl_model_5_40.pickle')

    try:
        tfl_model_decoded = pickle.loads(tfl_model_binaries)
        # save model locally on tmp
        open('/tmp/tfl_model' + '.tflite', 'wb').write(tfl_model_decoded.getvalue())
        interpreter = tflite.Interpreter(model_path='/tmp/tfl_model.tflite')
        return interpreter
    except:
        print('Could not find a model that could satisfy the input size required')
        return None  

def load_tf_generator():
    """This function loads the saved tflite generator model.

    Args:
       None

    Returns:
       generator (tflite generator) : handle on the TFLite generator

    """

    import importlib_resources
    import pickle
    from io import BytesIO
    import pyoxynet.models

    # get the model
    pip_install_tflite()
    import tflite_runtime.interpreter as tflite

    print('Classic Oxynet configuration model uploaded')
    tfl_model_binaries = importlib_resources.read_binary(pyoxynet.models, 'generator.pickle')

    try:
        tfl_model_decoded = pickle.loads(tfl_model_binaries)
        # save model locally on tmp
        open('/tmp/generator' + '.tflite', 'wb').write(tfl_model_decoded.getvalue())
        generator = tflite.Interpreter(model_path='/tmp/generator.tflite')
        return generator
    except:
        print('Could not load the generator')
        return None

def pip_install_tflite():
    """Makes sure TFLite is installed

    Parameters: 
        none

    Returns:
        none

    """
    import os
    import pkg_resources
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s" % (i.key) for i in installed_packages])

    if 'tflite-runtime' in installed_packages_list:
        print('Tflite runtime already present in the package list (skipping)')
    else:
        os.system("pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime")

def test_tfl_model(interpreter):
    """Test if the model is running correclty

    Parameters: 
        interpreter (loaded tf.lite.Interpreter) : Loaded interpreter TFLite model

    Returns:
        x (array) : Model output example

    """
    import numpy as np

    # Allocate tensors.
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])

def load_csv_data(csv_file='data_test.csv'):
    """Loads data from csv file (returns test data if no arguments)

    Parameters:
        n_inputs (int) :  Number of input variables.
        past_points (int) :  Number of past inputs in the time-series.

    Returns:
        df (pandas df) : Model output example

    """

    from importlib import resources
    import pandas as pd
    import pyoxynet.data_test

    if csv_file=='data_test.csv':
        import pkgutil
        from io import StringIO
        bytes_data = pkgutil.get_data('pyoxynet.data_test', csv_file)
        s = str(bytes_data, 'utf-8')
        data = StringIO(s)
        df = pd.read_csv(data)

    return df

def test_pyoxynet(input_df=[], n_inputs=7, past_points=40):
    """Test if the pyoxynet pipeline is running correclty

    Parameters: 
        n_inputs (int) : Number of inputs (deafult to Oxynet configuration)
        past_points (int) : Number of past points in the time series (deafult to Oxynet configuration)

    Returns:
        x (array) : Model output example

    """

    import numpy as np
    from uniplot import plot

    import json

    tfl_model = load_tf_model(n_inputs=n_inputs, past_points=past_points)

    if len(input_df) == 0:
        print('Using default py-oxynet data')
        df = load_csv_data()
    else:
        df = input_df

    # js = df1.to_json(orient='columns')

    # with open('test_data.json', 'w') as f:
    #    f.write(js)

    # Opening JSON file
    # with open('test_data.json') as json_file:
    #     data = json.load(json_file)

    # df = pd.DataFrame.from_dict(data)

    if n_inputs==7 and past_points==40:
        X = df[['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I']]
    if n_inputs==5 and past_points==40:
        X = df[['VO2_I', 'VE_I', 'PetO2_I', 'RF_I', 'VEVO2_I']]

    XN = normalize(X)

    # retrieve interpreter details
    input_details = tfl_model.get_input_details()
    output_details = tfl_model.get_output_details()

    time_series_len = input_details[0]['shape'][1]
    p_1 = []
    p_2 = []
    p_3 = []
    time = []

    for i in np.arange(len(XN)-time_series_len):
        XN_array = np.asarray(XN[i:(i+time_series_len)])
        input_data = np.reshape(XN_array, input_details[0]['shape'])
        input_data = input_data.astype(np.float32)
        tfl_model.allocate_tensors()
        tfl_model.set_tensor(input_details[0]['index'], input_data)
        tfl_model.invoke()
        output_data = tfl_model.get_tensor(output_details[0]['index'])
        p_1.append(output_data[0][0])
        p_2.append(output_data[0][1])
        p_3.append(output_data[0][2])
        time.append(df.time[i])

    import pandas as pd
    df = pd.DataFrame()
    df['time'] = time
    df['p_md'] = p_1
    df['p_hv'] = p_2
    df['p_sv'] = p_3

    plot([p_1, p_2, p_3],
         title="Exercise intensity domains",
         width=120,
         color=True,
         legend_labels=['1', '2', '3'])

    return df

def create_probabilities(duration=600, VT1=320, VT2=460):
    import numpy as np

    time_length = np.arange(1, duration + 1)
    pm_tanh = -1 * np.tanh(0.005 * np.arange(-duration / 2 + (duration / 2 - VT1), duration / 2 + (duration / 2 - VT1)))
    ps_tanh = 1 * np.tanh(0.0075 * np.arange(-duration / 2 + (duration / 2 - VT2), duration / 2 + (duration / 2 - VT2)))
    ph_tanh = 2 * np.exp(-((np.arange(0, duration)-(VT1 + (VT2-VT1)/2))**2)/(1.5*(((VT2-VT1)/2)**2)))-1

    p_mF = pm_tanh + np.flip(random_walk(length=len(time_length), scale_factor=200, variation=0.25))
    p_sF = ps_tanh + random_walk(length=len(time_length), scale_factor=200, variation=0.25)
    p_hF = ph_tanh + random_walk(length=len(time_length), scale_factor=200, variation=0.25)

    return p_mF, p_hF, p_sF

def random_walk(length=1, scale_factor=1, variation=1):
    from random import seed
    from random import random

    random_walk = list()
    random_walk.append(-variation if random() < 0.5 else variation)
    for i in range(1, length):
        movement = -variation if random() < 0.5 else variation
        value = random_walk[i - 1] + movement
        random_walk.append(value)

    return [i/scale_factor for i in random_walk]

def generate_CPET(generator, plot=False, fitness_group=None):

    import random
    import numpy as np
    import pandas as pd
    from uniplot import plot as terminal_plot

    import pkgutil
    from io import StringIO
    bytes_data = pkgutil.get_data('pyoxynet.data_test', 'database_statistics.csv')
    s = str(bytes_data, 'utf-8')
    data = StringIO(s)
    db_df = pd.read_csv(data)

    # extract sample from db
    if fitness_group == None:
        db_df_sample = db_df.sample()
    else:
        db_df_sample = db_df[db_df['fitness_group'] == fitness_group].sample()

    duration = int(db_df_sample.duration)
    VT1 = int(db_df_sample.VT1)
    VT2 = int(db_df_sample.VT2)

    VO2_peak = int(db_df_sample.VO2peak)
    VCO2_peak = int(db_df_sample.VCO2peak)
    VE_peak = int(db_df_sample.VEpeak)
    RF_peak = int(db_df_sample.RFpeak)
    PetO2_peak = int(db_df_sample.PetO2peak)
    PetCO2_peak = int(db_df_sample.PetCO2peak)
    HR_peak = int(db_df_sample.HRpeak)

    VO2_min = int(db_df_sample.VO2min)
    VCO2_min = int(db_df_sample.VCO2min)
    VE_min = int(db_df_sample.VEmin)
    RF_min = int(db_df_sample.RFmin)
    PetO2_min = int(db_df_sample.PetO2min)
    PetCO2_min = int(db_df_sample.PetCO2min)
    HR_min = int(db_df_sample.HRmin)

    # probability definition
    p_mF, p_hF, p_sF = create_probabilities(duration=duration, VT1=VT1, VT2=VT2)

    # Allocate tensors.
    generator.allocate_tensors()

    # Get input and output tensors.
    input_details = generator.get_input_details()
    output_details = generator.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    # initialise
    VO2 = []
    VCO2 = []
    VE = []
    HR = []
    RF = []
    PetO2 = []
    PetCO2 = []

    for steps_, seconds_ in enumerate(np.arange(0, duration)):
        input_data[0, -3:] = np.array([[p_hF[seconds_], p_sF[seconds_], p_mF[seconds_]]])
        generator.set_tensor(input_details[0]['index'], input_data)
        generator.invoke()
        output_data = generator.get_tensor(output_details[0]['index'])
        VO2.append(output_data[0, -1, 0])
        VCO2.append(output_data[0, -1, 1])
        VE.append(output_data[0, -1, 2])
        HR.append(output_data[0, -1, 3])
        RF.append(output_data[0, -1, 4])
        PetO2.append(output_data[0, -1, 5])
        PetCO2.append(output_data[0, -1, 6])
        # filter_vars = ['VO2_I', 'VCO2_I', 'VE_I', 'HR_I', 'RF_I', 'PetO2_I', 'PetCO2_I']

    df = pd.DataFrame()
    df['time'] = np.arange(0, duration)
    df['VO2_I'] = (np.asarray(VO2) - np.min(VO2))/(np.max((np.asarray(VO2) - np.min(VO2)))) * (VO2_peak - VO2_min) + VO2_min + random_walk(length=duration, scale_factor=2, variation=10)
    df['VCO2_I'] = (np.asarray(VCO2) - np.min(VCO2))/(np.max((np.asarray(VCO2) - np.min(VCO2)))) * (VCO2_peak - VCO2_min) + VCO2_min + random_walk(length=duration, scale_factor=2, variation=10)
    df['VE_I'] = (np.asarray(VE) - np.min(VE))/(np.max((np.asarray(VE) - np.min(VE)))) * (VE_peak - VE_min) + VE_min + random_walk(length=duration, scale_factor=2, variation=1)
    df['HR_I'] = (np.asarray(HR) - np.min(HR))/(np.max((np.asarray(HR) - np.min(HR)))) * (HR_peak - HR_min) + HR_min + random_walk(length=duration, scale_factor=2, variation=0.5)
    df['RF_I'] = (np.asarray(RF) - np.min(RF))/(np.max((np.asarray(RF) - np.min(RF)))) * (RF_peak - RF_min) + RF_min + random_walk(length=duration, scale_factor=2, variation=1)
    df['PetO2_I'] = (np.asarray(PetO2) - np.min(PetO2))/(np.max((np.asarray(PetO2) - np.min(PetO2)))) * (PetO2_peak - PetO2_min) + PetO2_min + random_walk(length=duration, scale_factor=2, variation=1)
    df['PetCO2_I'] = (np.asarray(PetCO2) - np.min(PetCO2))/(np.max((np.asarray(PetCO2) - np.min(PetCO2)))) * (PetCO2_peak - PetCO2_min) + PetCO2_min - random_walk(length=duration, scale_factor=2, variation=1)
    df['VEVO2_I'] = df['VE_I']/df['VO2_I']
    df['VEVCO2_I'] = df['VE_I']/df['VCO2_I']
    df['domain'] = np.NaN
    df.loc[df['time'] < VT1, 'domain'] = -1
    df.loc[df['time'] >= VT2, 'domain'] = 1
    df.loc[(df['time'] < VT2) & (df['time'] >= VT1), 'domain'] = 0

    if plot:
        terminal_plot([df['VO2_I'], df['VCO2_I']],
                      title="CPET variables", width=120,
                      color=True, legend_labels=['VO2_I', 'VCO2_I'])
        terminal_plot([df['VE_I'], df['HR_I'], df['RF_I']],
                      title="CPET variables", width=120,
                      color=True, legend_labels=['VE', 'HR', 'RF'])
        terminal_plot([df['PetO2_I'], df['PetCO2_I']],
                      title="CPET variables", width=120,
                      color=True, legend_labels=['PetO2_I', 'PetCO2_I'])

    if db_df_sample.fitness_group.values == 1:
        fitness_group = 'LOW'
    if db_df_sample.fitness_group.values == 2:
        fitness_group = 'MEDIUM'
    if db_df_sample.fitness_group.values == 3:
        fitness_group = 'HIGH'
    if db_df_sample.gender.values == -1:
        gender = 'MALE'
    if db_df_sample.gender.values == 1:
        gender = 'FEMALE'

    print('Data generated for a ', gender, ' individual with ', fitness_group, ' fitness capacity.')
    print('Weight: ', int(db_df_sample.weight.values), ' kg')
    print('Height: ', db_df_sample.height.values[0], 'm')
    print('Age: ', int(db_df_sample.Age.values), 'y')

    data = [{'Age': str(int(db_df_sample.Age.values)),
            'Height': str(db_df_sample.height.values[0]),
            'Weight': str(int(db_df_sample.weight.values)),
            'Gender': gender,
            'Aerobic_fitness_level': fitness_group},
            'VT1': str(VT1),
            'VT2': str(VT2)]

    return df, data