def PrintHello(hello='hello'):
    """This function prints to screen.

    Args:
       name (str):  The name to use.

    Returns:
       none

    """
    
    print(hello)
    
    return

def get_sec(time_str):
    """This function converts strings to time.

    Args:
       time_str (str):  The time str.

    Returns:
       total seconds (int)

    """

    my_time_str = time_str.replace('.', ':')
    try:
        h, m, s = my_time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    except:
        try:
            h, m, s, ms = my_time_str.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)
        except:
            m, s = my_time_str.split(':')
            return int(m) * 60 + int(s)

def normalize(df, min_target=-1, max_target=1):
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
        # TODO: pelase check here the normalisation between 0 and 1 or -1 and 1
        # result[feature_name] = 2 * (df[feature_name] - min_value) / (max_value - min_value) - 1
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    result = result.fillna(0)

    return result

def load_explainer(tf_model):
    """Loads a deep explainer from the shap library and automatically loads the data to train it

    Parameters:
        tf_model (Model class) : deep learning model that deep explainer needs to explain

    Returns:
        explainer (deep) : deep explainer SHAP

    """

    import shap
    from pyoxynet import data_test
    import importlib_resources
    import pickle
    from keras.layers import Input
    from keras.models import Model as mdl
    from .model import Model

    train_data_X_binaries = importlib_resources.read_binary(data_test, 'train_data_X.pickle')
    open('/tmp/train_data_X.pickle', 'wb').write(train_data_X_binaries)
    with open('/tmp/train_data_X.pickle', 'rb') as f:
        training_data = pickle.load(f)

    shap.explainers._deep.deep_tf.op_handlers['AddV2'] = shap.explainers._deep.deep_tf.passthrough
    model = Model(n_classes=3, n_input=6)
    newInput = Input(batch_shape=(1, 40, 6))
    newOutputs = model(newInput)
    newModel = mdl(newInput, newOutputs)
    newModel.set_weights(tf_model.get_weights())
    # explainer = shap.DeepExplainer(newModel, training_data)
    # Gradient explainer faster and perhaps more robust when it comes to TF2 applications
    explainer = shap.GradientExplainer(newModel, training_data)

    return explainer

def compute_shap(explainer, df, n_inputs=6, past_points=40, shap_stride=20):
    """Computes SHAP values

    Parameters:
        explainer (explainer SHAP) : deep explainer that explains the deep learning model
        data (pd df) : data for the local explanation

    Returns:
        shap (dict) : SHAP values

    """

    import shap
    import pandas as pd
    import numpy as np

    # some adjustments to input df
    # TODO: create dedicated function for this (duplicated taks)
    df = df.drop_duplicates('time')
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('timestamp')
    df = df.resample('1S').mean()
    df = df.interpolate()
    df['VO2_20s'] = df.VO2_I.rolling(20, win_type='triang', center=True).mean().fillna(method='bfill').fillna(
        method='ffill')
    df = df.reset_index()
    df = df.drop('timestamp', axis=1)

    if n_inputs == 6 and past_points == 40:
        if 'VCO2VO2_I' not in df.columns:
            df['VCO2VO2_I'] = df['VCO2_I'].values/df['VO2_I'].values
        filter_vars = ['VCO2VO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I']
        X = df[filter_vars]
        XN = normalize(X)
        XN = XN.filter(filter_vars, axis=1)
    if n_inputs == 5 and past_points == 40:
        filter_vars = ['VO2_I', 'VE_I', 'PetO2_I', 'RF_I', 'VEVO2_I']
        X = df[filter_vars]
        XN = normalize(X)
        XN = XN.filter(filter_vars, axis=1)

    data_X = np.zeros([0, past_points, n_inputs])
    for i in np.arange(1, XN.shape[0] - past_points, shap_stride):
        tmp_x = XN[i: i + past_points]
        tmp_x = np.expand_dims(tmp_x, 0)
        data_X = np.vstack((data_X, tmp_x))

    print('Computing SHAP values in progress')
    shap_values = explainer.shap_values(data_X)
    print('Done computing SHAP values!')

    tmp_hv = pd.DataFrame(np.mean(shap_values[0][:, :, :], axis=1), columns=['hv_shap_v' + str(i) for i in np.arange(0, 6)])
    tmp_sv = pd.DataFrame(np.mean(shap_values[1][:, :, :], axis=1),
                          columns=['sv_shap_v' + str(i) for i in np.arange(0, 6)])
    tmp_md = pd.DataFrame(np.mean(shap_values[2][:, :, :], axis=1),
                          columns=['md_shap_v' + str(i) for i in np.arange(0, 6)])

    df_tmp = pd.concat([tmp_md, tmp_hv, tmp_sv], axis=1)

    return shap_values, df_tmp

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

def load_tf_model(n_inputs=6, past_points=40, model='CNN'):
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
    from pyoxynet import regressor
    import tensorflow as tf
    import os

    if n_inputs == 6 and past_points == 40:
        if model == 'CNN':
            # load the classic Oxynet model configuration
            print('Classic Oxynet configuration model uploaded')
            saved_model_binaries = importlib_resources.read_binary(regressor, 'saved_model.pb')
            keras_metadata_model_binaries = importlib_resources.read_binary(regressor, 'keras_metadata.pb')
            variables_data_binaries = importlib_resources.read_binary(regressor, 'variables.data-00000-of-00001')
            variables_index_binaries = importlib_resources.read_binary(regressor, 'variables.index')
        if model == 'transformer':
            # load the classic Oxynet model configuration
            print('Classic Oxynet configuration model uploaded')
            tfl_model_binaries = importlib_resources.read_binary(regressor, 'transformer.pickle')
    if n_inputs == 5 and past_points == 40:
        # load the 5 input model configuration (e.g. in this case when on CO2 info is included)
        print('Specific configuration model uploaded (no VCO2 available)')
        tfl_model_binaries = importlib_resources.read_binary(regressor, 'tfl_model_5_40.pickle')

    try:
        if not os.path.isdir('/tmp/variables'):
            os.mkdir('/tmp/variables')
        open('/tmp/saved_model.pb', 'wb').write(saved_model_binaries)
        open('/tmp/keras_metadata.pb', 'wb').write(keras_metadata_model_binaries)
        open('/tmp/variables/variables.data-00000-of-00001', 'wb').write(variables_data_binaries)
        open('/tmp/variables/variables.index', 'wb').write(variables_index_binaries)
        model = tf.keras.models.load_model('/tmp/')
        from .model import Model
        my_model = Model(n_classes=3, n_input=6)
        my_model.build(input_shape=(1, 40, 6))
        my_model.set_weights(model.get_weights())

        return my_model
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
    from pyoxynet import generator
    import tensorflow as tf
    import os

    # load the classic Oxynet model configuration
    print('Classic Oxynet configuration model uploaded')
    saved_model_binaries = importlib_resources.read_binary(generator, 'saved_model.pb')
    keras_metadata_model_binaries = importlib_resources.read_binary(generator, 'keras_metadata.pb')
    variables_data_binaries = importlib_resources.read_binary(generator, 'variables.data-00000-of-00001')
    variables_index_binaries = importlib_resources.read_binary(generator, 'variables.index')

    try:
        if not os.path.isdir('/tmp/variables'):
            os.mkdir('/tmp/variables')
        open('/tmp/saved_model.pb', 'wb').write(saved_model_binaries)
        open('/tmp/keras_metadata.pb', 'wb').write(keras_metadata_model_binaries)
        open('/tmp/variables/variables.data-00000-of-00001', 'wb').write(variables_data_binaries)
        open('/tmp/variables/variables.index', 'wb').write(variables_index_binaries)
        model = tf.keras.models.load_model('/tmp/')
        from .model import Generator
        my_model = Generator()
        # TODO: this is hardcoded
        my_model.build(input_shape=(1, 53))
        my_model.set_weights(model.get_weights())

        return my_model
    except:
        print('Could not find a model that could satisfy the input size required')
        return None

def pip_install_tflite():
    """Makes sure TFLite is installed by executing a pip install command from the command line (sub-optimal solution)

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

def draw_real_test(resting='random'):
    """Draw a single data file from the directory containing all the real files

    Parameters:
        none

    Returns:
        df (pandas df) : Real data output

    """

    from importlib import resources
    import pandas as pd
    import pyoxynet.data_test
    import pkgutil
    from io import StringIO
    import random
    import numpy as np
    from datetime import datetime

    if resting == 'random':
        resting = random.randint(0, 1)

    if resting:
        file_index = str(random.randrange(1, 50))
        file_name = 'resting_real_test_' + file_index + '.csv'
        bytes_data = pkgutil.get_data('pyoxynet.data_test', file_name)
        print(file_name, ' from resting')
    else:
        file_index = str(random.randrange(1, 88))
        file_name = 'ramp_real_test_' + file_index + '.csv'
        bytes_data = pkgutil.get_data('pyoxynet.data_test', file_name)
        print(file_name, ' from ramp')

    s = str(bytes_data, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)

    if int(np.mean(df.fitness_group.values)) == 1:
        fitness_group = 'LOW'
    if int(np.mean(df.fitness_group.values)) == 2:
        fitness_group = 'MEDIUM'
    if int(np.mean(df.fitness_group.values)) == 3:
        fitness_group = 'HIGH'
    if int(np.mean(df.gender.values)) == -1:
        gender = 'MALE'
    if int(np.mean(df.gender.values)) == 1:
        gender = 'FEMALE'

    VT1 = df[np.diff(df.domain, prepend=-1) == 1].time.iloc[0]
    VT2 = df[np.diff(df.domain, prepend=-1) == 1].time.iloc[1]

    duration = len(df)

    df['VO2_I'] = df['VO2_I'] + [random.uniform(-100, 100) for i in np.arange(duration)]
    df['VCO2_I'] = df['VCO2_I'] + [random.uniform(-100, 100) for i in np.arange(duration)]
    df['VE_I'] = df['VE_I'] + [random.uniform(-2, 2) for i in np.arange(duration)]
    df['HR_I'] = df['HR_I'] + [random.uniform(-1, 1) for i in np.arange(duration)]
    df['RF_I'] = df['RF_I'] + [random.uniform(-2, 2) for i in np.arange(duration)]
    df['PetO2_I'] = df['PetO2_I'] + [random.uniform(-1, 1) for i in np.arange(duration)]
    df['PetCO2_I'] = df['PetCO2_I'] + [random.uniform(-1, 1) for i in np.arange(duration)]

    df['VEVO2_I'] = df['VE_I']/df['VO2_I']
    df['VEVCO2_I'] = df['VE_I']/df['VCO2_I']

    # Collect VO2 value at VT1 and VT2
    VO2VT1 = df.iloc[(df[df['domain'].diff().fillna(0) == 1].index[0] - 10):(df[df['domain'].diff().fillna(0) == 1].index[0] + 10)]['VO2_I'].mean()
    VO2VT2 = df.iloc[(df[df['domain'].diff().fillna(0) == 1].index[1] - 10):(
            df[df['domain'].diff().fillna(0) == 1].index[1] + 10)]['VO2_I'].mean()

    VO2_peak = df.VO2_I.rolling(20).mean().max()

    print('Data loaded for a ', gender, ' individual with ', fitness_group, ' fitness capacity.')
    print('Weight: ', int(np.mean(df.weight.values)), ' kg')
    print('Height: ', np.mean(df.height.values[0]), 'm')
    print('Age: ', int(np.mean(df.age.values)), 'y')
    print('VT1: ', str(VT1))
    print('VT2: ', str(VT2))
    print('VO2VT2: ', str(int(VO2VT1)))
    print('VO2VT2: ', str(int(VO2VT2)))

    data = dict()
    data['Age'] = str(int(df.age.values[0]))
    data['Height'] = str(df.height.values[0])
    data['Weight'] = str(int(df.weight.values[0]))
    data['Gender'] = gender
    data['Aerobic_fitness_level'] = fitness_group
    data['VT1'] = str(VT1)
    data['VT2'] = str(VT2)
    data['VO2VT1'] = str(int(VO2VT1))
    data['VO2VT2'] = str(int(VO2VT2))
    data['VO2max'] = str(int(VO2_peak))
    data['LT'] = str(int(VO2VT1))
    data['LT_vo2max'] = str(int((VO2VT1/VO2_peak)*100)) + '%'
    data['RCP'] = str(int(VO2VT2))
    data['RCP_vo2max'] = str(int((VO2VT2/VO2_peak) * 100)) + '%'
    data['id'] = 'real_#'
    data['noise_factor'] = 'NA'
    data['created'] = datetime.today().strftime("%m/%d/%Y - %H:%M:%S")
    data['resting'] = str(resting)

    df['breaths'] = (1/(df.RF_I/60))
    # df['recorded_timestamp'] = pd.to_datetime(df.time, unit='s')

    time_cum_sum = 0
    df_breath = pd.DataFrame()
    n = 0
    while time_cum_sum < (duration-20):
        try:
            tmp = df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + df.breaths.iloc[n]))].mean()
            df_breath = pd.concat([df_breath, pd.DataFrame(data=np.reshape(tmp.values, [1, len(df.columns)]),
                                                           columns=tmp.index.to_list())])
            time_cum_sum = (time_cum_sum + df.breaths.iloc[n])
            n = n + len(df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + df.breaths.iloc[n]))])
        except:
            break

    df_breath = df_breath.dropna()
    df_breath['time'] = df_breath['time'].astype(int)
    df_breath = df_breath.drop_duplicates('time')

    exercise_threshold_names = {"time": "t",
                                "VO2_I": "VO2",
                                "VCO2_I": "VCO2",
                                "VE_I": "VE",
                                "VCO2VO2_I": "R",
                                "VEVO2_I": "VE/VO2",
                                "VEVCO2_I": "VE/VCO2",
                                "PetO2_I": "PetO2",
                                "PetCO2_I": "PetCO2",
                                "HR_I": "HR",
                                "RF_I": "RF"}

    df_breath = df_breath.rename(columns=exercise_threshold_names)
    # create the dict for the exercise threshold app
    data['data'] = dict()
    data['data'] = df_breath.to_dict(orient='records')

    return df, data

def load_exercise_threshold_app_data(data_dict={}):
    """Loads data from data dict with format provided by https://www.exercisethresholds.com/

    Parameters:
        data_dict (dict) : Dictionary with format like test/exercise_threshold_app_test.json

    Returns:
        df (pandas df) : Pandas data frame with format that can be used by Pyoxynet for inference (columns needed: 'VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I')

    """

    import json
    import pandas as pd
    import numpy as np

    time = []
    VO2_I = []
    VCO2_I = []
    VE_I = []
    PetO2_I = []
    PetCO2_I = []
    VEVO2_I = []
    VEVCO2_I = []

    for data_points_ in data_dict[0]['data']:
        time.append(data_points_['t'])
        VO2_I.append(data_points_['VO2'])
        VCO2_I.append(data_points_['VCO2'])
        VE_I.append(data_points_['VE'])
        PetO2_I.append(data_points_['PetO2'])
        PetCO2_I.append(data_points_['PetCO2'])
        VEVO2_I.append(data_points_['VE/VO2'])
        VEVCO2_I.append(data_points_['VE/VCO2'])

    # transform in array
    time = np.asarray(time)
    VO2_I = np.asarray(VO2_I)
    VCO2_I = np.asarray(VCO2_I)
    VE_I = np.asarray(VE_I)
    PetO2_I = np.asarray(PetO2_I)
    PetCO2_I = np.asarray(PetCO2_I)
    VEVO2_I = np.asarray(VEVO2_I)
    VEVCO2_I = np.asarray(VEVCO2_I)

    # filter and interpolate
    time_I = np.arange(int(time[0]), int(time[-1]))
    VO2_I = np.interp(time_I, time, VO2_I)
    VCO2_I = np.interp(time_I, time, VCO2_I)
    VE_I = np.interp(time_I, time, VE_I)
    PetO2_I = np.interp(time_I, time, PetO2_I)
    PetCO2_I = np.interp(time_I, time, PetCO2_I)
    VEVO2_I = np.interp(time_I, time, VEVO2_I)
    VEVCO2_I = np.interp(time_I, time, VEVCO2_I)

    df = pd.DataFrame()
    df['time'] = time_I
    df['VO2_I'] = VO2_I
    df['VCO2_I'] = VCO2_I
    df['VE_I'] = VE_I
    df['PetO2_I'] = PetO2_I
    df['PetCO2_I'] = PetCO2_I
    df['VEVO2_I'] = VEVO2_I
    df['VEVCO2_I'] = VEVCO2_I

    return df

def test_pyoxynet(input_df=[], n_inputs=6, past_points=40, model='CNN', plot=True):
    """Runs the pyoxynet inference

    Parameters: 
        n_inputs (int) : Number of inputs (deafult to Oxynet configuration)
        past_points (int) : Number of past points in the time series (deafult to Oxynet configuration)

    Returns:
        x (array) : Model output example

    """

    import numpy as np
    from uniplot import plot
    import pandas as pd
    from scipy import stats
    import json

    tf_model = load_tf_model(n_inputs=n_inputs, past_points=past_points, model=model)

    if len(input_df) == 0:
        print('Using default pyoxynet data')
        df = load_csv_data()
    else:
        df = input_df

    # some adjustments to input df
    # TODO: create dedicated function for this
    df = df.drop_duplicates('time')
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('timestamp')
    df = df.resample('1S').mean()
    df = df.interpolate()
    df['VO2_20s'] = df.VO2_I.rolling(20, win_type='triang', center=True).mean().fillna(method='bfill').fillna(
        method='ffill')
    df = df.reset_index()
    df = df.drop('timestamp', axis=1)

    if n_inputs == 6 and past_points == 40:
        # filter_vars = ['VO2_I', 'VCO2_I', 'VE_I', 'HR_I', 'RF_I', 'PetO2_I', 'PetCO2_I']
        if 'VCO2VO2_I' not in df.columns:
            df['VCO2VO2_I'] = df['VCO2_I'].values/df['VO2_I'].values
        filter_vars = ['VCO2VO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I']
        X = df[filter_vars]
        XN = normalize(X)
        XN = XN.filter(filter_vars, axis=1)
    if n_inputs == 5 and past_points == 40:
        filter_vars = ['VO2_I', 'VE_I', 'PetO2_I', 'RF_I', 'VEVO2_I']
        X = df[filter_vars]
        XN = normalize(X)
        XN = XN.filter(filter_vars, axis=1)

    p_1 = []
    p_2 = []
    p_3 = []
    time = []

    for i in np.arange(int(past_points/2), len(XN) - int(past_points/2)):
        XN_array = np.asarray(XN[(i-int(past_points/2)):(i+int(past_points/2))])
        output_data = tf_model(XN_array.reshape(1, 40, 6))
        p_1.append(output_data.numpy()[0][0])
        p_2.append(output_data.numpy()[0][1])
        p_3.append(output_data.numpy()[0][2])
        time.append(df.time[i])

    tmp_df = pd.DataFrame()
    tmp_df['time'] = time
    tmp_df['p_md'] = optimal_filter(np.asarray(time), np.asarray(p_1), 100)
    tmp_df['p_hv'] = optimal_filter(np.asarray(time), np.asarray(p_2), 100)
    tmp_df['p_sv'] = optimal_filter(np.asarray(time), np.asarray(p_3), 100)

    mod_col = tmp_df[['p_md', 'p_hv', 'p_sv']].iloc[:20].mean().idxmax()
    sev_col = tmp_df[['p_md', 'p_hv', 'p_sv']].iloc[-20:].mean().idxmax()
    for labels_ in ['p_md', 'p_hv', 'p_sv']:
        if labels_ not in [mod_col, sev_col]:
            hv_col = labels_

    out_df = pd.DataFrame()
    out_df['time'] = time
    out_df['p_md'] = tmp_df[mod_col]
    out_df['p_hv'] = tmp_df[hv_col]
    out_df['p_sv'] = tmp_df[sev_col]

    if plot == True:
        plot([out_df['p_md'], out_df['p_hv'], out_df['p_sv']],
             title="Exercise intensity domains",
             width=120,
             color=True,
             legend_labels=['1', '2', '3'])

    out_dict = {}
    out_dict['VT1'] = {}
    out_dict['VT2'] = {}
    out_dict['VT1']['time'] = {}
    out_dict['VT2']['time'] = {}

    # FIXME: hard coded
    VT1_index = int(out_df[(out_df['p_hv'] <= out_df['p_md'])].index[-1] + past_points/2)
    VT2_index = int(out_df[(out_df['p_hv'] >= out_df['p_sv']) & (out_df['p_hv'] > out_df['p_md'])].index[-1] + past_points/2)

    out_dict['VT1']['time'] = df.iloc[VT1_index]['time']
    out_dict['VT2']['time'] = df.iloc[VT2_index]['time']

    out_dict['VT1']['VO2'] = df.iloc[VT1_index]['VO2_20s']
    out_dict['VT2']['VO2'] = df.iloc[VT2_index]['VO2_20s']

    return out_df, out_dict

def create_probabilities(duration=600, VT1=320, VT2=460, training=True, normalization=True, resting = True):
    """Creates the probabilities of being in different intensity domains

    These probabilities are then sent to the CPET generator and they are used ot generate CPET vars that can replicate those probabilities

    Parameters:
        duration (int): Length of the test file
        VT1 (int): First ventilatory threshold, in time samples from the beginning of the test
        VT2 (int): Second ventilatory threshold, in time samples from the beginning of the test

    Returns:
        p_mF (np array): Probability of being in the moderate intensity zone (-1:1)
        p_hF (np array): Probability of being in the heavy intensity zone (-1:1)
        p_sF (np array): Probability of being in the severe intensity zone (-1:1)

    """

    import numpy as np
    from scipy.interpolate import interp1d

    t = np.arange(1, duration + 1)

    if resting == True:
        step_1 = 60
        smooth_lambda = [duration*2, duration*2, duration*2]
    else:
        # no resting is included
        step_1 = 1
        smooth_lambda = [duration*2, duration*2, duration*2]

    y_pm = [1, 1, 0.5, 0, 0, 0]
    y_ph = [0, 0, 0.5, 1, 0.5, 0]
    y_ps = [0, 0, 0, 0, 0.5, 1]
    # linear coefficients
    x_p = [0, step_1, VT1, ((VT2 - VT1) / 2 + VT1), VT2, duration]

    from scipy.interpolate import UnivariateSpline
    # build splines
    # moderate => 2 splines
    pm_L1 = UnivariateSpline([0, step_1, VT1], [1, 1, 0.5], k=2)
    # compute additional points
    pm_L2 = UnivariateSpline([VT1, VT1 + 30, duration - 60, duration - 30, duration], [0.5, pm_L1(VT1 + 30), 0, 0, 0], k=2)
    p_mF = np.hstack((pm_L1(t[t < VT1]), pm_L2(t[t >= VT1])))
    # heavy => 3 splines
    ph_L1 = UnivariateSpline([0, step_1, VT1], [0, 0, 0.5], k=2)
    # compute additional points
    ph_L2 = UnivariateSpline([VT1, VT1 + 30, VT2], [0.5, ph_L1(VT1 + 30), 0.5], k=2)
    ph_L3 = UnivariateSpline([VT2, VT2 + 30, duration], [0.5, ph_L2(VT2 + 30), 0], k=2)
    p_hF = np.hstack((ph_L1(t[t < VT1]), ph_L2(t[(t >= VT1) & (t < VT2)]), ph_L3(t[t >= VT2])))
    # severe => 2 splines
    ps_L1 = UnivariateSpline([0, step_1, VT2], [0, 0, 0.5], k=2)
    # compute additional points
    ps_L2 = UnivariateSpline([VT2, VT2 + 15, VT2 + 30, duration - 30, duration],
                             [0.5, ps_L1(VT2 + 15), ps_L1(VT2 + 30), 1, 1], k=1)
    p_sF = np.hstack((ps_L1(t[t < VT2]), ps_L2(t[t >= VT2])))

    p_mF = optimal_filter(t, p_mF, 100)
    p_hF = optimal_filter(t, p_hF, 50)
    p_sF = optimal_filter(t, p_sF, 100)

    if training:
        p_mF = p_mF + np.random.randn(len(t)) / 18
        p_hF = p_hF + np.random.randn(len(t)) / 18
        p_sF = p_sF + np.random.randn(len(t)) / 18
    else:
        pass

    if normalization:
        p_mF = np.interp(p_mF, (p_mF.min(), p_mF.max()), (0, 1))
        p_hF = np.interp(p_hF, (p_hF.min(), p_hF.max()), (0, 1))
        p_sF = np.interp(p_sF, (p_sF.min(), p_sF.max()), (0, 1))
    else:
        pass

    return p_mF, p_hF, p_sF

def random_walk(length=1, scale_factor=1, variation=1):
    """Random walk generator

    Parameters:
        length (int): Length of the output list
        scale_factor (float): Scale factor to be applied to the whole output
        variation (float): Local variation of the main signal with the random walk

    Returns:
        none

    """

    from random import seed
    from random import random

    random_walk = list()
    random_walk.append(-variation if random() < 0.5 else variation)
    for i in range(1, length):
        movement = -variation if random() < 0.5 else variation
        value = random_walk[i - 1] + movement
        random_walk.append(value)

    return [i/scale_factor for i in random_walk]

def generate_CPET(generator,
                  plot=False,
                  fitness_group=None,
                  VT1=None, VT2=None,
                  duration=None,
                  noise_factor=0,
                  resting=False,
                  training=True,
                  normalization=True):
    """Actually generates the CPET file

    Parameters:
        length (int): Length of the output list
        fitness_group (int): Fitness level: low (1), medium (2), high (3). Default to random.
        noise_factor (float): Noise factor for white noise. Default to random.

    Returns:
        df (pd df): Pandas dataframe with CPET data included and ready to be processed by the model (if needed)
        data (dict): Data relative to the generated CPET

    """

    import random
    import numpy as np
    import pandas as pd
    from uniplot import plot as terminal_plot
    from datetime import datetime

    import pkgutil
    from io import StringIO

    if resting:
        bytes_data = pkgutil.get_data('pyoxynet.data_test', 'database_statistics_resting.csv')
    else:
        bytes_data = pkgutil.get_data('pyoxynet.data_test', 'database_statistics_ramp.csv')
    s = str(bytes_data, 'utf-8')
    data = StringIO(s)
    db_df = pd.read_csv(data)

    # extract sample from db
    if fitness_group == None:
        # if fitness group is not user defined, then a sample is randomly taken
        db_df_sample = db_df.sample()
    else:
        db_df_sample = db_df[db_df['fitness_group'] == fitness_group].sample()

    if duration == None:
        duration = int(db_df_sample.duration)

    if VT1 == None and VT2 == None:
        VT1 = 0
        VT2 = 0
        # check if difference in threshold is < 10% of the entire duration
        while (duration - VT2) < 60 or (VT2 - VT1) > 240 or (VT2 - VT1) < 60 or (duration - VT2) > 240 or (VT2 - VT1)/duration < 0.2:
            VT1 = int(random.randrange(25, 65) * duration * 0.01)
            VT2 = int(random.randrange(70, 96) * duration * 0.01)

    VO2_peak = int(db_df_sample.VO2peak)
    VCO2_peak = int(db_df_sample.VCO2peak)
    VE_peak = int(db_df_sample.VEpeak)
    RF_peak = int(db_df_sample.RFpeak)
    HR_peak = int(db_df_sample.HRpeak)
    PetO2_peak = int(db_df_sample.PetO2peak)
    PetCO2_peak = int(db_df_sample.PetCO2peak)

    VO2_min = int(db_df_sample.VO2min)
    VCO2_min = int(db_df_sample.VCO2min)
    VE_min = int(db_df_sample.VEmin)
    RF_min = int(db_df_sample.RFmin)
    HR_min = int(db_df_sample.HRmin)
    PetO2_min = int(db_df_sample.PetO2min)
    PetCO2_min = int(db_df_sample.PetCO2min)

    # probability definition
    # FIXME: hard coded here
    p_mF, p_hF, p_sF = create_probabilities(duration=duration,
                                            VT1=VT1,
                                            VT2=VT2,
                                            training=training,
                                            resting=resting,
                                            normalization=normalization)

    # initialise
    VO2 = []
    VCO2 = []
    VE = []
    HR = []
    RF = []
    PetO2 = []
    PetCO2 = []

    time_array = np.arange(duration)
    # TODO: this is hard coded
    input_data = np.array(np.random.random_sample([1, 53]))

    for seconds_ in time_array:
        # keep the seed?
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data[0, -3:] = np.array([[p_hF[seconds_], p_sF[seconds_], p_mF[seconds_]]])
        output_data = generator(input_data)
        VO2.append(np.median(output_data[0, :, 0]))
        VCO2.append(np.median(output_data[0, :, 1]))
        VE.append(np.median(output_data[0, :, 2]))
        HR.append(np.median(output_data[0, :, 3]))
        RF.append(np.median(output_data[0, :, 4]))
        PetO2.append(np.median(output_data[0, :, 5]))
        PetCO2.append(np.median(output_data[0, :, 6]))
        # vars -> ['VO2_I', 'VCO2_I', 'VE_I', 'HR_I', 'RF_I', 'PetO2_I', 'PetCO2_I']

    # filter before you expand again between min and max
    VO2 = optimal_filter(time_array, VO2, 10)
    VCO2 = optimal_filter(time_array, VCO2, 10)
    VE = optimal_filter(time_array, VE, 10)
    HR = optimal_filter(time_array, HR, 10)
    RF = optimal_filter(time_array, RF, 10)
    PetO2 = optimal_filter(time_array, PetO2, 10)
    PetCO2 = optimal_filter(time_array, PetCO2, 10)

    min_norm = 0
    max_norm = 1
    VO2 = np.interp(np.asarray(VO2), (np.asarray(VO2).min(), np.asarray(VO2).max()), (min_norm, max_norm))
    VCO2 = np.interp(np.asarray(VCO2), (np.asarray(VCO2).min(), np.asarray(VCO2).max()), (min_norm, max_norm))
    VE = np.interp(np.asarray(VE), (np.asarray(VE).min(), np.asarray(VE).max()), (min_norm, max_norm))
    HR = np.interp(np.asarray(HR), (np.asarray(HR).min(), np.asarray(HR).max()), (min_norm, max_norm))
    RF = np.interp(np.asarray(RF), (np.asarray(RF).min(), np.asarray(RF).max()),
                   (min_norm, max_norm))
    PetO2 = np.interp(np.asarray(PetO2), (np.asarray(PetO2).min(), np.asarray(PetO2).max()), (min_norm, max_norm))
    PetCO2 = np.interp(np.asarray(PetCO2), (np.asarray(PetCO2).min(), np.asarray(PetCO2).max()),
                       (min_norm, max_norm))

    df = pd.DataFrame()
    df['time'] = time_array

    if noise_factor == None:
        noise_factor = random.randint(3, 5)/2
    else:
        pass

    # TODO: should I compute VEVO2 or VEVCO2 before adding the noise (?)
    df['VO2_I'] = (np.asarray(VO2) - np.min(VO2))/(np.max((np.asarray(VO2) - np.min(VO2)))) * (VO2_peak - VO2_min) + VO2_min + np.random.randn(len(VO2)) * 40 * noise_factor
    df['VCO2_I'] = (np.asarray(VCO2) - np.min(VCO2))/(np.max((np.asarray(VCO2) - np.min(VCO2)))) * (VCO2_peak - VCO2_min) + VCO2_min + np.random.randn(len(VO2)) * 40 * noise_factor
    df['VE_I'] = (np.asarray(VE) - np.min(VE))/(np.max((np.asarray(VE) - np.min(VE)))) * (VE_peak - VE_min) + VE_min + np.random.randn(len(VO2)) * 1.5 * noise_factor
    df['HR_I'] = np.ndarray.astype((np.asarray(HR) - np.min(HR)) / (np.max((np.asarray(HR) - np.min(HR)))) * (HR_peak - HR_min) + HR_min + np.random.randn(len(VO2)) * 1 * noise_factor, int)
    df['RF_I'] = (np.asarray(RF) - np.min(RF)) / (np.max((np.asarray(RF) - np.min(RF)))) * (
            RF_peak - RF_min) + RF_min + np.random.randn(len(VO2)) * 1 * noise_factor
    df['PetO2_I'] = (np.asarray(PetO2) - np.min(PetO2))/(np.max((np.asarray(PetO2) - np.min(PetO2)))) * (PetO2_peak - PetO2_min) + PetO2_min + np.random.randn(len(VO2)) * 1.5 * noise_factor
    df['PetCO2_I'] = (np.asarray(PetCO2) - np.min(PetCO2))/(np.max((np.asarray(PetCO2) - np.min(PetCO2)))) * (PetCO2_peak - PetCO2_min) + PetCO2_min + np.random.randn(len(VO2)) * 1.5 * noise_factor

    df['p_mF'] = p_mF
    df['p_hF'] = p_hF
    df['p_sF'] = p_sF

    df['VEVO2_I'] = df['VE_I']/df['VO2_I']
    df['VEVCO2_I'] = df['VE_I']/df['VCO2_I']

    df['VCO2VO2_I'] = df['VCO2_I'] / df['VO2_I']
    df['PetO2VO2_I'] = df['PetO2_I'] / df['VO2_I']
    df['PetCO2VO2_I'] = df['PetCO2_I'] / df['VO2_I']

    df['domain'] = np.NaN
    df.loc[df['time'] < (VT1), 'domain'] = -1
    df.loc[df['time'] >= (VT2), 'domain'] = 1
    df.loc[(df['time'] < (VT2)) & (df['time'] >= (VT1)), 'domain'] = 0
    df['fitness_group'] = db_df_sample['fitness_group'].values[0]
    df['Age'] = db_df_sample['Age'].values[0]
    df['age_group'] = db_df_sample['age_group'].values[0]
    df['gender'] = db_df_sample['gender'].values[0]
    df['weight'] = db_df_sample['weight'].values[0]
    df['height'] = db_df_sample['height'].values[0]

    # Collect VO2 value at VT1 and VT2
    VO2VT1 = df.iloc[(df[df['domain'].diff().fillna(0) == 1].index[0] - 10):(df[df['domain'].diff().fillna(0) == 1].index[0] + 10)]['VO2_I'].mean()
    VO2VT2 = df.iloc[(df[df['domain'].diff().fillna(0) == 1].index[1] - 10):(
            df[df['domain'].diff().fillna(0) == 1].index[1] + 10)]['VO2_I'].mean()

    if plot:
        terminal_plot([df['VO2_I'], df['VCO2_I']],
                      title="CPET variables", width=120,
                      color=True, legend_labels=['VO2_I', 'VCO2_I'])
        terminal_plot([df['VE_I']],
                      title="CPET variables", width=120,
                      color=True, legend_labels=['VE'])
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
    print('Noise factor: ', round(noise_factor, 2))
    print('VT1: ', str(VT1))
    print('VT2: ', str(VT2))
    print('VO2VT2: ', str(int(VO2VT1)))
    print('VO2VT2: ', str(int(VO2VT2)))
    print('Resting:', resting)

    # TODO: create a function to generate this dict, as it is the same that we should use when drawing real data files
    # create dict fro Exercise Threshold App
    data = dict()
    data['Age'] = str(int(db_df_sample.Age.values))
    data['Height'] = str(db_df_sample.height.values[0])
    data['Weight'] = str(int(db_df_sample.weight.values))
    data['Gender'] = gender
    data['Aerobic_fitness_level'] = fitness_group
    data['VT1'] = str(VT1)
    data['VT2'] = str(VT2)
    data['VO2VT1'] = str(int(VO2VT1))
    data['VO2VT2'] = str(int(VO2VT2))
    data['VO2max'] = str(int(VO2_peak))
    data['LT'] = str(int(VO2VT1))
    data['LT_vo2max'] = str(int((VO2VT1/VO2_peak)*100)) + '%'
    data['RCP'] = str(int(VO2VT2))
    data['RCP_vo2max'] = str(int((VO2VT2/VO2_peak) * 100)) + '%'
    data['id'] = 'fake_#'
    data['noise_factor'] = str(noise_factor)
    data['created'] = datetime.today().strftime("%m/%d/%Y - %H:%M:%S")
    data['resting'] = str(resting)

    df['breaths'] = (1/(df.RF_I/60))
    # df['recorded_timestamp'] = pd.to_datetime(df.time, unit='s')

    time_cum_sum = 0
    df_breath = pd.DataFrame()
    n = 0
    while time_cum_sum < (duration-6):
        try:
            tmp = df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + df.breaths.iloc[n]))].mean()
            # TODO: this 25 is hardcoded, you should have len(df.columns)
            df_breath = pd.concat([df_breath, pd.DataFrame(data=np.reshape(tmp.values, [1, 24]),
                                                           columns=tmp.index.to_list())])
            time_cum_sum = (time_cum_sum + df.breaths.iloc[n])
            n = n + len(df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + df.breaths.iloc[n]))])
        except:
            break

    df_breath = df_breath.dropna()
    df_breath['time'] = df_breath['time'].astype(int)
    df_breath = df_breath.drop_duplicates('time')

    exercise_threshold_names = {"time": "t",
                                "VO2_I": "VO2",
                                "VCO2_I": "VCO2",
                                "VE_I": "VE",
                                "VCO2VO2_I": "R",
                                "VEVO2_I": "VE/VO2",
                                "VEVCO2_I": "VE/VCO2",
                                "PetO2_I": "PetO2",
                                "PetCO2_I": "PetCO2",
                                "HR_I": "HR",
                                "RF_I": "RF"}

    df_breath = df_breath.rename(columns=exercise_threshold_names)
    # create the dict for the exercise threshold app
    data['data'] = dict()
    data['data'] = df_breath.to_dict(orient='records')

    return df, data