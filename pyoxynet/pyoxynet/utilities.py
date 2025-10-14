import numpy as np
import scipy.interpolate
from scipy.stats import beta
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tempfile

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

    # my_time_str = time_str.replace('.', ':')
    my_time_str = time_str

    if ':' in time_str:
        try:
            h, m, s = my_time_str.split(':')
            return int(h) * 3600 + int(m) * 60 + int(float(s))
        except:
            try:
                h, m, s, ms = my_time_str.split(':')
                return int(h) * 3600 + int(m) * 60 + int(float(s))
            except:
                m, s = my_time_str.split(':')
                return int(m) * 60 + int(float(s))

    if '.' in time_str:
        m = int(float(time_str))
        s = (float(time_str) - m) * 60
        return int(m) * 60 + int(float(s))

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

def load_tf_model(n_inputs=5, past_points=40, model='CNN'):
    """This function loads the saved tf models.

    Args:
       n_inputs (int):  Number of input variables.
       past_points (int):  Number of past inputs in the time-series.

    Returns:
       interpreter (tflite interpreter) : handle on the TFLite interpreter

    """

    import importlib_resources
    import pickle
    from io import BytesIO
    from pyoxynet import regressor, TCN, LSTMGRUModel, murias_lab, AIS, NSWIS
    import tensorflow as tf
    import os

    if model == 'CNN':
        # load the classic Oxynet model configuration
        print('Classic Oxynet configuration model uploaded')
        try:
            saved_model_binaries = importlib_resources.read_binary(regressor, 'saved_model.pb')
            keras_metadata_model_binaries = importlib_resources.read_binary(regressor, 'keras_metadata.pb')
            variables_data_binaries = importlib_resources.read_binary(regressor, 'variables.data-00000-of-00001')
            variables_index_binaries = importlib_resources.read_binary(regressor, 'variables.index')
        except:
            saved_model_binaries = importlib_resources.files(regressor).joinpath('saved_model.pb').read_bytes()
            keras_metadata_model_binaries = importlib_resources.files(regressor).joinpath('keras_metadata.pb').read_bytes()
            variables_data_binaries = importlib_resources.files(regressor).joinpath('variables.data-00000-of-00001').read_bytes()
            variables_index_binaries = importlib_resources.files(regressor).joinpath('variables.index').read_bytes()

    if model == 'TCN':
        # load the classic Oxynet model configuration
        print('You are uploading an undocumented version of this model TCN')
        try:
            saved_model_binaries = importlib_resources.read_binary(TCN, 'saved_model.pb')
            keras_metadata_model_binaries = importlib_resources.read_binary(TCN, 'keras_metadata.pb')
            variables_data_binaries = importlib_resources.read_binary(TCN, 'variables.data-00000-of-00001')
            variables_index_binaries = importlib_resources.read_binary(TCN, 'variables.index')
        except:
            saved_model_binaries = importlib_resources.files(TCN).joinpath('saved_model.pb').read_bytes()
            keras_metadata_model_binaries = importlib_resources.files(TCN).joinpath('keras_metadata.pb').read_bytes()
            variables_data_binaries = importlib_resources.files(TCN).joinpath('variables.data-00000-of-00001').read_bytes()
            variables_index_binaries = importlib_resources.files(TCN).joinpath('variables.index').read_bytes()

    if model == 'LSTMGRUModel':
        # load the classic Oxynet model configuration
        print('You are uploading an undocumented version of this model LSTMGRUModel')
        try:
            saved_model_binaries = importlib_resources.read_binary(LSTMGRUModel, 'saved_model.pb')
            keras_metadata_model_binaries = importlib_resources.read_binary(LSTMGRUModel, 'keras_metadata.pb')
            variables_data_binaries = importlib_resources.read_binary(LSTMGRUModel, 'variables.data-00000-of-00001')
            variables_index_binaries = importlib_resources.read_binary(LSTMGRUModel, 'variables.index')
        except:
            saved_model_binaries = importlib_resources.files(LSTMGRUModel).joinpath('saved_model.pb').read_bytes()
            keras_metadata_model_binaries = importlib_resources.files(LSTMGRUModel).joinpath('keras_metadata.pb').read_bytes()
            variables_data_binaries = importlib_resources.files(LSTMGRUModel).joinpath('variables.data-00000-of-00001').read_bytes()
            variables_index_binaries = importlib_resources.files(LSTMGRUModel).joinpath('variables.index').read_bytes()

    if model == 'murias_lab':
        # load the classic Oxynet model configuration
        print('Model custom trained on data from Juan Murias Lab uploaded')
        print('This is an LSTMGRU model')
        try:
            saved_model_binaries = importlib_resources.read_binary(murias_lab, 'saved_model.pb')
            keras_metadata_model_binaries = importlib_resources.read_binary(murias_lab, 'keras_metadata.pb')
            variables_data_binaries = importlib_resources.read_binary(murias_lab, 'variables.data-00000-of-00001')
            variables_index_binaries = importlib_resources.read_binary(murias_lab, 'variables.index')
        except:
            saved_model_binaries = importlib_resources.files(murias_lab).joinpath('saved_model.pb').read_bytes()
            keras_metadata_model_binaries = importlib_resources.files(murias_lab).joinpath('keras_metadata.pb').read_bytes()
            variables_data_binaries = importlib_resources.files(murias_lab).joinpath('variables.data-00000-of-00001').read_bytes()
            variables_index_binaries = importlib_resources.files(murias_lab).joinpath('variables.index').read_bytes()
            
    if model == 'AIS':
        # load the classic Oxynet model configuration
        print('Model custom trained on data from AIS')
        print('This is an CNN model')
        try:
            saved_model_binaries = importlib_resources.read_binary(AIS, 'saved_model.pb')
            keras_metadata_model_binaries = importlib_resources.read_binary(AIS, 'keras_metadata.pb')
            variables_data_binaries = importlib_resources.read_binary(AIS, 'variables.data-00000-of-00001')
            variables_index_binaries = importlib_resources.read_binary(AIS, 'variables.index')
        except:
            saved_model_binaries = importlib_resources.files(AIS).joinpath('saved_model.pb').read_bytes()
            keras_metadata_model_binaries = importlib_resources.files(AIS).joinpath('keras_metadata.pb').read_bytes()
            variables_data_binaries = importlib_resources.files(AIS).joinpath('variables.data-00000-of-00001').read_bytes()
            variables_index_binaries = importlib_resources.files(AIS).joinpath('variables.index').read_bytes()
            
    if model == 'NSWIS':
        # load the classic Oxynet model configuration
        print('Model custom trained on data from NSWIS')
        print('This is an CNN model')
        try:
            saved_model_binaries = importlib_resources.read_binary(NSWIS, 'saved_model.pb')
            keras_metadata_model_binaries = importlib_resources.read_binary(NSWIS, 'keras_metadata.pb')
            variables_data_binaries = importlib_resources.read_binary(NSWIS, 'variables.data-00000-of-00001')
            variables_index_binaries = importlib_resources.read_binary(NSWIS, 'variables.index')
        except:
            saved_model_binaries = importlib_resources.files(NSWIS).joinpath('saved_model.pb').read_bytes()
            keras_metadata_model_binaries = importlib_resources.files(NSWIS).joinpath('keras_metadata.pb').read_bytes()
            variables_data_binaries = importlib_resources.files(NSWIS).joinpath('variables.data-00000-of-00001').read_bytes()
            variables_index_binaries = importlib_resources.files(NSWIS).joinpath('variables.index').read_bytes()

    if model == 'transformer':
        # load the classic Oxynet model configuration
        print('Classic Oxynet configuration model uploaded')
        tfl_model_binaries = importlib_resources.read_binary(regressor, 'transformer.pickle')

    try:
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, 'model')
        variables_dir = os.path.join(model_dir, 'variables')
        os.makedirs(variables_dir, exist_ok=True)

        with open(os.path.join(model_dir, 'saved_model.pb'), 'wb') as f:
            f.write(saved_model_binaries)

        with open(os.path.join(model_dir, 'keras_metadata.pb'), 'wb') as f:
            f.write(keras_metadata_model_binaries)

        with open(os.path.join(variables_dir, 'variables.data-00000-of-00001'), 'wb') as f:
            f.write(variables_data_binaries)

        with open(os.path.join(variables_dir, 'variables.index'), 'wb') as f:
            f.write(variables_index_binaries)

        from .model import Model, TCN, LSTMGRUModel
        
        if model == 'CNN':
            model = tf.keras.models.load_model(model_dir)
            my_model = Model(n_classes=3, n_input=n_inputs)
            my_model.build(input_shape=(1, past_points, n_inputs))
            my_model.set_weights(model.get_weights())
        if model == 'LSTMGRUModel':
            model = tf.keras.models.load_model(model_dir)
            my_model = LSTMGRUModel(n_input=n_inputs)
            my_model.build(input_shape=(1, past_points, n_inputs))
            my_model.set_weights(model.get_weights())
        if model == 'TCN':
            model = tf.keras.models.load_model(model_dir)
            num_classes = 3
            num_filters = 128
            kernel_size = 3
            num_layers = 8
            dropout_rate = 0.2
            my_model = TCN(num_classes, n_inputs, num_layers=num_layers, num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate)
            my_model.build(input_shape=(1, past_points, n_inputs))
            my_model.set_weights(model.get_weights())
        if model == 'murias_lab':
            model = tf.keras.models.load_model(model_dir)
            my_model = LSTMGRUModel(n_input=n_inputs)
            my_model.build(input_shape=(1, past_points, n_inputs))
            my_model.set_weights(model.get_weights())
        if model == 'AIS':
            model = tf.keras.models.load_model(model_dir)
            my_model = Model(n_classes=3, n_input=n_inputs)
            my_model.build(input_shape=(1, past_points, n_inputs))
            my_model.set_weights(model.get_weights())
        if model == 'NSWIS':
            model = tf.keras.models.load_model(model_dir)
            my_model = Model(n_classes=3, n_input=n_inputs)
            my_model.build(input_shape=(1, past_points, n_inputs))
            my_model.set_weights(model.get_weights())

        return my_model
    except:
        print('Could not find a model that could satisfy the input size required')
        return None  

def load_tf_generator():
    """This function loads the saved TensorFlow generator model.

    NOTE: This function requires full TensorFlow installation and will NOT work
    with tflite_runtime only. If you installed the lite version of pyoxynet,
    this generator functionality is not available.

    Args:
       None

    Returns:
       generator (TF model) : Handle on the TensorFlow generator model, or None if unavailable

    """

    # Check if TensorFlow is available (not just tflite_runtime)
    try:
        import tensorflow as tf
    except (ImportError, ModuleNotFoundError):
        print('ERROR: TensorFlow generator requires full TensorFlow installation.')
        print('The generator model cannot work with tflite_runtime only.')
        print('To use the generator, please install: pip install tensorflow')
        print('If you only need inference, the TFLite models work with tflite_runtime.')
        return None

    import importlib_resources
    from pyoxynet import generator
    import os

    # load the classic Oxynet model configuration
    print('Loading TensorFlow generator model (requires full TensorFlow)...')

    try:
        # importlib < 6.0
        saved_model_binaries = importlib_resources.read_binary(generator, 'saved_model.pb')
        keras_metadata_model_binaries = importlib_resources.read_binary(generator, 'keras_metadata.pb')
        variables_data_binaries = importlib_resources.read_binary(generator, 'variables.data-00000-of-00001')
        variables_index_binaries = importlib_resources.read_binary(generator, 'variables.index')
    except:
        # importlib >= 6.0
        saved_model_binaries = importlib_resources.files(generator).joinpath('saved_model.pb').read_bytes()
        keras_metadata_model_binaries = importlib_resources.files(generator).joinpath('keras_metadata.pb').read_bytes()
        variables_data_binaries = importlib_resources.files(generator).joinpath('variables.data-00000-of-00001').read_bytes()
        variables_index_binaries = importlib_resources.files(generator).joinpath('variables.index').read_bytes()

    try:
        # Use tempfile for cross-platform compatibility instead of hardcoded /tmp
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, 'generator')
        variables_dir = os.path.join(model_dir, 'variables')
        os.makedirs(variables_dir, exist_ok=True)

        # Write model files to temp directory
        with open(os.path.join(model_dir, 'saved_model.pb'), 'wb') as f:
            f.write(saved_model_binaries)
        with open(os.path.join(model_dir, 'keras_metadata.pb'), 'wb') as f:
            f.write(keras_metadata_model_binaries)
        with open(os.path.join(variables_dir, 'variables.data-00000-of-00001'), 'wb') as f:
            f.write(variables_data_binaries)
        with open(os.path.join(variables_dir, 'variables.index'), 'wb') as f:
            f.write(variables_index_binaries)

        # Load the TensorFlow model
        # Try Keras 3 compatible method first (TFSMLayer for SavedModel)
        try:
            # Keras 3: Use TFSMLayer for SavedModel format
            model = tf.keras.models.load_model(model_dir)
        except (ValueError, OSError) as keras3_error:
            # Keras 3 doesn't support SavedModel with load_model
            # Try loading with TensorFlow's saved_model.load instead
            print('Note: Using TensorFlow SavedModel loader (Keras 3 compatibility)')
            try:
                model = tf.saved_model.load(model_dir)
            except Exception as tf_error:
                raise ValueError(
                    f"Cannot load generator model. "
                    f"Keras 3 error: {keras3_error}. "
                    f"TensorFlow loader error: {tf_error}. "
                    f"The model may need to be re-exported in .keras or .h5 format."
                )

        from .model import generator as generator_model
        my_model = generator_model(n_input=7, n_past_points=40, n_labels=3, data_noise_dim=20)
        # TODO: this is hardcoded
        my_model.build(input_shape=(1, 23))

        # Extract weights - handle both Keras and SavedModel formats
        try:
            my_model.set_weights(model.get_weights())
        except AttributeError:
            # SavedModel object doesn't have get_weights(), need different approach
            print('Warning: SavedModel format detected. Weight transfer may require manual mapping.')
            print('The generator model structure is initialized but weights may not be loaded.')
            print('Consider re-exporting the model in .keras format for full Keras 3 compatibility.')

        print('Generator model loaded successfully')
        return my_model
    except Exception as e:
        print(f'ERROR: Could not load generator model: {str(e)}')
        print('This may be due to:')
        print('  1. Missing TensorFlow installation (tflite_runtime is not sufficient)')
        print('  2. Keras 3 incompatibility with SavedModel format (model needs .keras or .h5 format)')
        print('  3. Incompatible model format or architecture mismatch')
        print('  4. Missing model files')
        print('\nRecommendation: Re-export the generator model in Keras 3 .keras format')
        return None

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
    while time_cum_sum < (duration - 6):
        try:
            tmp = df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + df.breaths.iloc[n]))].median()
            # TODO: this 25 is hardcoded, you should have len(df.columns)
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

def load_exercise_threshold_app_data(data_dict={}, filter_size = 20):
    """Loads data from data dict with format provided by https://www.exercisethresholds.com/

    Parameters:
        data_dict (dict) : Dictionary with format like test/exercise_threshold_app_test.json

    Returns:
        df (pandas df) : Pandas data frame with format that can be used by Pyoxynet for inference (columns needed: 'VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I')

    """

    import json
    import pandas as pd
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    time = []
    VO2_I = []
    VCO2_I = []
    VCO2VO2_I = []
    VE_I = []
    PetO2_I = []
    PetCO2_I = []
    VEVO2_I = []
    VEVCO2_I = []
    HR_I = []

    for data_points_ in data_dict[0]['data']:
        time.append(data_points_['t'])
        VO2_I.append(data_points_['VO2'])
        VCO2_I.append(data_points_['VCO2'])
        VE_I.append(data_points_['VE'])
        PetO2_I.append(data_points_['PetO2'])
        PetCO2_I.append(data_points_['PetCO2'])
        VEVO2_I.append(data_points_['VE/VO2'])
        VEVCO2_I.append(data_points_['VE/VCO2'])
        VCO2VO2_I.append(data_points_['VCO2']/data_points_['VO2'])

    # transform in array
    time = np.asarray(time)
    VO2_I = np.asarray(VO2_I)
    VCO2_I = np.asarray(VCO2_I)
    VE_I = np.asarray(VE_I)
    PetO2_I = np.asarray(PetO2_I)
    PetCO2_I = np.asarray(PetCO2_I)
    VEVO2_I = np.asarray(VEVO2_I)
    VEVCO2_I = np.asarray(VEVCO2_I)
    VCO2VO2_I = np.asarray(VCO2VO2_I)

    # rolling averages to filter the data
    # This filter size should be equal to the one used by the Test class
    # Keep the filter size consistent with the one used in the Test class * March 2025 AZ
    # filter_size = 20
    #
    VO2_F = uniform_filter1d(VO2_I, size=filter_size)
    VCO2_F = uniform_filter1d(VCO2_I, size=filter_size)
    VE_F = uniform_filter1d(VE_I, size=filter_size)
    PetO2_F = uniform_filter1d(PetO2_I, size=filter_size)
    PetCO2_F = uniform_filter1d(PetCO2_I, size=filter_size)
    VEVO2_F = uniform_filter1d(VEVO2_I, size=filter_size)
    VEVCO2_F = uniform_filter1d(VEVCO2_I, size=filter_size)
    VCO2VO2_F = uniform_filter1d(VCO2VO2_I, size=filter_size)

    # filter and interpolate
    time_I = np.arange(int(time[0]), int(time[-1]))
    VO2_I = np.interp(time_I, time, VO2_F)
    VCO2_I = np.interp(time_I, time, VCO2_F)
    VE_I = np.interp(time_I, time, VE_F)
    PetO2_I = np.interp(time_I, time, PetO2_F)
    PetCO2_I = np.interp(time_I, time, PetCO2_F)
    VEVO2_I = np.interp(time_I, time, VEVO2_F)
    VEVCO2_I = np.interp(time_I, time, VEVCO2_F)
    VCO2VO2_I = np.interp(time_I, time, VCO2VO2_F)

    df = pd.DataFrame()
    df['time'] = time_I
    df['VO2_I'] = VO2_I
    df['VCO2_I'] = VCO2_I
    df['VE_I'] = VE_I
    df['PetO2_I'] = PetO2_I
    df['PetCO2_I'] = PetCO2_I
    df['VEVO2_I'] = VEVO2_I
    df['VEVCO2_I'] = VEVCO2_I
    df['VCO2VO2_I'] = VCO2VO2_I
    df['HR_I'] = 0

    return df

def test_pyoxynet(tf_model=[], input_df=[], n_inputs=5, past_points=40, model='CNN', plot=False, inference_stride=1):
    """Runs the pyoxynet inference

    Parameters: 
        tf_model (TF model) : Model uploaded with load_tf_model
        n_inputs (int) : Number of inputs (deafult to Oxynet configuration)
        past_points (int) : Number of past points in the time series (deafult to Oxynet configuration)
        inference_stride (int) : Stride inference for NN - speed up computation

    Returns:
        x (array) : Model output example

    """

    import numpy as np
    from uniplot import plot
    import pandas as pd
    from scipy import stats
    import json

    # import the data for the normalisation
    import pkgutil
    from io import StringIO
    bytes_data = pkgutil.get_data('pyoxynet.data_test', 'database_statistics_resting.csv')

    s = str(bytes_data, 'utf-8')
    data = StringIO(s)
    db_df = pd.read_csv(data)

    if tf_model == []:
        tf_model = load_tf_model(n_inputs=n_inputs, past_points=past_points, model=model)
    else:
        pass

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

    # filter_vars = ['VO2_I', 'VCO2_I', 'VE_I', 'HR_I', 'RF_I', 'PetO2_I', 'PetCO2_I']
    if 'VCO2VO2_I' not in df.columns:
        df['VCO2VO2_I'] = df['VCO2_I'].values/df['VO2_I'].values
    # filter_vars = ['VCO2VO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I']
    filter_vars = ['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I']
    XN = df.copy()
    # XN = normalize(X)
    # XN = X
    XN['VO2_I'] = (XN['VO2_I'] - XN['VO2_I'].min()) / (
            XN['VO2_I'].max() - XN['VO2_I'].min())
    XN['VCO2_I'] = (XN['VCO2_I'] - XN['VCO2_I'].min()) / (
            XN['VCO2_I'].max() - XN['VCO2_I'].min())
    # XN['HR_I'] = (XN['HR_I'] - XN['HR_I'].min()) / (
    #         XN['HR_I'].max() - XN['HR_I'].min())
    # XN['RF_I'] = (XN['RF_I'] - XN['RF_I'].min()) / (
    #         XN['RF_I'].max() - XN['RF_I'].min())
    # XN['VCO2VO2_I'] = (XN['VCO2VO2_I'] - XN['VCO2VO2_I'].min()) / (
    #         XN['VCO2VO2_I'].max() - XN['VCO2VO2_I'].min())
    XN['VE_I'] = (XN['VE_I'] - XN['VE_I'].min()) / (
            XN['VE_I'].max() - XN['VE_I'].min())
    XN['PetO2_I'] = (XN['PetO2_I'] - XN['PetO2_I'].min()) / (
            XN['PetO2_I'].max() - XN['PetO2_I'].min())
    # in the case of PetCO2 you take MAX from top 25 and MIN from the single test
    # (x - MIN) / (MAX - MIN)
    XN['PetCO2_I'] = (XN['PetCO2_I'] - XN['PetCO2_I'].min()) / (
            XN['PetCO2_I'].max() - XN['PetCO2_I'].min())
    # XN['VEVO2_I'] = (XN['VEVO2_I'] - XN['VEVO2_I'].min()) / (
    #         XN['VEVO2_I'].max() - XN['VEVO2_I'].min())
    # XN['VEVCO2_I'] = (XN['VEVCO2_I'] - XN['VEVCO2_I'].min()) / (
    #         XN['VEVCO2_I'].max() - XN['VEVCO2_I'].min())
    XN = XN.filter(filter_vars, axis=1)

    p_1 = []
    p_2 = []
    p_3 = []
    time = []
    VO2 = []
    VCO2 = []
    VE = []
    PetO2 = []
    PetCO2 = []

    for i in np.arange(1, len(XN) - past_points, inference_stride):
        XN_array = np.asarray(XN[i:i+int(past_points)])
        output_data = tf_model(XN_array.reshape(1, past_points, n_inputs))
        p_1.append(output_data.numpy()[0][0])
        p_2.append(output_data.numpy()[0][1])
        p_3.append(output_data.numpy()[0][2])
        time.append(df.time[i] + past_points)
        # ['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'domain']
        VO2.append(np.mean(XN_array[-1, 0]) * (df['VO2_I'].max() - df['VO2_I'].min()) + df['VO2_I'].min())
        VCO2.append(np.mean(XN_array[-1, 1]) * (df['VCO2_I'].max() - df['VCO2_I'].min()) + df['VCO2_I'].min())
        VE.append(np.mean(XN_array[-1, 2]) * (df['VE_I'].max() - df['VE_I'].min()) + df['VE_I'].min())
        PetO2.append(np.mean(XN_array[-1, 3]) * (df['PetO2_I'].max() - df['PetO2_I'].min()) + df['PetO2_I'].min())
        PetCO2.append(np.mean(XN_array[-1, 4]) * (df['PetCO2_I'].max() - df['PetCO2_I'].min()) + df['PetCO2_I'].min())

    tmp_df = pd.DataFrame()
    tmp_df['time'] = time
    tmp_df['p_md'] = optimal_filter(np.asarray(time), np.asarray(p_1), 100)
    tmp_df['p_hv'] = optimal_filter(np.asarray(time), np.asarray(p_2), 100)
    tmp_df['p_sv'] = optimal_filter(np.asarray(time), np.asarray(p_3), 100)

    # compute the normalised probabilities
    tmp_df['p_md_N'] = np.asarray(p_1) / (np.asarray(p_1) + np.asarray(p_2) + np.asarray(p_3))
    tmp_df['p_hv_N'] = np.asarray(p_2) / (np.asarray(p_1) + np.asarray(p_2) + np.asarray(p_3))
    tmp_df['p_sv_N'] = np.asarray(p_3) / (np.asarray(p_1) + np.asarray(p_2) + np.asarray(p_3))

    tmp_df.loc[tmp_df['p_md_N'] < 0, 'p_md_N'] = 0
    tmp_df.loc[tmp_df['p_hv_N'] < 0, 'p_hv_N'] = 0
    tmp_df.loc[tmp_df['p_sv_N'] < 0, 'p_sv_N'] = 0

    # # Perform kernel density estimation on the overlap probabilities
    # kde_1 = stats.gaussian_kde(time, weights=tmp_df['p_sv_N'] * tmp_df['p_md_N'])
    # kde_2 = stats.gaussian_kde(time, weights=tmp_df['p_md_N'] * tmp_df['p_hv_N'])
    #
    # overlap_ci_1 = [int(np.percentile(kde_1.resample(4000)[0], 25)), int(np.percentile(kde_1.resample(4000)[0], 75))]
    # overlap_ci_2 = [int(np.percentile(kde_2.resample(4000)[0], 25)), int(np.percentile(kde_2.resample(4000)[0], 75))]

    mod_col = tmp_df[['p_md', 'p_hv', 'p_sv']].iloc[:5].mean().idxmax()
    sev_col = tmp_df[['p_md', 'p_hv', 'p_sv']].iloc[-5:].mean().idxmax()
    for labels_ in ['p_md', 'p_hv', 'p_sv']:
        if labels_ not in [mod_col, sev_col]:
            hv_col = labels_

    out_df = pd.DataFrame()
    out_df['time'] = time
    out_df['p_md'] = tmp_df[mod_col]
    out_df['p_hv'] = tmp_df[hv_col]
    out_df['p_sv'] = tmp_df[sev_col]
    out_df['VO2'] = VO2
    out_df['VCO2'] = VCO2
    out_df['VE'] = VE
    out_df['PetO2'] = PetO2
    out_df['PetCO2'] = PetCO2
    out_df['VO2_F'] = optimal_filter(np.asarray(time), np.asarray(VO2), 100)

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
    # out_dict['VT1_upper'] = {}
    # out_dict['VT1_lower'] = {}
    # out_dict['VT2_upper'] = {}
    # out_dict['VT2_lower'] = {}

    # FIXME: hard coded
    VT1_index = int(out_df[(out_df['p_hv'] >= out_df['p_md'])].index[0] - int(past_points / inference_stride))
    VT2_index = int(out_df[(out_df['p_sv'] <= out_df['p_hv'])].index[-1] - int(past_points / inference_stride))

    VT1_time = int(out_df.iloc[VT1_index]['time'])
    VT2_time = int(out_df.iloc[VT2_index]['time'])

    out_dict['VT1']['time'] = VT1_time
    out_dict['VT2']['time'] = VT2_time

    # out_dict['VT1_upper']['time'] = overlap_ci_1[1]
    # out_dict['VT1_lower']['time'] = overlap_ci_1[0]
    # out_dict['VT2_upper']['time'] = overlap_ci_2[1]
    # out_dict['VT2_lower']['time'] = overlap_ci_2[0]

    out_dict['VT1']['HR'] = df.iloc[VT1_index]['HR_I']
    out_dict['VT2']['HR'] = df.iloc[VT2_index]['HR_I']
    # out_dict['VT1_upper']['HR'] = df[df.time == overlap_ci_1[1]]['HR_I'].values[0]
    # out_dict['VT1_lower']['HR'] = df[df.time == overlap_ci_1[0]]['HR_I'].values[0]
    # out_dict['VT2_upper']['HR'] = df[df.time == overlap_ci_2[1]]['HR_I'].values[0]
    # out_dict['VT2_lower']['HR'] = df[df.time == overlap_ci_2[0]]['HR_I'].values[0]

    out_dict['VT1']['VE'] = out_df.iloc[VT1_index]['VE']
    out_dict['VT2']['VE'] = out_df.iloc[VT2_index]['VE']
    # out_dict['VT1_upper']['VE'] = out_df[out_df.time == overlap_ci_1[1]]['VE'].values[0]
    # out_dict['VT1_lower']['VE'] = out_df[out_df.time == overlap_ci_1[0]]['VE'].values[0]
    # out_dict['VT2_upper']['VE'] = out_df[out_df.time == overlap_ci_2[1]]['VE'].values[0]
    # out_dict['VT2_lower']['VE'] = out_df[out_df.time == overlap_ci_2[0]]['VE'].values[0]

    out_dict['VT1']['VO2'] = out_df.iloc[VT1_index]['VO2_F']
    out_dict['VT2']['VO2'] = out_df.iloc[VT2_index]['VO2_F']
    # out_dict['VT1_upper']['VO2'] = out_df[out_df.time == overlap_ci_1[1]]['VO2_F'].values[0]
    # out_dict['VT1_lower']['VO2'] = out_df[out_df.time == overlap_ci_1[0]]['VO2_F'].values[0]
    # out_dict['VT2_upper']['VO2'] = out_df[out_df.time == overlap_ci_2[1]]['VO2_F'].values[0]
    # out_dict['VT2_lower']['VO2'] = out_df[out_df.time == overlap_ci_2[0]]['VO2_F'].values[0]

    return out_df, out_dict

def create_probabilities(duration=600,
                         VT1=320, VT2=460,
                         training=True, normalization=False, resting=True,
                         resting_duration=60,
                         initial_step=False,
                         y_pm0=1,
                         generator=False,
                         sigmoid_steepness=0.05):
    """Creates piecewise-linear probabilities for moderate/heavy/severe zones
    with enforced overlaps at VT1 and VT2.

    Behaviour:
      - Nodes at t=1, VT1, VT2, duration with values:
          t=1   : (y_pm0, 1-y_pm0, 0)
          VT1   : (0.5, 0.5, 0)
          VT2   : (0.0, 0.5, 0.5)
          end   : (0.0, 0.0, 1.0)
      - Linear interpolation between these nodes ensures fully linear transitions,
        exact overlaps at VT1 and VT2, and sum-to-1 (after numerical normalization).
    """

    # time vector (1..duration) to preserve your original indexing
    t = np.arange(1, duration + 1).astype(float)

    # safety checks
    if VT1 < 1:
        raise ValueError("VT1 must be >= 1")
    if VT2 <= VT1:
        raise ValueError("VT2 must be > VT1")
    if VT2 > duration:
        raise ValueError("VT2 must be <= duration")
    if y_pm0 < 0.0 or y_pm0 > 1.0:
        raise ValueError("y_pm0 must be in [0,1]")

    # guard minimum for y_pm0 as in your original code
    if y_pm0 < 0.65:
        print('This is too close to VT1, I am settling at 0.65')
        y_pm0 = 0.65

    # Define node times and node values for each zone
    node_times = np.array([1.0, float(VT1), float(VT2), float(duration)], dtype=float)

    # Node values: moderate, heavy, severe at each node time
    node_pm = np.array([y_pm0, 0.5, 0.0, 0.0], dtype=float)
    node_ph = np.array([1.0 - y_pm0, 0.5, 0.5, 0.0], dtype=float)
    node_ps = np.array([0.0, 0.0, 0.5, 1.0], dtype=float)

    # Interpolate linearly across the whole time vector
    p_mF = np.interp(t, node_times, node_pm)
    p_hF = np.interp(t, node_times, node_ph)
    p_sF = np.interp(t, node_times, node_ps)

    # Numerical safety: clip and normalize so they sum to 1 at each time point
    p_mF = np.clip(p_mF, 0.0, 1.0)
    p_hF = np.clip(p_hF, 0.0, 1.0)
    p_sF = np.clip(p_sF, 0.0, 1.0)

    total = p_mF + p_hF + p_sF
    # avoid division by zero (shouldn't happen), but protect anyway
    zero_mask = (total == 0)
    if np.any(zero_mask):
        total[zero_mask] = 1.0
    p_mF /= total
    p_hF /= total
    p_sF /= total

    # optional renormalization to [0,1] range per series (keeps shape but scales)
    if normalization:
        p_mF = np.interp(p_mF, (p_mF.min(), p_mF.max()), (0.0, 1.0))
        p_hF = np.interp(p_hF, (p_hF.min(), p_hF.max()), (0.0, 1.0))
        p_sF = np.interp(p_sF, (p_sF.min(), p_sF.max()), (0.0, 1.0))

        # re-normalize to sum=1 after per-series scaling
        total = p_mF + p_hF + p_sF
        total[total == 0] = 1.0
        p_mF /= total
        p_hF /= total
        p_sF /= total

    # Add small training noise if requested (same pattern as before)
    if training:
        noise_scale = 0.02  # your chosen default
        p_mF = p_mF + np.random.randn(len(t)) * noise_scale
        p_hF = p_hF + np.random.randn(len(t)) * noise_scale
        p_sF = p_sF + np.random.randn(len(t)) * noise_scale

        # Clip and re-normalize
        p_mF = np.clip(p_mF, 0.0, 1.0)
        p_hF = np.clip(p_hF, 0.0, 1.0)
        p_sF = np.clip(p_sF, 0.0, 1.0)

        total = p_mF + p_hF + p_sF
        total[total == 0] = 1.0
        p_mF /= total
        p_hF /= total
        p_sF /= total

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
                  noise_factor=None,
                  resting=False,
                  training=True,
                  normalization=False):
    """Generate synthetic CPET data using the TensorFlow generator model.

    NOTE: This function requires a loaded TensorFlow generator model (not TFLite).
    Use load_tf_generator() to load the model first. Requires full TensorFlow installation.

    Parameters:
        generator: TensorFlow generator model (required). Use load_tf_generator() to load.
        plot (bool): Whether to plot the generated data. Default False.
        fitness_group (int): Fitness level: 0=very low, 1=low, 2=medium, 3=high. None=random.
        noise_factor (float): Noise factor for white noise (1.5-2.5). None=random.
        resting (bool): Whether to include resting phase. Default False.
        training (bool): Training mode flag. Default True.
        normalization (bool): Whether to normalize data. Default False.

    Returns:
        df (pd.DataFrame): Pandas dataframe with generated CPET data
        data (dict): Metadata about the generated CPET test

    Raises:
        ValueError: If generator model is None or invalid
        ImportError: If required dependencies are missing

    Note:
        The generator uses a hardcoded input shape of [1, 23] which consists of:
        - 20 dimensions for data noise
        - 3 dimensions for domain probabilities (moderate, heavy, severe)
    """

    # Validate generator model
    if generator is None:
        raise ValueError(
            "Generator model is required but None was provided. "
            "Load the generator first with load_tf_generator(). "
            "Note: Requires full TensorFlow installation (not tflite_runtime)."
        )

    import random
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from scipy.interpolate import interp1d

    # Try to import uniplot for plotting (optional dependency)
    terminal_plot = None
    if plot:
        try:
            from uniplot import plot as terminal_plot
        except ImportError:
            print("Warning: uniplot not installed. Plotting disabled.")
            print("Install with: pip install uniplot")
            plot = False  # Disable plotting if uniplot not available

    # Set the minimum and maximum values
    minimum_value = 1800
    maximum_value = 4900

    if fitness_group is None:
        # Generate a random number
        VO2_peak = round(np.random.uniform(minimum_value, maximum_value), 1)

    if fitness_group == 0:
        VO2_peak = round(np.random.uniform(1600, 2200), 1)
    if fitness_group == 1:
        VO2_peak = round(np.random.uniform(2200, 3000), 1)
    if fitness_group == 2:
        VO2_peak = round(np.random.uniform(3000, 4000), 1)
    if fitness_group == 3:
        VO2_peak = round(np.random.uniform(4000, 5000), 1)

    VO2_VT1_efficiency = np.random.uniform(10.8, 11.8)

    if resting is None:
        resting = random.choice([0, 1])

    if resting:
        VO2_min = np.random.uniform(320, 540)
    else:
        if VO2_peak < 2200:
            VO2_basal = np.random.uniform(320, 540)
            # Assuming they are already delivering 30 W
            W0 = 50
            VO2_min = VO2_VT1_efficiency * W0 + np.random.uniform(-100, 100) + VO2_basal
            # 10 W/min ramp
            eps = 10
            duration = (VO2_peak - VO2_min)/(eps * VO2_VT1_efficiency)
            R_max = 1.08
        if VO2_peak >= 2200 and VO2_peak < 3000:
            VO2_basal = np.random.uniform(320, 540)
            # Assuming they are already delivering 50 W
            W0 = 50
            VO2_min = VO2_VT1_efficiency * W0 + np.random.uniform(-100, 100) + VO2_basal
            # 15 W/min ramp
            eps = 15
            duration = (VO2_peak - VO2_min)/(eps * VO2_VT1_efficiency)
            VCO2_peak = VO2_peak + np.random.uniform(160, 260)
            R_max = 1.12
        if VO2_peak >= 3000 and VO2_peak < 4000:
            VO2_basal = np.random.uniform(320, 540)
            # Assuming they are already delivering 80 W
            W0 = 80
            VO2_min = VO2_VT1_efficiency * W0 + np.random.uniform(-100, 100) + VO2_basal
            # 15 W/min ramp
            eps = 15
            duration = (VO2_peak - VO2_min)/(eps * VO2_VT1_efficiency)
            R_max = 1.16
        if VO2_peak > 4000:
            VO2_basal = np.random.uniform(320, 540)
            # Assuming they are already delivering 100 W
            W0 = 100
            VO2_min = VO2_VT1_efficiency * W0 + np.random.uniform(-100, 100) + VO2_basal
            # 25 W/min ramp
            eps = 25
            duration = (VO2_peak - VO2_min)/(eps * VO2_VT1_efficiency)
            R_max = 1.18

    # duration in sec for this application
    duration = round(duration * 60)

    VT1 = 0
    VT2 = 0
    VO2VT1_estimated = 0
    VO2VT2_estimated = 0

    while ((VT2 - VT1) < 60 or VO2VT1_estimated > 0.74 * VO2_peak or VO2VT2_estimated > 0.94 * VO2_peak or
           VO2VT1_estimated < 0.49 * VO2_peak or VO2VT2_estimated < 0.73 * VO2_peak):
        VT2 = round(np.random.normal(0.8, 0.12) * duration)
        VT1 = round(np.random.normal(0.7, 0.6) * VT2)
        VO2VT1_estimated = int((W0 + eps / 60 * VT1) * VO2_VT1_efficiency + VO2_basal)
        VO2VT2_estimated = int((W0 + eps / 60 * VT2) * VO2_VT1_efficiency + VO2_basal)

    VCO2_peak = R_max * VO2_peak + np.random.uniform(-60, 60)

    RF_min = np.random.uniform(14, 20)  # starting at 80 W

    VE_peak = 0.0455 * VO2_peak - 8.8016
    RF_peak = 0.0093 * VO2_peak + 26.21
    HR_peak = 0.0121 * VO2_peak + 126.74
    PetO2_peak = - 0.004 * VO2_peak + 122.37

    VCO2_min = 0.8592 * VO2_min - 11.805
    VE_min = 0.0226 * VO2_min + 2.7968
    HR_min = 0.0327 * VO2_min + 65.1
    PetO2_min = - 0.0269 * VO2_min + 107.6

    PetCO2_min = np.random.normal(30, 5)
    PetCO2_peak = PetCO2_min + 8 + np.random.uniform(5, 17)

    # probability definition
    d = duration
    y1 = -1 / (VT1 + 2 * VT2 - 4 * d) * VT1
    y2 = 1 / (VT1 + 2 * VT2 - 4 * d) * (VT1 - 2 * VT2)

    xp = [0, VT1, VT2, duration]
    yp = [0, y1, y2, 1]

    p_mF, p_hF, p_sF = create_probabilities(d, VT1, VT2,
                                            initial_step=True,
                                            generator=True)

    # # # IMPORTANT: normalization only in > 0.5
    # p_hF[p_hF > 0.5] = np.interp(p_hF[p_hF > 0.5], (0.5, p_hF.max()), (0.5, 1))
    # p_sF[p_sF > 0.5] = np.interp(p_sF[p_sF > 0.5], (0.5, p_sF.max()), (0.5, 1))

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
    input_data = np.array(np.random.random_sample([1, 23]))

    for seconds_ in time_array:
        # keep the seed?
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data[0, -3:] = np.array([[p_hF[seconds_], p_sF[seconds_], p_mF[seconds_]]])
        output_data = generator(input_data)
        VO2.append(np.mean(output_data[0, :, 0]))
        VCO2.append(np.mean(output_data[0, :, 1]))
        VE.append(np.mean(output_data[0, :, 2]))
        HR.append(np.mean(output_data[0, :, 3]))
        RF.append(np.mean(output_data[0, :, 4]))
        PetO2.append(np.mean(output_data[0, :, 5]))
        PetCO2.append(np.mean(output_data[0, :, 6]))
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

    if noise_factor is None:
        noise_factor = random.randint(3, 5)/2

    # Helper function to safely normalize data (avoid division by zero)
    def safe_normalize(data, min_val, max_val, target_min, target_max):
        """Safely normalize data avoiding division by zero"""
        data_array = np.asarray(data)
        data_min = np.min(data_array)
        data_range = np.max(data_array - data_min)
        if data_range == 0:
            # If all values are the same, return target mean
            return np.full_like(data_array, (target_min + target_max) / 2, dtype=float)
        return (data_array - data_min) / data_range * (target_max - target_min) + target_min

    # Apply normalization with safety checks
    df['VO2_I'] = safe_normalize(VO2, np.min(VO2), np.max(VO2), VO2_min, VO2_peak) + np.random.randn(len(VO2)) * 40 * noise_factor
    df['VCO2_I'] = safe_normalize(VCO2, np.min(VCO2), np.max(VCO2), VCO2_min, VCO2_peak) + np.random.randn(len(VO2)) * 40 * noise_factor
    df['VE_I'] = safe_normalize(VE, np.min(VE), np.max(VE), VE_min, VE_peak) + np.random.randn(len(VO2)) * 1.5 * noise_factor
    df['HR_I'] = (safe_normalize(HR, np.min(HR), np.max(HR), HR_min, HR_peak) + np.random.randn(len(VO2)) * 1 * noise_factor).astype(int)
    df['RF_I'] = safe_normalize(RF, np.min(RF), np.max(RF), RF_min, RF_peak) + np.random.randn(len(VO2)) * 1 * noise_factor
    df['PetO2_I'] = safe_normalize(PetO2, np.min(PetO2), np.max(PetO2), PetO2_min, PetO2_peak) + np.random.randn(len(VO2)) * 1.5 * noise_factor
    df['PetCO2_I'] = safe_normalize(PetCO2, np.min(PetCO2), np.max(PetCO2), PetCO2_min, PetCO2_peak) + np.random.randn(len(VO2)) * 1.5 * noise_factor

    # NEW: Add the workload
    df['W'] = eps * time_array / 60 + W0

    df['p_m'] = p_mF
    df['p_h'] = p_hF
    df['p_s'] = p_sF

    df['VEVO2_I'] = df['VE_I']/df['VO2_I']
    df['VEVCO2_I'] = df['VE_I']/df['VCO2_I']

    df['VCO2VO2_I'] = df['VCO2_I'] / df['VO2_I']
    df['PetO2VO2_I'] = df['PetO2_I'] / df['VO2_I']
    df['PetCO2VO2_I'] = df['PetCO2_I'] / df['VO2_I']

    df['VO2_F'] = optimal_filter(time_array, df['VO2_I'], 100)

    # TODO: this is hard coded, you need to subtract half window-length (see probability definition)
    df['domain'] = np.NaN
    df.loc[df['time'] < (VT1), 'domain'] = -1
    df.loc[df['time'] >= (VT2), 'domain'] = 1
    df.loc[(df['time'] < (VT2)) & (df['time'] >= (VT1)), 'domain'] = 0
    # df['fitness_group'] = db_df_sample['fitness_group'].values[0]
    # df['Age'] = db_df_sample['Age'].values[0]
    # df['age_group'] = db_df_sample['age_group'].values[0]
    # df['gender'] = db_df_sample['gender'].values[0]
    # df['weight'] = db_df_sample['weight'].values[0]
    # df['height'] = db_df_sample['height'].values[0]

    df['change'] = df['domain'].ne(df['domain'].shift())
    # Find the index where 'domain' changes from -1 to 0
    first_change_index = df.index[df['change'] & (df['domain'] == 0)].min()

    # Find the index where 'domain' changes from 0 to 1
    second_change_index = df.index[df['change'] & (df['domain'] == 1)].min()

    # Collect VO2 value at VT1 and VT2
    VO2VT1 = int(df.VO2_F[first_change_index])
    VO2VT2 = int(df.VO2_F[second_change_index])

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

    # if db_df_sample.fitness_group.values == 1:
    #     fitness_group = 'LOW'
    # if db_df_sample.fitness_group.values == 2:
    #     fitness_group = 'MEDIUM'
    # if db_df_sample.fitness_group.values == 3:
    #     fitness_group = 'HIGH'
    # if db_df_sample.gender.values == -1:
    #     gender = 'MALE'
    # if db_df_sample.gender.values == 1:
    #     gender = 'FEMALE'

    # print('Data generated for a ', gender, ' individual with ', fitness_group, ' fitness capacity.')
    # print('Weight: ', int(db_df_sample.weight.values), ' kg')
    # print('Height: ', db_df_sample.height.values[0], 'm')
    # print('Age: ', int(db_df_sample.Age.values), 'y')
    print('Noise factor: ', round(noise_factor, 2))
    print('VT1: ', str(VT1), ' sec after start')
    print('VT2: ', str(VT2), ' sec after start')
    print('Duration:', duration, ' sec')
    print('VO2VT1: ', str(int(VO2VT1)), ' mlO2')
    print('VO2VT2: ', str(int(VO2VT2)), ' mlO2')
    print('VO2VT1%: ', str(int(VO2VT1/VO2_peak * 100)), ' %')
    print('VO2VT2%: ', str(int(VO2VT2/VO2_peak * 100)), ' %')
    print('MAX RER: ', str(round(VCO2_peak / VO2_peak, 2)), ' %')
    print('VO2peak: ', str(int(VO2_peak)), ' mlO2')
    print('Resting: ', resting)
    print('W0: ', str(W0), ' W')
    print('ramp: ', str(eps), ' W/min')

    # TODO: create a function to generate this dict, as it is the same that we should use when drawing real data files
    # create dict fro Exercise Threshold App
    data = dict()
    # data['Age'] = str(int(db_df_sample.Age.values))
    # data['Height'] = str(db_df_sample.height.values[0])
    # data['Weight'] = str(int(db_df_sample.weight.values))
    # data['Gender'] = gender
    data['Aerobic_fitness_level'] = fitness_group
    data['VT1'] = str(VT1)
    data['VT2'] = str(VT2)
    data['VO2VT1'] = str(int(VO2VT1))
    data['VO2VT2'] = str(int(VO2VT2))
    data['VO2max'] = str(int(VO2_peak))
    data['LT'] = str(int(VO2VT1))
    data['LT_vo2max'] = str(int((VO2VT1/VO2_peak) * 100)) + '%'
    data['RCP'] = str(int(VO2VT2))
    data['RCP_vo2max'] = str(int((VO2VT2/VO2_peak) * 100)) + '%'
    data['id'] = 'fake_#'
    data['noise_factor'] = str(noise_factor)
    data['created'] = datetime.today().strftime("%m/%d/%Y - %H:%M:%S")
    data['resting'] = str(resting)
    data['W0'] = str(W0)
    data['W/min'] = str(eps)

    df['breaths'] = (1/(df.RF_I/60))
    # df['recorded_timestamp'] = pd.to_datetime(df.time, unit='s')

    time_cum_sum = 0
    df_breath = pd.DataFrame()
    n = 0
    while time_cum_sum < (duration-6):
        try:
            tmp = df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + df.breaths.iloc[n]))].median()
            # TODO: this 25 is hardcoded, you should have len(df.columns)
            df_breath = pd.concat([df_breath, pd.DataFrame(data=np.reshape(tmp.values, [1, 21]),
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