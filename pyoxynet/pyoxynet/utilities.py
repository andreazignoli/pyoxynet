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

def load_tf_model():
    """This function loads the saved tflite models.

    Args:
       name (str):  The name of the pickle file of the model.

    Returns:
       none

    """

    import importlib_resources
    import pickle
    from io import BytesIO
    import pyoxynet.models

    # get the model
    pip_install_tflite()
    import tflite_runtime.interpreter as tflite
    tfl_model_binaries = importlib_resources.read_binary(pyoxynet.models, 'tfl_model.pickle')
    tfl_model_decoded = pickle.loads(tfl_model_binaries)

    # save model locally on tmp
    open('/tmp/tfl_model' + '.tflite', 'wb').write(tfl_model_decoded.getvalue())
    interpreter = tflite.Interpreter(model_path='/tmp/tfl_model.tflite')

    return interpreter

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
        csv_file (str) : name of the data csv file

    Returns:
        df (pandas df) : Model output example

    """

    from importlib import resources
    import pandas as pd
    import pyoxynet.data_test

    if csv_file == 'data_test.csv':
        import pkgutil
        from io import StringIO
        bytes_data = pkgutil.get_data('pyoxynet.data_test', "data_test.csv")
        s = str(bytes_data, 'utf-8')
        data = StringIO(s)
        df = pd.read_csv(data)

    return df

def test_pyoxynet():

    tfl_model = load_tf_model()
    df = load_csv_data()

    X = df[['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I']]
    XN = normalize(X)

    import numpy as np

    input_details = tfl_model.get_input_details()
    output_details = tfl_model.get_output_details()

    time_series_len = input_details[0]['shape'][1]
    p_md = []
    p_hv = []
    p_sv = []
    time = []

    for i in np.arange(len(XN)-time_series_len):
        XN_array = np.asarray(XN[i:(i+time_series_len)])
        input_data = np.reshape(XN_array, input_details[0]['shape'])
        input_data = input_data.astype(np.float32)

        tfl_model.allocate_tensors()
        tfl_model.set_tensor(input_details[0]['index'], input_data)
        tfl_model.invoke()
        output_data = tfl_model.get_tensor(output_details[0]['index'])
        p_md.append(output_data[0][2])
        p_sv.append(output_data[0][1])
        p_hv.append(output_data[0][0])
        time.append(df.time[i])

    import pandas as pd
    df = pd.DataFrame()
    df['time'] = time
    df['p_md'] = p_md
    df['p_hv'] = p_hv
    df['p_sv'] = p_sv

    from uniplot import plot

    plot([p_md, p_sv, p_hv], title="Probabilities", color=True, legend_labels=['Moderate', 'Heavy', 'Severe'])