import numpy as np
import scipy.interpolate
from scipy.stats import beta
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

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

# The models this package ships: frozen .tflite files in pyoxynet/models, run with LiteRT.
# models/README.md records where each one came from.
MODELS = {
    'CNN': 'cnn.tflite',
    'LSTM': 'lstm.tflite',
}
# Names accepted before 1.0.0, kept so that old code fails with a clear message.
REMOVED_MODELS = ('transformer', 'TCN', 'LSTMGRUModel', 'murias_lab', 'AIS', 'NSWIS')


class ModelOutput(np.ndarray):
    """A numpy array that also answers .numpy(), as the TensorFlow tensors returned before 1.0.0 did."""

    def numpy(self):
        return np.asarray(self)


class TFLiteModel:
    """A frozen pyoxynet model, called like a Keras model: model(x) returns an array.

    x is one sample or a batch: for the threshold models (40, 5), (1, 40, 5) or (n, 40, 5), which
    give (1, 3) or (n, 3). Any other shape raises a ValueError. Calls are serialised, so one model
    can be shared between threads.

    Args:
       filename (str):  Name of a .tflite file in pyoxynet/models.

    """

    def __init__(self, filename):
        import threading
        from importlib.resources import files
        from ai_edge_litert.interpreter import Interpreter

        self.filename = filename
        model_content = files('pyoxynet').joinpath('models').joinpath(filename).read_bytes()
        self._interpreter = Interpreter(model_content=model_content)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        # One interpreter holds one input and one output buffer, so calls must not overlap
        self._lock = threading.Lock()
        self.input_shape = tuple(int(n) for n in self._input['shape'])
        self.sample_shape = self.input_shape[1:]

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape == self.sample_shape:
            x = x[np.newaxis]
        if x.ndim != len(self.input_shape) or x.shape[1:] != self.sample_shape or len(x) == 0:
            raise ValueError(f'{self.filename} takes an array of shape {self.sample_shape}, or a batch '
                             f'of them, (n, {", ".join(map(str, self.sample_shape))}); got {x.shape}.')
        outputs = []
        with self._lock:
            for sample in x:
                self._interpreter.set_tensor(self._input['index'], sample[np.newaxis])
                self._interpreter.invoke()
                outputs.append(self._interpreter.get_tensor(self._output['index'])[0].copy())
        return np.stack(outputs).view(ModelOutput)

    def predict(self, x):
        """As Keras' predict: model(x) as a plain numpy array."""
        return np.asarray(self(x))

    def __repr__(self):
        return f'TFLiteModel({self.filename!r}, input_shape={self.input_shape})'


def load_model(model='CNN'):
    """Loads one of the frozen pyoxynet models.

    Args:
       model (str):  'CNN' (default), the convolutional network evaluated in Keir et al. (2026),
          or 'LSTM', a recurrent network with LSTM and GRU layers.

    Returns:
       model (TFLiteModel) : call it with an array of shape (1, 40, 5) to get the three domain outputs

    """

    if model in REMOVED_MODELS:
        raise ValueError(f"The '{model}' model was removed in pyoxynet 1.0.0. "
                         f"Available models: {', '.join(MODELS)}.")
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Available models: {', '.join(MODELS)}.")

    return TFLiteModel(MODELS[model])


def load_tf_model(n_inputs=5, past_points=40, model='CNN'):
    """Loads one of the frozen pyoxynet models. Kept for code written before 1.0.0: use load_model.

    Args:
       n_inputs (int):  Number of input variables. The models take 5.
       past_points (int):  Number of past inputs in the time-series. The models take 40.
       model (str):  'CNN' (default) or 'LSTM'.

    Returns:
       model (TFLiteModel) : see load_model

    """

    if (n_inputs, past_points) != (5, 40):
        raise ValueError('The pyoxynet models take 5 inputs over 40 past points '
                         f'(n_inputs=5, past_points=40), not n_inputs={n_inputs}, past_points={past_points}.')

    return load_model(model)


def load_generator():
    """Loads the conditional GAN generator that generate_CPET draws synthetic tests from.

    Returns:
       generator (TFLiteModel) : call it with an array of shape (1, 23) to get a (1, 40, 7) window

    """

    return TFLiteModel('generator.tflite')


def load_tf_generator():
    """Loads the generator. Kept for code written before 1.0.0: use load_generator.

    Returns:
       generator (TFLiteModel) : see load_generator

    """

    return load_generator()

def load_csv_data(csv_file='data_test.csv'):
    """Loads a CSV file, or the bundled example test when called without arguments

    Parameters:
        csv_file (str) : Path to a CSV file. 'data_test.csv' (the default) is the bundled example.

    Returns:
        df (pandas df) : The file's contents

    """

    import pandas as pd

    if csv_file == 'data_test.csv':
        import pkgutil
        from io import StringIO
        bytes_data = pkgutil.get_data('pyoxynet.data_test', csv_file)
        return pd.read_csv(StringIO(str(bytes_data, 'utf-8')))

    return pd.read_csv(csv_file)

EXERCISE_THRESHOLD_NAMES = {"time": "t",
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


def _breath_by_breath(df, duration):
    """Rebuild breaths from a one-second test: each breath is the median of the rows it spans

    Breath lengths come from RF_I. The result uses the Exercise Thresholds names (t, VO2, ...).
    Adds a 'breaths' column (breath length, s) to df, as draw_real_test and generate_CPET always did.

    Parameters:
        df (pd df) : one-second test with time and RF_I
        duration (int) : length of the test in seconds

    Returns:
        records (list of dict) : one dict per breath

    """

    import numpy as np
    import pandas as pd

    df['breaths'] = (1/(df.RF_I/60))

    time_cum_sum = 0
    df_breath = pd.DataFrame()
    n = 0
    while time_cum_sum < (duration - 6):
        try:
            breath = df.breaths.iloc[n]
            # Some example tests record RF as 0, which added noise can make negative: a breath of
            # zero or negative length would never move the loop forward, so step one second instead.
            if not np.isfinite(breath) or breath <= 0:
                breath = 1.0
            tmp = df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + breath))].median()
            df_breath = pd.concat([df_breath, pd.DataFrame(data=np.reshape(tmp.values, [1, len(df.columns)]),
                                                           columns=tmp.index.to_list())])
            time_cum_sum = (time_cum_sum + breath)
            n = n + len(df[(df.time >= time_cum_sum) & (df.time < (time_cum_sum + breath))])
        except:
            break

    df_breath = df_breath.dropna()
    df_breath['time'] = df_breath['time'].astype(int)
    df_breath = df_breath.drop_duplicates('time')
    df_breath = df_breath.rename(columns=EXERCISE_THRESHOLD_NAMES)

    return df_breath.to_dict(orient='records')


def draw_real_test(resting='random', index=None):
    """Draw one of the bundled example tests, with a little random noise added to its signals

    Parameters:
        resting (bool or 'random') : a test with a resting phase (50 of them) or a ramp test (88)
        index (int) : which test, from 1 (with resting True or False); a random one if not given

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

    if index is not None and resting == 'random':
        raise ValueError('Give resting=True or resting=False with index: the 50 resting tests and '
                         'the 88 ramp tests are numbered separately.')

    if resting == 'random':
        resting = random.randint(0, 1)

    kind, count = ('resting', 50) if resting else ('ramp', 88)
    if index is None:
        index = random.randint(1, count)
    elif not 1 <= index <= count:
        raise ValueError(f'index must be between 1 and {count} for the {kind} tests, not {index}.')

    file_name = f'{kind}_real_test_{index}.csv'
    bytes_data = pkgutil.get_data('pyoxynet.data_test', file_name)
    print(file_name, ' from', kind)

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

    # create the dict for the exercise threshold app
    data['data'] = _breath_by_breath(df, duration)

    return df, data

def load_exercise_threshold_app_data(data_dict={}, filter_size = 20):
    """Loads data from data dict with format provided by https://www.exercisethresholds.com/

    Parameters:
        data_dict (dict) : Dictionary with format like test/exercise_threshold_app_test.json

    Returns:
        df (pandas df) : Pandas data frame with format that can be used by Pyoxynet for inference (columns needed: 'VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'VEVO2_I', 'VEVCO2_I'; 'HR_I' when the data has HR)

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
        HR_I.append(data_points_.get('HR', np.nan))

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
    HR_I = np.asarray(HR_I, dtype=float)

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
    # Heart rate only when every point has it: a column of zeros would be reported as 0 bpm
    if len(HR_I) and np.all(np.isfinite(HR_I)):
        df['HR_I'] = np.interp(time_I, time, uniform_filter1d(HR_I, size=filter_size))

    return df

def test_pyoxynet(tf_model=None, input_df=None, n_inputs=5, past_points=40, model='CNN', plot=False, inference_stride=1):
    """Runs the pyoxynet inference

    Parameters:
        tf_model (TFLiteModel) : Model loaded with load_model (loads `model` if not given)
        input_df (pd df) : Test with columns time (s), VO2_I, VCO2_I, VE_I, PetO2_I, PetCO2_I and,
            optionally, HR_I; any sampling, put on a one-second grid here (uses the bundled
            example test if not given)
        n_inputs (int) : Number of inputs (deafult to Oxynet configuration)
        past_points (int) : Number of past points in the time series (deafult to Oxynet configuration)
        model (str) : 'CNN' (default) or 'LSTM', used when tf_model is not given
        plot (bool) : Plot the domain probabilities in the terminal
        inference_stride (int) : Stride inference for NN - speed up computation

    Returns:
        out_df (pd df) : Domain probabilities over time
        out_dict (dict) : VT1 and VT2 (time, HR, VE, VO2)

    """

    import numpy as np
    import pandas as pd

    # [] was the default before 1.0.0, so old calls may still pass it
    if tf_model is None or isinstance(tf_model, list):
        tf_model = load_tf_model(n_inputs=n_inputs, past_points=past_points, model=model)

    if input_df is None or len(input_df) == 0:
        print('Using default pyoxynet data')
        df = load_csv_data()
    else:
        df = input_df

    signals = ['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I']
    missing = [c for c in ['time'] + signals if c not in df.columns]
    if missing:
        raise ValueError(f'The test has no {", ".join(missing)} column. test_pyoxynet reads time (s), '
                         f'{", ".join(signals)} and, optionally, HR_I.')
    if not pd.api.types.is_numeric_dtype(df['time']):
        raise ValueError('The time column must hold seconds from the start of the test, as numbers.')

    # some adjustments to input df
    # TODO: create dedicated function for this
    df = df.drop_duplicates('time')
    # Only numeric columns can be averaged into one-second bins. Text columns (a phase marker, an id)
    # would make resample().mean() fail, and the models do not read them.
    df = df[[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]].copy()
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('timestamp')
    df = df.resample('1s').mean()
    df = df.interpolate()
    # interpolate() leaves the rows before a signal's first value empty: hold that first value
    df[signals] = df[signals].bfill().ffill()
    unusable = [c for c in signals if df[c].isna().all() or df[c].max() == df[c].min()]
    if unusable:
        raise ValueError(f'{", ".join(unusable)} is empty or constant in this test; the models need all '
                         'five signals to vary.')
    min_length = 2 * past_points
    if len(df) < min_length:
        raise ValueError(f'The test covers {len(df)} s; test_pyoxynet needs at least {min_length} s.')
    df['VO2_20s'] = df.VO2_I.rolling(20, win_type='triang', center=True).mean().bfill().ffill()
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

    # each signal's range, to undo the scaling; computed once rather than in every window
    low = {c: df[c].min() for c in signals}
    span = {c: df[c].max() - df[c].min() for c in signals}

    for i in np.arange(1, len(XN) - past_points, inference_stride):
        XN_array = np.asarray(XN[i:i+int(past_points)])
        output_data = np.asarray(tf_model(XN_array.reshape(1, past_points, n_inputs)))
        p_1.append(output_data[0][0])
        p_2.append(output_data[0][1])
        p_3.append(output_data[0][2])
        time.append(df.time[i] + past_points)
        # ['VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'domain']
        VO2.append(np.mean(XN_array[-1, 0]) * span['VO2_I'] + low['VO2_I'])
        VCO2.append(np.mean(XN_array[-1, 1]) * span['VCO2_I'] + low['VCO2_I'])
        VE.append(np.mean(XN_array[-1, 2]) * span['VE_I'] + low['VE_I'])
        PetO2.append(np.mean(XN_array[-1, 3]) * span['PetO2_I'] + low['PetO2_I'])
        PetCO2.append(np.mean(XN_array[-1, 4]) * span['PetCO2_I'] + low['PetCO2_I'])

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

    if plot:
        from uniplot import plot as terminal_plot
        terminal_plot([out_df['p_md'], out_df['p_hv'], out_df['p_sv']],
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

    VT1_candidates = out_df[(out_df['p_hv'] >= out_df['p_md'])].index
    VT2_candidates = out_df[(out_df['p_sv'] <= out_df['p_hv'])].index
    # The three columns are labelled by which output leads at the start and at the end of the test,
    # so a missing crossing says the test does not span three domains, not which one is missing.
    for name, candidates in (('VT1', VT1_candidates), ('VT2', VT2_candidates)):
        if len(candidates) == 0:
            raise ValueError(f'The model outputs never cross where {name} would be, so no {name} can be '
                             'placed in this test. This happens when a test does not span all three '
                             'domains, for example when it stops before the severe domain.')

    # Each threshold is placed at the start of the 40 s window in which the outputs cross
    VT1_index = int(VT1_candidates[0] - int(past_points / inference_stride))
    VT2_index = int(VT2_candidates[-1] - int(past_points / inference_stride))
    if VT1_index < 0 or VT2_index < 0:
        raise ValueError(f'The outputs cross within the first {past_points} s of the test, before the '
                         'model has seen a full window, so the threshold cannot be placed.')

    VT1_time = int(out_df.iloc[VT1_index]['time'])
    VT2_time = int(out_df.iloc[VT2_index]['time'])
    if VT1_time >= VT2_time:
        raise ValueError(f'The outputs place VT1 ({VT1_time} s) at or after VT2 ({VT2_time} s), so the '
                         'two thresholds are not consistent in this test.')

    out_dict['VT1']['time'] = VT1_time
    out_dict['VT2']['time'] = VT2_time

    # out_dict['VT1_upper']['time'] = overlap_ci_1[1]
    # out_dict['VT1_lower']['time'] = overlap_ci_1[0]
    # out_dict['VT2_upper']['time'] = overlap_ci_2[1]
    # out_dict['VT2_lower']['time'] = overlap_ci_2[0]

    # Heart rate is reported, not used by the models, so a test without it still runs. It is read at
    # the threshold's time: VT1_index counts windows, not the seconds of df.
    def hr_at(seconds):
        return float(df.at[(df['time'] - seconds).abs().idxmin(), 'HR_I'])

    has_hr = 'HR_I' in df.columns
    out_dict['VT1']['HR'] = hr_at(VT1_time) if has_hr else float('nan')
    out_dict['VT2']['HR'] = hr_at(VT2_time) if has_hr else float('nan')
    # out_dict['VT1_upper']['HR'] = df[df.time == overlap_ci_1[1]]['HR_I'].values[0]
    # out_dict['VT1_lower']['HR'] = df[df.time == overlap_ci_1[0]]['HR_I'].values[0]
    # out_dict['VT2_upper']['HR'] = df[df.time == overlap_ci_2[1]]['HR_I'].values[0]
    # out_dict['VT2_lower']['HR'] = df[df.time == overlap_ci_2[0]]['HR_I'].values[0]

    out_dict['VT1']['VE'] = float(out_df.iloc[VT1_index]['VE'])
    out_dict['VT2']['VE'] = float(out_df.iloc[VT2_index]['VE'])
    # out_dict['VT1_upper']['VE'] = out_df[out_df.time == overlap_ci_1[1]]['VE'].values[0]
    # out_dict['VT1_lower']['VE'] = out_df[out_df.time == overlap_ci_1[0]]['VE'].values[0]
    # out_dict['VT2_upper']['VE'] = out_df[out_df.time == overlap_ci_2[1]]['VE'].values[0]
    # out_dict['VT2_lower']['VE'] = out_df[out_df.time == overlap_ci_2[0]]['VE'].values[0]

    out_dict['VT1']['VO2'] = float(out_df.iloc[VT1_index]['VO2_F'])
    out_dict['VT2']['VO2'] = float(out_df.iloc[VT2_index]['VO2_F'])
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
    """Generate synthetic CPET data using the conditional GAN generator.

    Load the generator first with load_generator().

    Parameters:
        generator: Generator model (required). Use load_generator() to load.
        plot (bool): Whether to plot the generated data. Default False.
        fitness_group (int): Fitness level: 0=very low, 1=low, 2=medium, 3=high. None=random.
        noise_factor (float): Noise factor for white noise (1.5-2.5). None=random.
        resting (bool): Whether to include a resting phase. Only False (or None) is implemented: True raises NotImplementedError.
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
            "Load the generator first with load_generator()."
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

    # None used to pick at random, but only tests without a resting phase can be generated
    if resting is None:
        resting = False

    if resting:
        # This branch never set the ramp (W0, ramp rate, duration) the rest of the function needs,
        # so it always failed with an UnboundLocalError.
        raise NotImplementedError('generate_CPET does not generate a resting phase: '
                                  'only resting=False is implemented.')
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
        if VO2_peak >= 4000:
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

    # VT1 below 1 s or VT2 past the end would fail in create_probabilities: draw again
    while ((VT2 - VT1) < 60 or VO2VT1_estimated > 0.74 * VO2_peak or VO2VT2_estimated > 0.94 * VO2_peak or
           VO2VT1_estimated < 0.49 * VO2_peak or VO2VT2_estimated < 0.73 * VO2_peak or
           VT1 < 1 or VT2 > duration):
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
        output_data = np.asarray(generator(input_data))
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
    df['domain'] = np.nan
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

    # create the dict for the exercise threshold app
    data['data'] = _breath_by_breath(df, duration)

    return df, data