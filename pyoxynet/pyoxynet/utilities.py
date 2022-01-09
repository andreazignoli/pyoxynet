def PrintHello(hello='hello'):
    """This function prints to screen.

    Args:
       name (str):  The name to use.

    Returns:
       none

    """
    
    print(hello)
    
    return

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

    pip_install_tflite()
    import tflite_runtime.interpreter as tflite

    #interpreter = tflite.Interpreter(model_path='models/tfl_model.tflite')
    #output_details = interpreter.get_output_details()
    #interpreter.allocate_tensors()

    # tfl_model.tflite
    tfl_model_binaries = importlib_resources.read_binary(pyoxynet.models, 'tfl_model.pickle')
    tfl_model_decoded = pickle.loads(tfl_model_binaries)

    # save model locally on tmp
    open('/tmp/tfl_model' + '.tflite', 'wb').write(tfl_model_decoded.getvalue())
    interpreter = tflite.Interpreter(model_path='/tmp/tfl_model.tflite')

    return interpreter

def pip_install_tflite():

    import os
    import pkg_resources
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s" % (i.key) for i in installed_packages])
    print(installed_packages_list)

    if 'tflite-runtime' in installed_packages_list:
        print('Tflite runtime already present in the package list (skipping)')
    else:
        os.system("pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime")