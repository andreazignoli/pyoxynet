"""The shipped models and the inference pipeline, against what 0.1.12 gave with the same weights.

reference_0_1_12.json was recorded by running pyoxynet 0.1.12's own test_pyoxynet on the bundled
example tests (see "recorded_with" in the file).
"""
import json
import pkgutil
import random
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyoxynet
from pyoxynet import utilities

REFERENCE = json.loads((Path(__file__).parent / 'reference_0_1_12.json').read_text())['thresholds']

_loaded = {}


def model(name):
    if name not in _loaded:
        _loaded[name] = utilities.load_model(name)
    return _loaded[name]


def sample(name):
    if name == 'data_test':
        return utilities.load_csv_data()
    raw = pkgutil.get_data('pyoxynet.data_test', f'{name}.csv')
    return pd.read_csv(StringIO(raw.decode()))


@pytest.mark.parametrize('key', sorted(REFERENCE))
def test_thresholds_match_0_1_12(key):
    model_name, sample_name = key.split('/')
    df = sample(sample_name)
    expected = REFERENCE[key]

    if 'error' in expected:
        # 0.1.12 raised an IndexError here; 1.0.0 says why
        with pytest.raises(ValueError, match='never crosses'):
            utilities.test_pyoxynet(tf_model=model(model_name), input_df=df)
        return

    out_df, out = utilities.test_pyoxynet(tf_model=model(model_name), input_df=df)
    assert len(out_df) == expected['n']
    for vt in ('VT1', 'VT2'):
        assert out[vt]['time'] == expected[vt]['time']
        assert out[vt]['VO2'] == pytest.approx(expected[vt]['VO2'], abs=0.5)
        assert out[vt]['VE'] == pytest.approx(expected[vt]['VE'], abs=0.05)
        assert out[vt]['HR'] == pytest.approx(expected[vt]['HR'], abs=0.5, nan_ok=True)
    assert out_df['p_md'].sum() == pytest.approx(expected['p_md_sum'], rel=1e-4)
    assert out_df['p_sv'].sum() == pytest.approx(expected['p_sv_sum'], rel=1e-4)


def test_heart_rate_is_optional():
    df = utilities.load_csv_data()[['time', 'VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I']]
    _, out = utilities.test_pyoxynet(tf_model=model('CNN'), input_df=df)
    assert out['VT1']['time'] == REFERENCE['CNN/data_test']['VT1']['time']
    assert np.isnan(out['VT1']['HR'])


def test_load_csv_data_reads_a_path(tmp_path):
    path = tmp_path / 'test.csv'
    utilities.load_csv_data().to_csv(path, index=False)
    assert len(utilities.load_csv_data(str(path))) == len(utilities.load_csv_data())


def test_defaults_are_the_cnn_on_the_bundled_test():
    _, out = utilities.test_pyoxynet()
    assert out['VT1']['time'] == REFERENCE['CNN/data_test']['VT1']['time']
    assert out['VT2']['time'] == REFERENCE['CNN/data_test']['VT2']['time']


@pytest.mark.parametrize('name', ['CNN', 'LSTM'])
def test_threshold_models_take_40_by_5_and_give_3(name):
    m = model(name)
    assert m.input_shape == (1, 40, 5)
    out = m(np.zeros((1, 40, 5)))
    assert out.shape == (1, 3)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize('name', utilities.REMOVED_MODELS)
def test_removed_models_raise_with_the_available_ones(name):
    with pytest.raises(ValueError, match='removed in pyoxynet 1.0.0.*CNN, LSTM'):
        utilities.load_model(name)


def test_old_loader_still_works_and_refuses_other_shapes():
    assert utilities.load_tf_model(n_inputs=5, past_points=40, model='LSTM').input_shape == (1, 40, 5)
    with pytest.raises(ValueError, match='5 inputs over 40 past points'):
        utilities.load_tf_model(n_inputs=7, past_points=40)


def test_generator_makes_a_test():
    random.seed(0)
    np.random.seed(0)
    generator = utilities.load_generator()
    assert generator(np.zeros((1, 23))).shape == (1, 40, 7)

    df, data = utilities.generate_CPET(generator)
    for column in ('time', 'VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'domain'):
        assert column in df.columns
    assert int(data['VT1']) < int(data['VT2']) < len(df)
    assert len(data['data']) > 0


def test_a_generated_test_runs_through_inference():
    random.seed(1)
    np.random.seed(1)
    df, _ = utilities.generate_CPET(utilities.load_generator())
    _, out = utilities.test_pyoxynet(input_df=df)
    assert out['VT1']['time'] < out['VT2']['time']


@pytest.mark.parametrize('kind', ['ramp_real_test', 'resting_real_test'])
def test_every_example_test_is_present(kind):
    count = {'ramp_real_test': 88, 'resting_real_test': 50}[kind]
    for n in range(1, count + 1):
        assert pkgutil.get_data('pyoxynet.data_test', f'{kind}_{n}.csv')


def test_draw_real_test():
    random.seed(0)
    df, data = utilities.draw_real_test(resting=False)
    assert {'VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I'} <= set(df.columns)
    assert int(data['VT1']) < int(data['VT2'])


@pytest.mark.parametrize('index', [35, 36, 38])
def test_draw_real_test_finishes_on_tests_without_respiratory_rate(index):
    # these three record RF as 0; with the added noise the breath loop used to run forever
    for seed in range(5):
        random.seed(seed)
        df, data = utilities.draw_real_test(resting=True, index=index)
        assert len(data['data']) > 0


def test_generate_cpet_refuses_a_resting_phase():
    with pytest.raises(NotImplementedError):
        utilities.generate_CPET(utilities.load_generator(), resting=True)


def test_the_file_reader_reads_a_generated_test(tmp_path):
    random.seed(4)
    np.random.seed(4)
    df, _ = utilities.generate_CPET(utilities.load_generator())
    df.to_csv(tmp_path / 'generated.csv', index=False)

    t = pyoxynet.Test(str(tmp_path / 'generated'))
    t.set_data_extension('.csv')
    t.infer_metabolimeter()
    t.load_file()
    t.create_data_frame()
    t.create_raw_data_frame()
    assert t.metabolimeter == 'generated_pyoxynet'
    _, out = utilities.test_pyoxynet(input_df=t.data_frame)
    assert out['VT1']['time'] < out['VT2']['time']


def test_exercise_thresholds_json_runs_through_inference():
    data = json.loads(pkgutil.get_data('pyoxynet.test', 'exercise_threshold_app_test.json'))
    df = utilities.load_exercise_threshold_app_data(data_dict=data)
    _, out = utilities.test_pyoxynet(input_df=df)
    assert out['VT1']['time'] < out['VT2']['time']


def test_version():
    assert pyoxynet.__version__ == '1.0.0'
