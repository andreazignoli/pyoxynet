"""One test per finding of the 1.0.0 code review, fixed in 1.0.1."""
import contextlib
import io
import json
import pkgutil
import random
import threading
from io import StringIO

import numpy as np
import pandas as pd
import pytest

import pyoxynet
from pyoxynet import utilities

_loaded = {}


def model(name='CNN'):
    if name not in _loaded:
        _loaded[name] = utilities.load_model(name)
    return _loaded[name]


def example():
    return utilities.load_csv_data()


def ramp(n):
    return pd.read_csv(StringIO(pkgutil.get_data('pyoxynet.data_test', f'ramp_real_test_{n}.csv').decode()))


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# --- inference

def test_heart_rate_is_read_at_the_threshold_time():
    df = example()
    _, out = utilities.test_pyoxynet(tf_model=model(), input_df=df)
    for vt in ('VT1', 'VT2'):
        at = float(df.loc[df['time'] == out[vt]['time'], 'HR_I'].iloc[0])
        assert out[vt]['HR'] == pytest.approx(at)


def test_vt1_after_vt2_raises():
    with pytest.raises(ValueError, match='at or after VT2'):
        utilities.test_pyoxynet(tf_model=model(), input_df=ramp(4))


def test_leading_blank_rows_are_filled():
    df = example()
    df.loc[df.index[:6], 'VO2_I'] = np.nan
    _, out = utilities.test_pyoxynet(tf_model=model(), input_df=df)
    # the filled rows move VO2's minimum, so the scaling, and the thresholds, move a little
    assert abs(out['VT1']['time'] - 372) <= 10 and abs(out['VT2']['time'] - 550) <= 10


@pytest.mark.parametrize('column', ['PetCO2_I', 'VE_I'])
def test_a_constant_or_empty_signal_raises_clearly(column):
    df = example()
    df[column] = 35.0
    with pytest.raises(ValueError, match=f'{column} is empty or constant'):
        utilities.test_pyoxynet(tf_model=model(), input_df=df)
    df[column] = np.nan
    with pytest.raises(ValueError, match=f'{column} is empty or constant'):
        utilities.test_pyoxynet(tf_model=model(), input_df=df)


def test_a_missing_column_raises_clearly():
    with pytest.raises(ValueError, match='no PetO2_I column'):
        utilities.test_pyoxynet(tf_model=model(), input_df=example().drop(columns='PetO2_I'))


def test_text_columns_are_ignored():
    df = example()
    df['phase'] = np.where(df['time'] < 300, 'warm-up', 'ramp')
    df['subject'] = 'S01'
    _, out = utilities.test_pyoxynet(tf_model=model(), input_df=df)
    assert (out['VT1']['time'], out['VT2']['time']) == (372, 550)


def test_a_short_test_raises_clearly():
    with pytest.raises(ValueError, match='at least 80 s'):
        utilities.test_pyoxynet(tf_model=model(), input_df=example().iloc[:40])


def test_an_early_crossing_raises_instead_of_wrapping():
    calls = []

    def early(x):
        # moderate leads for 10 windows, then heavy, then severe: VT1's window starts before t=0
        k = len(calls)
        calls.append(k)
        return np.array([[0.9, 0.1, 0.0]] if k < 10 else [[0.1, 0.8, 0.1]] if k < 500 else [[0.0, 0.1, 0.9]])

    with pytest.raises(ValueError, match='within the first 40 s'):
        utilities.test_pyoxynet(tf_model=early, input_df=example())


@pytest.mark.parametrize('end', [300, 450])
def test_no_crossing_message_names_the_threshold_not_a_domain(end):
    df = example()
    with pytest.raises(ValueError, match='never cross where VT2 would be') as err:
        utilities.test_pyoxynet(tf_model=model(), input_df=df[df['time'] <= end])
    assert 'severe stays above heavy' not in str(err.value)


def test_exercise_thresholds_data_without_hr_reports_nan():
    data = json.loads(pkgutil.get_data('pyoxynet.test', 'exercise_threshold_app_test.json'))
    df = utilities.load_exercise_threshold_app_data(data_dict=data)
    assert 'HR_I' not in df.columns
    _, out = utilities.test_pyoxynet(tf_model=model(), input_df=df)
    assert np.isnan(out['VT1']['HR']) and np.isnan(out['VT2']['HR'])


# --- the model wrapper

def test_one_model_can_be_shared_between_threads():
    m = model()
    rng = np.random.default_rng(0)
    inputs = [rng.random((1, 40, 5)) for _ in range(240)]
    expected = [np.asarray(m(x)) for x in inputs]
    got = [None] * len(inputs)

    def work(part):
        for i in range(part, len(inputs), 8):
            got[i] = np.asarray(m(inputs[i]))

    threads = [threading.Thread(target=work, args=(p,)) for p in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e, g in zip(expected, got):
        np.testing.assert_array_equal(e, g)


def test_the_model_takes_batches_and_answers_numpy():
    m = model()
    batch = np.random.default_rng(1).random((5, 40, 5))
    out = m(batch)
    assert out.shape == (5, 3)
    for i in range(5):
        np.testing.assert_array_equal(out[i], m(batch[i])[0])
    assert m(batch[0]).shape == (1, 3)
    assert isinstance(out.numpy(), np.ndarray) and type(out.numpy()) is np.ndarray
    assert type(m.predict(batch)) is np.ndarray


@pytest.mark.parametrize('shape', [(1, 5, 40), (200,), (0, 40, 5), (2, 40, 6)])
def test_the_model_refuses_other_shapes(shape):
    with pytest.raises(ValueError, match='takes an array of shape'):
        model()(np.zeros(shape))


# --- generation and the example tests

def test_low_fitness_generation_never_draws_vt1_below_one():
    generator = utilities.load_generator()
    for seed in range(40):
        random.seed(seed)
        np.random.seed(seed)
        df, info = quiet(utilities.generate_CPET, generator, fitness_group=0)
        assert 1 <= int(info['VT1']) < int(info['VT2']) <= len(df)


def test_resting_none_generates_a_ramp_test():
    generator = utilities.load_generator()
    for seed in range(8):
        random.seed(seed)
        np.random.seed(seed)
        _, info = quiet(utilities.generate_CPET, generator, resting=None)
        assert info['resting'] == 'False'


def test_a_peak_vo2_of_exactly_4000_has_a_ramp(monkeypatch):
    real = np.random.uniform

    def uniform(low=0.0, high=1.0, size=None):
        return 4000.0 if (low, high) == (3000, 4000) else real(low, high, size)

    monkeypatch.setattr(np.random, 'uniform', uniform)
    random.seed(0)
    np.random.seed(0)
    _, info = quiet(utilities.generate_CPET, utilities.load_generator(), fitness_group=2)
    assert info['W0'] == '100'


@pytest.mark.parametrize('kwargs, match', [
    ({'resting': False, 'index': 0}, 'between 1 and 88'),
    ({'resting': False, 'index': 89}, 'between 1 and 88'),
    ({'resting': True, 'index': 51}, 'between 1 and 50'),
    ({'index': 12}, 'resting=True or resting=False'),
])
def test_draw_real_test_checks_its_index(kwargs, match):
    with pytest.raises(ValueError, match=match):
        utilities.draw_real_test(**kwargs)


def test_draw_real_test_draws_the_test_it_is_asked_for():
    random.seed(0)
    df, _ = quiet(utilities.draw_real_test, resting=False, index=12)
    assert len(df) == len(ramp(12))
