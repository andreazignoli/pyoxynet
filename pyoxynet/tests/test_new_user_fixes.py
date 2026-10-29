"""One test per finding of the new-user walkthrough of 1.0.1, fixed in 1.0.2."""
import contextlib
import io
import random

import numpy as np
import pandas as pd
import pytest

import pyoxynet
from pyoxynet import utilities

_cnn = []


def cnn():
    if not _cnn:
        _cnn.append(utilities.load_model('CNN'))
    return _cnn[0]


def thresholds(df):
    _, out = utilities.test_pyoxynet(tf_model=cnn(), input_df=df)
    return out['VT1']['time'], out['VT2']['time']


def example():
    return utilities.load_csv_data()


# --- input that used to mislead

def test_an_empty_table_raises_instead_of_using_the_example():
    with pytest.raises(ValueError, match='no rows'):
        utilities.test_pyoxynet(tf_model=cnn(), input_df=example().iloc[0:0])


@pytest.mark.parametrize('bad', ['---', '', 'n/a'])
def test_a_non_numeric_cell_in_a_signal_is_treated_as_a_gap(bad):
    df = example()
    df['VO2_I'] = df['VO2_I'].astype(object)
    df.loc[100, 'VO2_I'] = bad
    vt1, vt2 = thresholds(df)
    assert abs(vt1 - 372) <= 2 and abs(vt2 - 550) <= 2


def test_a_units_row_under_the_header_is_dropped(tmp_path):
    cols = ['time', 'VO2_I', 'VCO2_I', 'VE_I', 'PetO2_I', 'PetCO2_I', 'HR_I']
    df = example()[cols]
    units = pd.DataFrame([['s', 'mL/min', 'mL/min', 'L/min', 'mmHg', 'mmHg', 'bpm']], columns=cols)
    pd.concat([units, df]).to_csv(tmp_path / 'export.csv', index=False)
    assert thresholds(pd.read_csv(tmp_path / 'export.csv')) == (372, 550)


def test_time_in_milliseconds_is_caught():
    df = example()
    df['time'] = df['time'] * 1000
    with pytest.raises(ValueError, match='milliseconds'):
        utilities.test_pyoxynet(tf_model=cnn(), input_df=df)


def test_time_in_minutes_gets_a_hint():
    df = example()
    df['time'] = df['time'] / 60
    with pytest.raises(ValueError, match=r'Is time in seconds\?'):
        utilities.test_pyoxynet(tf_model=cnn(), input_df=df)


def test_time_as_mm_ss_points_at_get_sec():
    df = example()
    df['time'] = [f'{int(s) // 60}:{int(s) % 60:02d}' for s in df['time']]
    with pytest.raises(ValueError, match='get_sec'):
        utilities.test_pyoxynet(tf_model=cnn(), input_df=df)
    df['time'] = df['time'].map(pyoxynet.get_sec)
    assert thresholds(df) == (372, 550)


def test_the_callers_table_is_not_changed():
    df = example()
    df['VO2_I'] = df['VO2_I'].astype(object)
    df.loc[5, 'VO2_I'] = '---'
    before = df.copy()
    thresholds(df)
    pd.testing.assert_frame_equal(df, before)


def test_load_csv_data_reads_a_local_data_test_csv(tmp_path, monkeypatch):
    example().iloc[:100].to_csv(tmp_path / 'data_test.csv', index=False)
    monkeypatch.chdir(tmp_path)
    assert len(utilities.load_csv_data('data_test.csv')) == 100
    assert len(utilities.load_csv_data()) == len(example())


# --- generation and the example tests

def test_labels_are_numbers():
    random.seed(0)
    np.random.seed(0)
    df, info = utilities.generate_CPET(utilities.load_generator())
    for key in ('VT1', 'VT2', 'VO2VT1', 'VO2VT2', 'VO2max', 'LT_vo2max', 'RCP_vo2max', 'W0', 'W/min'):
        assert isinstance(info[key], int), key
    assert info['resting'] is False and isinstance(info['noise_factor'], float)
    _, out = utilities.test_pyoxynet(tf_model=cnn(), input_df=df)
    assert isinstance(out['VT1']['time'] - info['VT1'], int)

    random.seed(0)
    _, info = utilities.draw_real_test(resting=True, index=3)
    for key in ('VT1', 'VT2', 'VO2VT1', 'VO2VT2', 'VO2max', 'Age', 'Weight'):
        assert isinstance(info[key], int), key
    assert info['resting'] is True and info['noise_factor'] is None


def test_generation_and_drawing_are_quiet_unless_asked(capsys):
    random.seed(1)
    np.random.seed(1)
    utilities.generate_CPET(utilities.load_generator())
    utilities.draw_real_test(resting=False, index=3)
    assert capsys.readouterr().out == ''

    utilities.generate_CPET(utilities.load_generator(), verbose=True)
    utilities.draw_real_test(resting=False, index=3, verbose=True)
    printed = capsys.readouterr().out
    assert 'VT1' in printed and 'maximal RER' in printed and 'RER:' not in printed
    assert printed.count('VT2') == 2


def test_draw_real_test_without_noise_is_the_recording():
    import pkgutil
    from io import StringIO
    raw = pd.read_csv(StringIO(pkgutil.get_data('pyoxynet.data_test', 'ramp_real_test_12.csv').decode()))
    df, _ = utilities.draw_real_test(resting=False, index=12, noise=False)
    np.testing.assert_array_equal(df['VO2_I'].to_numpy(), raw['VO2_I'].to_numpy())


# --- the package's surface

def test_import_exposes_only_pyoxynet_names():
    for name in ('np', 'pd', 'csv', 'scipy', 'read_csv'):
        assert not hasattr(pyoxynet, name), name
    for name in utilities.__all__ + ['Test', 'utilities', 'testing', '__version__']:
        assert hasattr(pyoxynet, name), name


def test_help_for_test_pyoxynet_says_what_it_raises():
    doc = utilities.test_pyoxynet.__doc__
    assert 'Raises' in doc and 'not calibrated' in doc and 'deafult' not in doc
