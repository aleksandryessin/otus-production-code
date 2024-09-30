import pandas as pd
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, split_data


def test_load_data() -> None:
    df = load_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (150, 5)
    assert df['target'].nunique() == 3


def test_split_data() -> None:
    df = load_data()
    train, test = split_data(df)
    assert train is not None
    assert test is not None
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert train.shape == (120, 5)
    assert test.shape == (30, 5)
