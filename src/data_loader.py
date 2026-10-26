"""Data loading module for AeroGuard AI Phase 1."""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    ALL_COLUMNS,
)


def load_dataset_file(
    file_path: Path,
    column_names: list[str] = ALL_COLUMNS
) -> pd.DataFrame:
    """
    Load a C-MAPSS whitespace-delimited dataset file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the text file.
    column_names : list[str]
        List of column names to assign to the DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with correctly typed columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at expected location: {file_path}")

    # C-MAPSS files are space-delimited with trailing/multiple spaces
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=column_names,
        engine="python"
    )

    # Enforce integer types for unit and cycle
    if "unit_number" in df.columns:
        df["unit_number"] = df["unit_number"].astype(int)
    if "cycle" in df.columns:
        df["cycle"] = df["cycle"].astype(int)

    return df


def load_rul_file(file_path: Path = RUL_FILE) -> pd.DataFrame:
    """
    Load ground-truth Remaining Useful Life (RUL) values for test engines.

    Parameters
    ----------
    file_path : Path
        Path to RUL_FD001.txt.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'unit_number' (1-indexed) and ground truth 'RUL'.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"RUL file not found at: {file_path}")

    df_rul = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python"
    )
    df_rul["unit_number"] = df_rul.index + 1
    # Reorder columns
    df_rul = df_rul[["unit_number", "RUL"]]
    df_rul["unit_number"] = df_rul["unit_number"].astype(int)
    df_rul["RUL"] = df_rul["RUL"].astype(int)
    return df_rul


def add_training_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and append Remaining Useful Life (RUL) for run-to-failure training data.

    In the training set, each engine runs until failure.
    For engine `u` at cycle `t`, RUL = max_cycle(u) - t.
    At the final cycle of each engine, RUL = 0.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame with 'unit_number' and 'cycle'.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with new 'RUL' column.
    """
    df = df_train.copy()
    max_cycles = df.groupby("unit_number")["cycle"].max().reset_index()
    max_cycles.rename(columns={"cycle": "max_cycle"}, inplace=True)
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def load_all_data(
    train_file: Path = TRAIN_FILE,
    test_file: Path = TEST_FILE,
    rul_file: Path = RUL_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load train, test, and test RUL datasets.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, test_df, rul_df)
    """
    train_df = load_dataset_file(train_file)
    test_df = load_dataset_file(test_file)
    rul_df = load_rul_file(rul_file)
    return train_df, test_df, rul_df
