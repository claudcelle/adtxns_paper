import os 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pickle
import json
from typing import Union,List

from adtxns_utils import *
from stats_utils import * 


END = pd.Timestamp('2021-06-15 05:00:30.887568') #*2021-06-15 05:00:30.887568
START = pd.Timestamp('2020-01-25 19:13:17.731529') #*2020-01-25 19:13:17.731529
FIRST_RENEWAL_TIME = 47318.40424 #*seconds from the first activation event in the reference dataset 
NUM_PERIODS = 20 #*free parameter, number of subperiods to divide the transaction record (both sarafu or synthetic data)



def load_metadata(
    folder_path: Union[str, Path] = "/home/claudio/tesi/sarafu/final_simulations/008",
) -> pd.DataFrame:
    """
    Load metadata JSON files that have a matching Parquet file in the given folder.

    Pairing rule:
        - For each `X.parquet`, the corresponding metadata file is `X_metadata.json`.

    Returns
    -------
    pd.DataFrame
        DataFrame built from all matched metadata JSON files, with list values
        converted to tuples so that all entries are hashable.
    """
    folder = Path(folder_path)

    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a valid directory")

    # Collect files in the directory
    files = list(folder.iterdir())

    # Map parquet base name -> Path("X.parquet")
    parquet_files = {
        f.stem: f
        for f in files
        if f.is_file() and f.suffix == ".parquet"
    }

    # Map base name X -> Path("X_metadata.json")
    json_files = {}
    for f in files:
        if f.is_file() and f.name.endswith("_metadata.json"):
            stem = f.stem  # e.g. "X_metadata"
            if stem.endswith("_metadata"):
                base = stem[: -len("_metadata")]  # -> "X"
                json_files[base] = f

    # Find common identifiers that have both parquet and json
    common_ids = parquet_files.keys() & json_files.keys()

    # Load metadata from JSON files for the pairs
    metadata_list = []
    for identifier in common_ids:
        json_path = json_files[identifier]
        with json_path.open() as f:
            metadata = json.load(f)
        metadata_list.append(metadata)

    # Build DataFrame
    metadata_df = pd.DataFrame(metadata_list)

    # Ensure all entries are hashable: convert lists to tuples
    metadata_df = metadata_df.applymap(
        lambda x: tuple(x) if isinstance(x, list) else x
    )

    metadata_df.sort_values(["s","burstiness"],inplace=True)
    metadata_df.reset_index(inplace=True,drop=True)

    metadata_df = metadata_df.reindex(columns=['s','burstiness','simulation_id','sprate_type', 'sprate_params', 'inbal_type', 'inbal_params',
        'activity_type', 'activity_params', 'attractivity_type',
        'attractivity_params', 'N', 'T', 'D',  'decimals', 'SIZE_SCALE',
        'LENGTH_SCALE', 'MEAN_IET', 'copula_type', 'copula_param',       ])

    return metadata_df


def process_single_simulation(
    row: pd.Series,
    folder_path: Union[str, Path] = '/home/claudio/tesi/sarafu/final_simulations/008',
    first_renewal_time: float = FIRST_RENEWAL_TIME,
    start: pd.Timestamp = START,
) -> dict:
    """
    Process a single simulation given its metadata row.

    Parameters
    ----------
    row : pd.Series
        One row from metadata_df (must contain 'simulation_id', 'burstiness', 's').
    folder_path : str or Path
        Folder where the parquet files live.
    first_renewal_time :
        Value to add when computing scaled_time (YOUR FIRST_RENEWAL_TIME).
    start :
        Reference datetime (YOUR START).

    Returns
    -------
    dict
        Simulation results dictionary (same structure as before).
    """
    folder = Path(folder_path)
    simulation_id = row["simulation_id"]

    # Load transaction data
    parquet_path = folder / f"{simulation_id}.parquet"
    print(parquet_path)  # keep the print if you still want the trace

    df = pd.read_parquet(parquet_path)
    df = df.astype(
        {
            "amount": "float",
            "source_bal": "float",
            "target_bal": "float",
        }
    )  # patch class.Decimal invalid division error



    # *Converting the timestamps into dates: to do so we consider the timestamp to be seconds elapsed from the reference date START (June 2020 ca.)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["scaled_time"] = df["timestamp"] - df["timestamp"].iloc[0] + first_renewal_time
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df["correct_date"] = pd.to_timedelta(df["scaled_time"], unit="s") + start

    # Amount / weight and fractions
    df = df.rename(columns={"amount": "weight"})
    df["frac_out"] = df["weight"] / (df["source_bal"] + df["weight"])
    df["frac_in"] = df["weight"] / (df["target_bal"] - df["weight"])
    df = df[df["weight"] >= 1]

    # Agents
    agents = basic_agents(df, how="inner")
    filtered_crid = agents.crid.unique()

    # Balances
    all_balances = balance_constructor_v2(df)
    all_balances["period"] = pd.cut(all_balances["date"], bins=20, labels=False)

    filtered_balances = all_balances.loc[
        (all_balances["crid"].isin(filtered_crid)) & (all_balances["weight"] >= 1)
    ]

    bal_pivot = pd.pivot_table(
        filtered_balances,
        index="crid",
        columns="period",
        values="balance",
        aggfunc="last",
    )
    bal_pivot = bal_pivot.ffill(axis=1)

    # sim_index is the original index of the row in metadata_df
    sim_dict = {
        "transactions": df,
        "agents": agents,
        "balances": all_balances,
        "filtered_balances": filtered_balances,
        "balance_pivot": bal_pivot,
        "metadata": row,
        "sim_index": row.name,
        "sim_id": row["simulation_id"],
        "b": row["burstiness"],
        "s": row["s"],
    }

    return sim_dict


def load_and_prepare_transactions(
    simulation_id: Union[int, str],
    folder_path: Union[str, Path],
    first_renewal_time: float,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load one simulation parquet and produce the same `df` you had before:
    - typed numeric columns
    - sorted by timestamp
    - scaled_time, date, correct_date
    - renamed amount -> weight
    - frac_out, frac_in
    - filtered to weight >= 1
    """
    folder = Path(folder_path)
    parquet_path = folder / f"{simulation_id}.parquet"
    print(parquet_path)

    df = pd.read_parquet(parquet_path)
    df = df.astype(
        {
            "amount": "float",
            "source_bal": "float",
            "target_bal": "float",
        }
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["scaled_time"] = df["timestamp"] - df["timestamp"].iloc[0] + first_renewal_time
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df["correct_date"] = pd.to_timedelta(df["scaled_time"], unit="s") + start

    df = df.rename(columns={"amount": "weight"})
    df["frac_out"] = df["weight"] / (df["source_bal"] + df["weight"])
    df["frac_in"] = df["weight"] / (df["target_bal"] - df["weight"])
    df = df[df["weight"] >= 1]

    return df

def build_balance_tables(df: pd.DataFrame, agents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given transactions df and agents, compute:
    - all_balances
    - filtered_balances
    - bal_pivot (ffilled)
    """
    filtered_crid = agents.crid.unique()

    all_balances = balance_constructor_v2(df)
    all_balances["period"] = pd.cut(all_balances["date"], bins=20, labels=False)

    filtered_balances = all_balances.loc[
        (all_balances["crid"].isin(filtered_crid))
        & (all_balances["weight"] >= 1)
    ]

    bal_pivot = pd.pivot_table(
        filtered_balances,
        index="crid",
        columns="period",
        values="balance",
        aggfunc="last",
    ).ffill(axis=1)

    return all_balances, filtered_balances, bal_pivot


def compute_inter_event_times(
    df: pd.DataFrame,
    start: pd.Timestamp,
    source_col: str = "source",
    time_col: str = "correct_date",
) -> List[float]:
    """
    Replicates the old 'temp' logic:

    time_from_start = (df.groupby(source).first().correct_date - START)
    For each source:
        diffs = group.correct_date.diff().fillna(time_from_start[source])
        iet.extend(diffs.dt.total_seconds())
    """
    # first event for each source, relative to START
    first_per_source = df.groupby(source_col).first()[time_col]
    time_from_start = first_per_source - start

    iet: List[float] = []
    for src, group in df.groupby(source_col):
        diffs = group[time_col].diff().fillna(time_from_start.loc[src])
        iet.extend(diffs.dt.total_seconds().values)

    return iet


def process_single_simulation(
    row: pd.Series,
    folder_path: Union[str, Path],
    first_renewal_time: float,
    start: pd.Timestamp,
    compute_iet_flag: bool = True,
) -> dict:
    """
    Process one metadata row into a simulation dict.
    Optionally compute IET and store as 'iet' (list of floats).
    """
    if not isinstance(row, pd.Series):
        raise TypeError("row must be a pandas Series (single metadata row).")

    simulation_id = row["simulation_id"]

    # 1) Transactions
    df = load_and_prepare_transactions(
        simulation_id=simulation_id,
        folder_path=folder_path,
        first_renewal_time=first_renewal_time,
        start=start,
    )

    # 2) Agents
    agents = basic_agents(df, how="inner")

    # 3) Balances
    all_balances, filtered_balances, bal_pivot = build_balance_tables(df, agents)

    # 4) IET (optional)
    iet = None
    if compute_iet_flag:
        iet = compute_inter_event_times(df, start=start)

    sim_dict = {
        "transactions": df,
        "agents": agents,
        "balances": all_balances,
        "filtered_balances": filtered_balances,
        "balance_pivot": bal_pivot,
        "metadata": row,
        "sim_index": row.name,
        "sim_id": simulation_id,
        "b": row["burstiness"],
        "s": row["s"],
        "iet": iet,  # list[float] or None
    }

    return sim_dict
