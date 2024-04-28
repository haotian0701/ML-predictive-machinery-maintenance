from typing import Dict, Iterator, List, Sequence
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def identify_future_failures_dict(
    df: pd.DataFrame,
    machine_example_id_col: str,
    hours_ahead: int = 168,
    datetime_col: str = 'datetime',
    machine_id_col: str = 'machineID',
) -> Dict[str, int]:
    """
    Purpose: Creates dictionary indicating if any failure (1 through 4) occurs
        within a specified number of hours for each machine_example_ID.
    :param df: pd.DataFrame containing machine data.
    :param hours_ahead: int representing the number of hours to look ahead for
        failures from the max datetime of each machine_example_ID.
    :param datetime_col: str representing the name of DTM column in df.
    :param machine_id_col: str representing the name of machine ID column in df.
    :return: Dict with 'machine_example_ID' as keys and binary values indicating
        future failures.
    """
    printed_machineIDs_set = set()

    # grouping by 'machine_example_ID' and findind the max DTM for each group;
    # max DTM for each group should be 23:00
    max_datetime_per_id = df.groupby(machine_example_id_col)[datetime_col].max()

    # obtain dict by iterating through each unique machine_example_ID
    future_failures_dict = dict()
    for machine_example_id, max_datetime in max_datetime_per_id.items():

        # calculate window for the specified number of hours after the max DTM
        window_start = max_datetime + pd.Timedelta(hours = 1)
        window_end = max_datetime + pd.Timedelta(hours = hours_ahead)

        # get machineID
        machineID = int(machine_example_id[1:4])

        if machineID not in printed_machineIDs_set:
            #print(machineID, end="   ")
            printed_machineIDs_set.add(machineID)

        filtered_df = df[
            # filter for machineID
            (df[machine_id_col] == machineID)

            # filter for rows within "look-ahead" window
            & (window_start <= df[datetime_col])
            & (df[datetime_col] <= window_end)

            # filter for rows with any failure
            & (
                (df['comp1_failure'] == 1)
                | (df['comp2_failure'] == 1)
                | (df['comp3_failure'] == 1)
                | (df['comp4_failure'] == 1)
            )
        ]

        # update dict for this machine_example_ID
        if len(filtered_df) > 0:
            future_failures_dict[machine_example_id] = 1
        else:
            future_failures_dict[machine_example_id] = 0

    return future_failures_dict


def aggregate_by_time_window(
    df: pd.DataFrame,
    time_window: int
) -> pd.DataFrame:
    """
    Purpose: Groups and aggregates the data based on the time windows passed in.
        Used in notebooks/create_LogReg_MLP_features.ipynb file.
    :param df: pd.DataFrame containing machine data.
    :param time_window: int representing the number of hours to group and aggregate the data into;
        choose from 12, 24, 48, 72.
    :return: pd.DataFrame with data grouped and aggregated by 'machine_example_ID'.
    """
    # check if time window is valid
    valid_time_windows = [12, 24, 48, 72]
    if time_window not in valid_time_windows:
        raise ValueError("Invalid time window. Please choose one of 12, 24, 48, 72.")

    # define helper function for range
    def _range(
        x: pd.Series
    ) -> float:
        """
        Purpose: Calculates range of values in a pandas Series.
        :param x: pd.Series representing a column of numeric data.
        :return: float representing the range (max minus min) of the series.
        """
        return np.max(x) - np.min(x)

    # define aggregation functions for each column
    agg_funcs = {
        'datetime': 'first',
        'machineID': 'first',
        'volt': [
            'mean', 'std', skew, kurtosis, 'min', 'max', _range
        ],
        'rotate': [
            'mean', 'std', skew, kurtosis, 'min', 'max', _range
        ],
        'pressure': [
            'mean', 'std', skew, kurtosis, 'min', 'max', _range
        ],
        'vibration': [
            'mean', 'std', skew, kurtosis, 'min', 'max', _range
        ],
        'comp1_maint': 'sum',
        'comp2_maint': 'sum',
        'comp3_maint': 'sum',
        'comp4_maint': 'sum',
        'error1': 'sum',
        'error2': 'sum',
        'error3': 'sum',
        'error4': 'sum',
        'error5': 'sum',
        'comp1_failure': 'sum',
        'comp2_failure': 'sum',
        'comp3_failure': 'sum',
        'comp4_failure': 'sum',
        'age': 'first',
        'model2': 'max',
        'model3': 'max',
        'model4': 'max'
    }

    # group by selected columns and aggregate using agg_funcs
    aggregated_df = df.groupby(
       by = f'{time_window}_machine_example_ID'
    ).agg(
       func = agg_funcs
    ).reset_index()

    return aggregated_df


def rename_first_column_aggregated_df(
    aggregated_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Purpose: Renames the first column of the input dataframe by removing the first
        3 characters. Used in notebooks/create_LogReg_MLP_features.ipynb file.
    :param aggregated_df: pd.DataFrame containing machine data.
    :return: DataFrame with the first column renamed.
    """
    # extract common part of column names;
    # assuming first column follows the pattern
    common_part = aggregated_df.columns[0][:3]

    # replace common part with an empty string
    new_column_name = aggregated_df.columns[0].replace(
        common_part, ''
    )

    # rename column
    aggregated_df.rename(
        columns = {
            aggregated_df.columns[0]: new_column_name
        },
        inplace = True
    )

    return aggregated_df


def remove_max_time_window(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Purpose: Removes the last time window from the input dataframe.
        Note: This is necessary because while the final time window
            does contain features, it does not have a subsequent time
            window from which to determine a label/outcome.
        Used in notebooks/create_LogReg_MLP_features.ipynb file.
    :param df: pd.DataFrame containing machine data.
    :return: pd.DataFrame with the last time window removed.
    """
    #print(f"- {format(len(df), ',')} total entries.")

    # extract machine ID and time window
    df[['machineID', 'time_window']] = df['machine_example_ID'].str.extract(
        pat = r'\((\d+),\s*(\d+)\)'
    )

    # convert machineID and time_window to integers
    df['machineID'] = df['machineID'].astype(int)
    df['time_window'] = df['time_window'].astype(int)

    # group by machine ID and find the maximum time window for each machine
    max_time_windows = df.groupby(
        by = 'machineID'
    )['time_window'].max()

    # filter out rows where time window matches the maximum time window for each machine
    filtered_df = df[
        ~df.apply(
            lambda row: row['time_window'] == max_time_windows[row['machineID']],
            axis = 1
        )
    ].copy()

    # drop machineID and time_window columns
    filtered_df.drop(
        columns = ['machineID', 'time_window'],
        inplace = True
    )

    #print(f"- {format(len(filtered_df), ',')} total entries after removal.")

    return filtered_df


def _chunker(
    seq: Sequence,
    size: int
) -> Iterator[Sequence]:
    """
    Purpose: Splits the sequence into chunks of given size.
    :param seq: Sequence representing the list, tuple, or other indexable collection of data.
    :param size: int representing the number of elements in each chunk.
    :return: An iterator over chunks of the original sequence.
    """
    return (
        seq[pos : pos+size] for pos in range(0, len(seq), size)
    )


def _custom_agg(
    sub_df: pd.DataFrame
) -> List[int | float]:
    """
    Purpose: Performs custom aggregation for a sub-dataframe, ensuring output is JSON serializable.
    :param sub_df: pd.DataFrame representing the grouped sub-dataframe for which aggregation is required.
    :return: List[int | float] representing aggregated metrics including means, maximums, sums, and 
        standard deviations, ensuring all values are JSON serializable.
    """
    aggregated_list = [
        float(sub_df['age'].mean()),
        int(sub_df['model2'].max()),
        int(sub_df['model3'].max()),
        int(sub_df['model4'].max()),
        float(sub_df['volt'].mean()),
        float(sub_df['volt'].std(ddof = 0)),  # ddof=0 for population std, ensuring result not NaN
        float(sub_df['rotate'].mean()),
        float(sub_df['rotate'].std(ddof = 0)),
        float(sub_df['pressure'].mean()),
        float(sub_df['pressure'].std(ddof = 0)),
        float(sub_df['vibration'].mean()),
        float(sub_df['vibration'].std(ddof = 0)),
        int(sub_df['comp1_maint'].sum()),
        int(sub_df['comp2_maint'].sum()),
        int(sub_df['comp3_maint'].sum()),
        int(sub_df['comp4_maint'].sum()),
        int(sub_df['error1'].sum()),
        int(sub_df['error2'].sum()),
        int(sub_df['error3'].sum()),
        int(sub_df['error4'].sum()),
        int(sub_df['error5'].sum()),
        int(sub_df['comp1_failure'].sum()),
        int(sub_df['comp2_failure'].sum()),
        int(sub_df['comp3_failure'].sum()),
        int(sub_df['comp4_failure'].sum()),
    ]
    # replace NaNs with zeros, assuming missing data can be treated as zero
    aggregated_list = [
        0 if np.isnan(x) else x for x in aggregated_list
    ]
    return aggregated_list


def generate_rnn_features_dict(
    df: pd.DataFrame,
    hour_feature_window: int,
    chunk_size: int = 1
) -> Dict[
    str,
    List[List[int | float]]
]:
    """
    Purpose: Generates a dictionary of features from the DataFrame based on chunk size and a
        specified feature window, suitable as RNN features.
    :param df: pd.DataFrame representing preprocessed machinery data.
    :param hour_feature_window: int representing the size of the time window in hours for which
        features are aggregated.
    :param chunk_size: int representing the number of rows in each chunk for which individual 
        feature aggregations are computed.
    :return: Dict[str, List[List[int | float]]] representing aggregated feature data organized by 
        machine example IDs, with each list containing aggregated data for corresponding chunks.
    """
    assert hour_feature_window % chunk_size == 0, \
        f"chunk_size ({chunk_size}) must evenly divide hour_feature_window ({hour_feature_window})."

    # obtain column names to use for this hour_feature_window
    machineID_time_window_col = f'{hour_feature_window}_machineID_time_window'
    order_in_time_window_col = f'{hour_feature_window}_order_in_time_window'
    machine_example_ID_col = f'{hour_feature_window}_machine_example_ID'
    step_ID_col	= f'{hour_feature_window}_step_ID'

    # sort DataFrame
    df.sort_values(
        by = [machine_example_ID_col, order_in_time_window_col],
        inplace = True
    )

    # create features_dict
    features_dict = dict()
    for machine_example_ID, group in df.groupby(machine_example_ID_col):

        # process in chunks
        aggregated_data_list = [
            _custom_agg(
                sub_df = group.iloc[chunk_indices]
            ) for chunk_indices in _chunker(
                seq = range(group.shape[0]),
                size = chunk_size
            )
        ]

        # store data
        features_dict[machine_example_ID] = aggregated_data_list

    return features_dict


