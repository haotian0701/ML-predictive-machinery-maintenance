from typing import Dict, Iterator, List, Sequence
import pandas as pd
import numpy as np


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


def generate_features_dict(
    df: pd.DataFrame,
    hour_feature_window: int,
    chunk_size: int = 1
) -> Dict[
    str,
    List[List[int | float]]
]:
    """
    Generates a dictionary of features from the DataFrame based on chunk size.
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
