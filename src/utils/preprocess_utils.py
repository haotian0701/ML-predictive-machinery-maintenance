from typing import Dict
import pandas as pd


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
