import os.path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.constants import g

acceleration_path = os.path.join(os.path.expanduser('~'), 'Documents/b4u/acceleration.json')
location_path = os.path.join(os.path.expanduser('~'), 'Documents/b4u/location.json')
import json

def get_datetime(entry, type='location'):
    if type == 'location':
        datetimes = pd.to_datetime(
            pd.Series(
                [int(ts["$numberLong"]) + int(offset["$numberLong"])
                 for ts, offset in zip(entry["elapsedRealTimeNanos"], entry["timeTimezoneOffset"])]),
                       unit="ms")

    elif type == 'acceleration':
        datetimes = pd.to_datetime(
            pd.Series(
                int(recording["timestamp"]["$numberLong"]) + 7200000 for recording in entry["logged_accelerometer_data"]
            ), unit="ms"
        )

    return datetimes

def get_acceleration(fs: float, threshold: float):
    with open(acceleration_path) as f:
        data = json.load(f)

    entries = [
        pd.DataFrame({
            "subject": entry["username"],
            "entry": e,
            "timestamp": get_datetime(entry, type="acceleration"),
            "acc_x": [recording["xValue"] for recording in entry["logged_accelerometer_data"]],
            "acc_y": [recording["yValue"] for recording in entry["logged_accelerometer_data"]],
            "acc_z": [recording["zValue"] for recording in entry["logged_accelerometer_data"]]
        })
        for e, entry in enumerate(data)
    ]

    data = pd.concat(entries)
    data = data.sort_values(by=['subject', 'entry', 'timestamp']).reset_index()

    data = resample(data, type = 'acceleration', fs = fs, threshold = threshold)

    return data


def get_location(T: float, threshold: float):
    with open(location_path, 'r') as f:
        data = json.load(f)

    entries = [
        pd.DataFrame({
            "subject": entry["username"],
            "entry": e,
            "timestamp": get_datetime(entry, type="location"),
            "latitude": entry["latitude"],
            "longitude": entry["longitude"]
        })
        for e, entry in enumerate(data)
    ]

    data = pd.concat(entries)
    data = data.sort_values(by=['subject', 'entry', 'timestamp'])

    data = resample(data, type = 'location', T = T, threshold = threshold)

    return data

def resample(df: pd.DataFrame, type: str, fs: float = 0,
             T: float = 0, threshold: float = 0, only_valid: bool = True) -> pd.DataFrame:

    if type == 'acceleration':
        features = df.columns[df.columns.str.contains('acc')]
        T = 1. / fs
        factor = g / 1000.
    elif type == 'location':
        features = df.columns[df.columns.str.contains('latitude|longitude')]
        factor = 1

    resampled = pd.DataFrame()

    for sub_id, sub_df in df.groupby("subject"):
        old_x = sub_df[features]
        old_x = old_x.interpolate(method="linear").values * factor
        old_t = sub_df["timestamp"].values.astype('m') / np.timedelta64(1, 's')

        new_t = np.arange(start = old_t[0], stop = old_t[-1], step = T)
        new_x = interp1d(old_t, old_x, kind = 'linear', axis = 0)(new_t)

        new_df = pd.DataFrame(new_x, columns = features)

        NaNs = sub_df.isna().any(axis = 1).values.astype(int)
        prev_NaNs = interp1d(old_t, NaNs, kind = 'previous', axis = 0)(new_t)
        next_NaNs = interp1d(old_t, NaNs, kind = 'next', axis = 0)(new_t)
        new_df['valid'] = True
        new_df.loc[(prev_NaNs == 1) | (next_NaNs == 1), 'valid'] = False

        prev_t = interp1d(old_t, old_t, kind = 'previous', axis = 0)(new_t)
        next_t = interp1d(old_t, old_t, kind = 'next', axis = 0)(new_t)
        new_df.loc[(abs(new_t - next_t) > threshold) | (abs(new_t - prev_t) > threshold),
        'valid'] = False

        new_df['timestamp'] = pd.to_datetime(new_t, unit='s')
        new_df['subject'] = sub_id

        if only_valid:
            new_df.loc[~new_df.valid, features] = np.nan
            new_df = new_df.drop(columns = 'valid')

        resampled = pd.concat([resampled, new_df], axis = 0, ignore_index = True)
        del new_df

    return resampled
