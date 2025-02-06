
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fusion_MIL import Fusion_MIL
from extract import get_acceleration, get_location

hours = 8

def generate_acceleration():
    sampling_rate = 10  # Hz
    duration = hours * 60 * 60  # 8 hours in seconds
    num_samples = duration * sampling_rate
    subject_id = "S001"

    # Generate timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i / sampling_rate) for i in range(num_samples)]

    # Generate acceleration data with gaps
    acc_x = np.random.normal(0, 1, num_samples)
    acc_y = np.random.normal(0, 1, num_samples)
    acc_z = np.random.normal(0, 1, num_samples)

    # Define probability weights for different gap sizes
    small_gap_size = np.arange(1, 60 * 10)  # Small gaps (1 - 60 seconds)
    large_gap_size = np.arange(60 * 10, 30 * 60 * 10)  # Large gaps (1 - 30 minute)
    small_gap_count = int(num_samples * 0.001)  # 1% of total samples will be small gaps
    large_gap_count = 5  # Introduce 5 large gaps

    # Generate small gaps
    P = 1. / np.arange(1, 60 * 10)
    P = P / np.sum(P)
    small_gaps = np.random.choice(range(num_samples), size=small_gap_count, replace=False)
    for idx in small_gaps:
        gap_size = np.random.choice(small_gap_size, p = P)
        end_idx = min(idx + gap_size, num_samples)
        acc_x[idx:end_idx] = np.nan
        acc_y[idx:end_idx] = np.nan
        acc_z[idx:end_idx] = np.nan

    # Generate large gaps
    large_gaps = np.random.choice(range(num_samples - max(large_gap_size)), size=large_gap_count, replace=False)
    for idx in large_gaps:
        gap_size = np.random.choice(large_gap_size)
        end_idx = min(idx + gap_size, num_samples)
        acc_x[idx:end_idx] = np.nan
        acc_y[idx:end_idx] = np.nan
        acc_z[idx:end_idx] = np.nan

    # Create DataFrame
    df = pd.DataFrame({
        "subject": subject_id,
        "timestamp": timestamps,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z
    })

    df = df.astype({'subject': str, 'timestamp': 'datetime64[ns]',
                    'acc_x': float, 'acc_y': float, 'acc_z': float})

    return df

def generate_location():
    duration = hours * 60  # 8 hours in minutes
    num_samples = duration
    subject_id = "S001"

    # Generate timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]

    # Generate random location data (latitude and longitude within a realistic range)
    latitudes = 37.0 + np.random.normal(0, 0.01, num_samples)  # Simulating small variations around a central location
    longitudes = -122.0 + np.random.normal(0, 0.01, num_samples)

    large_gap_size = np.arange(4, 30)  # Large gaps (4 - 30 minutes)
    large_gap_count = 10  # Introduce 10 large gaps

    # Generate large gaps
    large_gaps = np.random.choice(range(num_samples - max(large_gap_size)), size=large_gap_count, replace=False)
    for idx in large_gaps:
        gap_size = np.random.choice(large_gap_size)
        end_idx = min(idx + gap_size, num_samples)
        latitudes[idx:end_idx] = np.nan
        longitudes[idx:end_idx] = np.nan

    # Create DataFrame
    df = pd.DataFrame({
        "subject": subject_id,
        "timestamp": timestamps,
        "latitude": latitudes,
        "longitude": longitudes
    })

    df = df.astype({'subject': str, 'timestamp': 'datetime64[ns]',
                    'longitude': float, 'latitude': float})

    return df


def main():
    pd.set_option('display.max_rows', None)
    start = datetime.now()

    acc_df = get_acceleration(fs = 10, threshold = 2)
    loc_df = get_location(T = 60, threshold = 300)
    modes = fusion_mil((acc_df, loc_df), verbose = True)

    end = datetime.now()
    print(end - start)


fusion_mil = Fusion_MIL()
if __name__ == "__main__":
    main()




