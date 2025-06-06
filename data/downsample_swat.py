import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import MinMaxScaler


def norm(train, test):
    #for each feature calc min and max in train set.
    #formula 12 from paper
    #return
    # Assuming train.shape = (num_timepoints, num_sensors)
    min_values = np.min(train, axis=0)  # min for each sensor
    max_values = np.max(train, axis=0)  # max for each sensor

    # Normalize training data
    train_normalized = (train - min_values) / (max_values - min_values)

    # Apply same normalization to test data
    test_normalized = (test - min_values) / (max_values - min_values)
    return train_normalized, test_normalized

# downsample by 10
def downsample(data, labels, timestamps, down_len):
    # Convert data and labels to numpy arrays
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    # Process numeric data
    np_data = np_data.transpose()
    d_data = np_data[:, :down_time_len * down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    # Process labels
    d_labels = np_labels[:down_time_len * down_len].reshape(-1, down_len)
    # If any anomaly exists in window, mark the downsampled point as anomalous
    d_labels = np.round(np.max(d_labels, axis=1))

    # Process timestamps separately
    processed_timestamps = []
    for i in range(down_time_len):
        # Get the middle timestamp for each window
        window_timestamps = timestamps[i * down_len:(i + 1) * down_len]
        middle_idx = len(window_timestamps) // 2
        processed_timestamps.append(window_timestamps.iloc[middle_idx])

    # Transpose data back to original format
    d_data = d_data.transpose()

    return d_data.tolist(), d_labels.tolist(), processed_timestamps


def main():
    # Correctly load the normal and attack datasets
    train_orig = pd.read_csv('~/baselines/GTA/data/raw_data/swat_train.csv',header=0)
    test_orig = pd.read_csv('~/baselines/GTA/data/raw_data/swat_test.csv',header=0)

    print(train_orig.shape)
    print(test_orig.shape)
    train = train_orig
    test = test_orig

    # Fill missing values
    # train_mean = train.mean()
    # test_mean = test.mean()
    #
    # train = train.fillna(train_mean)
    # test = test.fillna(test_mean)
    #
    # train = train.fillna(0)
    # test = test.fillna(0)
    print(train_orig.shape)
    print(test_orig.shape)

    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    train_labels = train.attack
    test_labels = test.attack
    train_timestamps = train_orig[' Timestamp']
    test_timestamps = test_orig[' Timestamp']
    # Drop non-feature columns and attack labels
    train = train_orig.drop(columns=[' Timestamp', 'attack'])
    test = test_orig.drop(columns=[' Timestamp', 'attack'])

    # Normalize data
    #x_train, x_test = norm(train.values, test.values)

    # for i, col in enumerate(train.columns):
    #     train.loc[:, col] = x_train[:, i]
    #     test.loc[:, col] = x_test[:, i]

    # Perform downsampling with timestamps
    d_train_x, d_train_labels, d_train_timestamps = downsample(train.values, train_labels, train_timestamps, 10)
    d_test_x, d_test_labels, d_test_timestamps = downsample(test.values, test_labels, test_timestamps, 10)

    print(train_orig.shape)
    print(test_orig.shape)

    # Create dataframes with the downsampled data
    train_df = pd.DataFrame(d_train_x, columns=train.columns)
    test_df = pd.DataFrame(d_test_x, columns=test.columns)

    # Add labels
    train_df['Normal/Attack'] = d_train_labels
    test_df['Normal/Attack'] = d_test_labels

    # Add timestamps if needed for GTA
    train_df[' Timestamp'] = d_train_timestamps
    test_df[' Timestamp'] = d_test_timestamps

    # Remove the first 2160 samples from training data (system stabilization period)
    # Uncomment if needed for GTA
    # train_df = train_df.iloc[2160:]
    print(train_orig.shape)
    print(test_orig.shape)

    #remove redundant columns in test set.
    # common_columns = [col for col in test_df.columns if col in train_df.columns]
    # train_df = train_df[common_columns]
    # test_df = test_df[common_columns]

    # Save processed data
    train_df.to_csv('~/baselines/GTA/data/SWaT_normaldata_downsampled1.csv', index=False)
    test_df.to_csv('~/baselines/GTA/data/SWaT_attackdata_downsampled1.csv', index=False)

    # Save feature list
    with open('./list.txt', 'w') as f:
        for col in train.columns:
            f.write(col + '\n')

if __name__ == '__main__':
    main()
