import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):
    #for each feature calc min and max in train set.
    #formula 12 from paper
    # Assuming train.shape = (num_timepoints, num_sensors)
    min_values = np.min(train, axis=0)  # min for each sensor
    max_values = np.max(train, axis=0)  # max for each sensor

    # Normalize training data
    train_normalized = (train - min_values) / (max_values - min_values)

    # Apply same normalization to test data
    test_normalized = (test - min_values) / (max_values - min_values)
    return train_normalized, test_normalized


# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()
    # print('before downsample', np_data.shape)

    d_data = np_data[:, :down_time_len * down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len * down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))

    d_data = d_data.transpose()

    # print('after downsample', d_data.shape, d_labels.shape)

    return d_data.tolist(), d_labels.tolist()


def main():
    train = pd.read_csv('/home/andreas/Thesis/datasets/WaDi/WADI.A1_9 Oct 2017/WADI_14days.csv',skiprows=4, index_col=0)
    test = pd.read_csv('/home/andreas/Thesis/datasets/WaDi_preprocessed/WADI.A1_9 Oct 2017/WADI_attackdata_labelled.csv', index_col=0)


    train = train.drop(train.columns[[0, 2]], axis=1)
    test = test.drop(train.columns[[0, 2]], axis=1)

    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)
    print(train.columns)
    print(test.columns)
    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    train_labels = np.zeros(len(train))
    test_labels = test.attack

    # train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    cols = [x[46:] for x in train.columns]  # remove column name prefixes
    train.columns = cols
    test.columns = cols

    x_train, x_test = norm(train.values, test.values)

    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]

    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns=train.columns)
    test_df = pd.DataFrame(d_test_x, columns=test.columns)

    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels

    #train_df = train_df.iloc[2160:]

    train_df.to_csv('./WADI_14days_downsampled.csv')
    test_df.to_csv('./WADI_attackdata_downsampled.csv')

    f = open('./list.txt', 'w')
    for col in train.columns:
        f.write(col + '\n')
    f.close()


if __name__ == '__main__':
    main()
