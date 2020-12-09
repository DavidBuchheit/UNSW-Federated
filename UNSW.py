import nest_asyncio
import collections
import functools
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers, optimizers, backend as K

nest = tf.contrib.framework.nest

def fetch_minibatch(X, y, batch_size):
    n_points = X.shape[0]
    _, X_test, _, y_test = train_test_split(X, y, test_size=float(batch_size)/n_points)
    return (X_test, y_test)

np.random.seed(0)
nest_asyncio.apply()
np.random.seed(0)

federated_float_on_clients = tff.type_at_clients(tf.float32)
print(str(federated_float_on_clients.member))

print(str(federated_float_on_clients.placement))

federated_float_on_clients.all_equal
print(str(tff.type_at_clients(tf.float32, all_equal=True)))

simple_regression_model_type = (
    tff.StructType([('a', tf.float32), ('b', tf.float32)]))

print(str(simple_regression_model_type))

train_set = 'UNSW_NB15_training-set.csv'
test_set = 'UNSW_NB15_testing-set.csv'

training = pd.read_csv(train_set, index_col='id')
test = pd.read_csv(test_set, index_col='id')

data = pd.concat([training, test]) # Just merging the sets for now

NUM_CLIENTS = 10
BATCH_SIZE = 20
ROUNDS = 10

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


##Looking for attack_cat
labelt = data.drop("label", axis=1, inplace=True)
CATEGORICAL_COLUMNS = ['proto', 'service', 'state']
NUMERIC_COLUMNS = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
                   'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
                   'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                   'response_body_len', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
                   'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                   'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_sm_ips_ports']

for col in CATEGORICAL_COLUMNS:
    encode_text_dummy(data, col)

for col in NUMERIC_COLUMNS:
    encode_numeric_zscore(data, col)

x_columns = data.columns.drop('attack_cat')
x = data[x_columns].values
dummies = pd.get_dummies(data['attack_cat'])  # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values

TFF_data = np.array_split(data, NUM_CLIENTS)

def create_model(x, y):
    model = Sequential()
    model.add(Dense(15, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def model_fn():
  keras_model = create_model(x, y)
  return tff.learning.from_compiled_keras_model(keras_model, data)

iterative_process = tff.learning.build_federated_averaging_process(model_fn)

print(str(iterative_process.initialize.type_signature))

state = iterative_process.initialize()

for round_num in range(1, ROUNDS):
  state, metrics = iterative_process.next(state, data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

