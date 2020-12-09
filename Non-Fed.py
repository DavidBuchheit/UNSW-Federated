import os
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import io
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers, optimizers, backend as K


train_set = 'UNSW_NB15_training-set.csv'
test_set = 'UNSW_NB15_testing-set.csv'

training = pd.read_csv(train_set, index_col='id')
test = pd.read_csv(test_set, index_col='id')

data = pd.concat([training, test]) # Just merging the sets for now

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

data.dropna(inplace=True, axis=1)
print(data[0:5])

x_columns = data.columns.drop('attack_cat')
x = data[x_columns].values
dummies = pd.get_dummies(data['attack_cat'])  # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values

print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42)
# model = tf.keras.models.load_model('my_model.h5')

model = Sequential()
model.add(Dense(15, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(30, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
# model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.fit(x_train, y_train, validation_data=(x_test,y_test),callbacks=[monitor], verbose=2, epochs=50)

pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_eval = np.argmax(y_test,axis=1)
score = metrics.accuracy_score(y_eval, pred)
print("Validation score: {}".format(score))

