import numpy as np
from keras.models import Sequential, model_from_json
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import load_data 

model=model_from_json(open('my_model_arch.json').read())
print 'foooo'


model.load_weights('model_model_weights.h5')




TRAIN_NUM=51853
TEST_NUM=22218
DIM=10000
label_sizes=73
def gen_label(prob_vector):
    label = np.argmax(prob_vector)
    return label
(X_train, y_train), (X_test, y_test) =load_data.get_data()
X_test=np.asarray(X_test)
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
X_train=X_train.reshape(TRAIN_NUM,DIM)
X_test=X_test.reshape(TEST_NUM,DIM)

preds = model.predict_proba(X_test[:10],batch_size=1,verbose=2)
preds2 = model.predict_proba(X_test[-10:],batch_size=1,verbose=2)
preds = [gen_label(i) for i in preds]
preds2 = [gen_label(i) for i in preds2]

print preds, preds2

