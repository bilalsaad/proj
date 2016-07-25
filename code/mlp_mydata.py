'''a simple deep NN on the MNIST dataset
'''

from __future__ import print_function

import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import load_data 

batch_size=128
nb_classes=73
nb_epoch=20
TRAIN_NUM=51853
TEST_NUM=22218
DIM=100*100

(X_train, y_train), (X_test, y_test) =load_data.get_data()
print(X_test)
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)

X_train=X_train.reshape(TRAIN_NUM,DIM)
X_test=X_test.reshape(TEST_NUM,DIM)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train /=255
X_test /=255
print('shape of an input tensor' , X_train[0].shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
#convert class vectors  to binary class matrices 
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)

model=Sequential()
model.add(Dense(512,input_shape=(DIM,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(73))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)
model.fit(X_train,Y_train,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True,verbose=2,
        validation_data=(X_test,Y_test))
score= model.evaluate(X_test,Y_test,
        show_accuracy=True,verbose=0)
#json_string = model.to_json()
#f = open('my_model_arch.json','w').write(json_string)
#model.save_weights('model_model_weights.h5') 
print('Test score:', score[0])
print('Test accuracy:', score[1])




