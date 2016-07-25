import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import load_data 

batch_size = 128
nb_classes = 68
nb_epoch = 20
DIM = 100 * 100
path_to_train_data = '/media/pics/train/*'
path_to_test_data = '/media/pics/test/*'
(X_train, y_train), (X_test, y_test) = load_data.get_printed_data(
        path_to_train_data, path_to_test_data)
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
print X_train.shape
print X_test.shape
X_train=X_train.reshape(X_train.shape[0], DIM)
X_test=X_test.reshape(X_test.shape[0], DIM)
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
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)
model.fit(X_train,Y_train,
        batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True,verbose=2,
        validation_data=(X_test,Y_test))
score= model.evaluate(X_test,Y_test,
        show_accuracy=True,verbose=0)
