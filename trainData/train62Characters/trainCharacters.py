import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, MaxPool2D, Conv2D, Dense, Reshape, Dropout
from emnist import extract_training_samples
from emnist import extract_test_samples
from keras.preprocessing.image import ImageDataGenerator

#get data and re processing
data=pd.read_csv("charactersData.csv") 
data.rename(columns={'0':'label'}, inplace=True)
X = data.drop('label',axis = 1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X,y)


X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train=255-X_train
X_test=255-X_test
X_train=X_train/255
X_test=X_test/255

y_train = np_utils.to_categorical(y_train, 62)
y_test = np_utils.to_categorical(y_test, 62)
#model

model = Sequential()
model.add(Conv2D(32, 3, 3, activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2), strides = 2))
model.add(Conv2D(32, 3, 3, activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(62, activation = 'softmax'))
 


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=30, verbose=1)
scores = model.evaluate(X_test,y_test, verbose=0)



#********save model*****
model_yaml = model.to_yaml()
with open("model5.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model5.h5")
