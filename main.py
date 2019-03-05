import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
from scipy import sparse

x_train = sparse.load_npz("data/train_x.npz")
y_train = sparse.load_npz("data/train_y.npz")
INPUT_DIM = 3800

model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=INPUT_DIM))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(INPUT_DIM, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['top_k_categorical_accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=512)
model.save("learning-path-1.h5")
