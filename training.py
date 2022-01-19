import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

import pandas as pd

DATASEP = 5

ACTIONS = ["left", "right"]
reshape = (-1, 8, 40)

def create_data(starting_dir="data"):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            #print(action, item)
            data = np.load(os.path.join(data_dir, item))
            for idx, item in enumerate(data):
            	# adding this for cushion between insample data.
            	if idx % DATASEP == 0:
                	training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        np.random.shuffle(training_data[action])
        training_data[action] = training_data[action][:min(lengths)]

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)
    # creating X, y
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:

            if action == "left":
                combined_data.append([data, [1, 0]])

            elif action == "right":
                combined_data.append([data, [0, 1]])

    np.random.shuffle(combined_data)
    print("length:",len(combined_data))
    return combined_data

print("creating training data")
traindata = create_data(starting_dir="data")
train_X = []
train_y = []
for X, y in traindata:
    train_X.append(X)
    train_y.append(y)

print("creating testing data")
distribution = 0.9

test_X = train_X[-int(len(train_X)*(1-distribution)):]
test_y = train_y[-int(len(train_y)*(1-distribution)):]

train_X = train_X[:int(len(train_X)*distribution)]
train_y = train_y[:int(len(train_y)*distribution)]

print(len(train_X))
print(len(test_X))

print(np.array(train_X).shape)


train_X = np.array(train_X).reshape(reshape)
test_X = np.array(test_X).reshape(reshape)

train_y = np.array(train_y)
test_y = np.array(test_y)


model = Sequential()

model.add(Conv1D(64, (5), padding='same', input_shape=train_X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv1D(128, (5), padding='same'))
model.add(Activation('relu'))

#model.add(Conv1D(256, (5), padding='same'))
#model.add(Activation('relu'))

#model.add(Conv1D(512, (5), padding='same'))
#model.add(Activation('relu'))

model.add(Conv1D(2, (8)))
model.add(Reshape((2,)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

epochs = 100
batch_size = 32

es = EarlyStopping(
    monitor='val_accuracy',
    min_delta = 0,
    patience = 10,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y), callbacks=es)
score = model.evaluate(test_X, test_y, batch_size=batch_size)

MODEL_NAME = f"models/model"
model.save(MODEL_NAME)
print("saved:")
print(MODEL_NAME)

res = pd.DataFrame(history.history)
plt.plot(res[['accuracy','val_accuracy']])
plt.show()