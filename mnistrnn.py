import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

#Normalize data
x_train = x_train/255.0
x_test = x_test/255.0

#Build the model itself
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu')) #Dense layer, 32 nodes
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax')) #10 = amount of classes

#Optimize

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6) #Shrink learning rate over time (decay)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)
#Train
model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))
