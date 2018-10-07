import numpy as np
import features
from keras.models import Sequential
from keras.layers import Dense
trainX,trainY,testX,testY = features.input_data('./data/ham','./data/spam',.1)
np.savetxt("./data/trainX.csv", trainX, delimiter=",")
np.savetxt("./data/trainY.csv", trainY, delimiter=",")
np.savetxt("./data/testX.csv", testX, delimiter=",")
np.savetxt("./data/testY.csv", testY, delimiter=",")



def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

print("loading training data")
trainX = csv_to_numpy_array("./data/trainX.csv", delimiter=",")
trainY = csv_to_numpy_array("./data/trainY.csv", delimiter=",")
trainY=trainY[:,0]
print("loading test data")
testX = csv_to_numpy_array("./data/testX.csv", delimiter=",")
testY = csv_to_numpy_array("./data/testY.csv", delimiter=",")
testY=testY[:,0]

model = Sequential()
model.add(Dense(12, input_dim=trainX.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=20, batch_size=256)

scores = model.evaluate(testX, testY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
