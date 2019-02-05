from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.optimizers import RMSprop
from numpy import genfromtxt
import csv
import numpy as np
import random
import sys
import pickle
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.signal
from keras import metrics

maxlen = 4#35#2#17#861
input_dim = 34
output_dim = 1
ims_freq = 172#172.26666666666668

# X = np.zeros((24005, maxlen, input_dim), dtype=np.float16)
# Y = np.zeros((24005, output_dim), dtype=np.int)

xT = np.zeros((0, maxlen, input_dim))
yT = np.zeros((0))

with open('/home/nomad/Documents/DeepMusic/VanicFeatureExtraction/featureDict1.pickle', 'rb') as handle:
    segmentFeatures = pickle.load(handle)

with open('./Data/segmentEngagementPranay.pickle', 'rb') as handle:
    segmentEngagement = pickle.load(handle)


xKeys = ['5out.wav', '4out.wav', '3out.wav', '2out.wav']

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def getSlices(x,y):
    sliceX = []
    sliceY = []
    cutoff = x.shape[1]-maxlen
    # x = x[:,:cutoff]
    y = y[:2]
    y = np.mean(y,axis=0)
    y = scipy.signal.resample(y,cutoff)

    for index,val in enumerate(y):
        if val == np.inf:
            y[index] = y[index-1]
    bro = y
    response = np.zeros(len(y))
    for i in range(maxlen-1,cutoff,1):
        print(i)
        seg = x[:,i:i+maxlen]
        sliceX.append(seg.T)
    
    return sliceX,y[-len(sliceX):]

j = 0
for key in xKeys:
    x = segmentFeatures[key]
    y = segmentEngagement[key]
    xS,yS = getSlices(x,y)
    xS = np.asarray(xS)
    xT = np.concatenate((xT,xS))
    yT = np.concatenate((yT,yS))


batch_size = yT.shape[0]
scaler = MinMaxScaler(feature_range=(0, 1))
yT = scaler.fit_transform(yT)

# trainX = xT[:2000]
# trainY = yT[:2000]

trainX = xT
trainY = yT

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)

batch_size = 1

model = Sequential()
model.add(LSTM(20, batch_input_shape=(batch_size,maxlen,input_dim)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=[metrics.sparse_categorical_accuracy])
model.fit(X_train,y_train,epochs=1,batch_size=batch_size)

# model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1, sample_weight=None)


xTest = segmentFeatures['1out.wav']
yTest = segmentEngagement['1out.wav']

xTest,yTest = getSlices(xTest,yTest)
xTest = np.asarray(xTest)
yTest = scaler.fit_transform(yTest)

# xTest = xTest[:500]
# yTest = yTest[:500]

preds = model.predict(xTest)
plt.figure()
plt.subplot(211)
preds = scaler.fit_transform(preds)
preds = movingaverage(preds[:,0],30)
plt.plot(preds)
plt.title("Preds")
plt.subplot(212)
plt.plot(movingaverage(yTest,30))
plt.title("Actual")
plt.show()

# # make predictions
# trainPredict = model.predict(X_train)
# testPredict = model.predict(X_test)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([y_train])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([y_test])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(y_train, trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(y_test, testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))


a = movingaverage(yTest,30)
b = preds

from scipy import spatial

res = 1- spatial.distance.cosine(a,b)
print(res)