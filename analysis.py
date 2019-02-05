import scipy.signal
import numpy as np
from keras.models import load_model
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers import Dense,Activation,Flatten
from keras.layers.pooling import MaxPooling1D,MaxPooling2D
from keras.models import Sequential
from keras.layers.recurrent import LSTM

maxlen = 4
np.random.seed(1337)

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

# x = np.load('xForGuru.npy')
# y = np.load('yForGuru.npy')
newX = np.load('bossLevelX.npy')
newY = np.load('bossLevelY.npy')
# x_test = np.load('xForGuruRaw.npy')
# y_test = np.load('yForGuruRaw.npy')

# (newX,newY) = getSlices(xraw,yraw)
# (x_test,y_test) = getSlices(x_test,y_test)
newX = np.array(newX)
# x_test = np.array(x_test)
x_test = newX[14832:]
y_test = newY[14832:]
newX = np.reshape(newX,(len(newY),4,34,1))
x_test = np.reshape(x_test,(len(y_test),4,34,1))
newX = np.mean(newX,axis=1)
x_test = np.mean(x_test,axis=1)

trainMean = np.mean(newX,axis=0)
trainStd = np.std(newX,axis=0)

testMean = np.mean(x_test,axis=0)
testStd = np.std(x_test,axis=0)

#Normalization
for i,vec in enumerate(newX):
    newX[i] = (vec-trainMean)/trainStd

for i,vec in enumerate(x_test):
    x_test[i] = (vec-testMean)/testStd


model = Sequential()
# model.add(LSTM(32,input_shape=(4,34)))
# model.add(Reshape())
model.add(Conv1D(128,2,strides=1,input_shape=(34,1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(2,strides=None))
model.add(Conv1D(128,2,strides=1))
model.add(Activation('relu'))
model.add(MaxPooling1D(2,strides=None))
model.add(Conv1D(128,2,strides=1))
model.add(Activation('relu'))
model.add(MaxPooling1D(2,strides=None))
model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('tanh'))

optimizers = ['adagrad','rmsprop','adadelta','adam','']

model.compile(optimizer='nadam',loss='mse')

model.summary()

# x_train = newX[:3836]
# x_test = newX[3836:]
# y_train = newY[:3836]
# y_test = newY[3836:]

model.fit(newX[:14832],newY[:14832],epochs=14,batch_size=256,validation_data=(x_test,y_test))
score = model.evaluate(x_test,y_test,batch_size=256)
print('score')