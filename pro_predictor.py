import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import scipy.signal
from sklearn.model_selection import train_test_split

def getSlices(x,y):
    
  maxlen = 4
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

X = np.load('xForGuruRaw.npy')
Y = np.load('yForGuruRaw.npy')
new_X, new_Y = getSlices(X,Y)
final_X = []
for x in new_X:
        final_X.append(np.mean(x,axis=0))
final_X = np.asarray(final_X)
processed_X = preprocessing.scale(final_X)
X_train, X_test, y_train, y_test = train_test_split(
        processed_X, new_Y, test_size=0.33, random_state=42)
exported_pipeline = KNeighborsRegressor(leaf_size=30, metric='minkowski', p=2, n_jobs=-1, n_neighbors=1, weights="uniform")
exported_pipeline.fit(X_train, y_train)
print 'R^2 for Test Set:' + str(exported_pipeline.score(X_test, y_test))
print 'MSE for Test Set:' + str(mean_squared_error(exported_pipeline.predict(X_test), y_test))
