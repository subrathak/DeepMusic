from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import os
import pickle
dirNameFile = '/home/nomad/Documents/DeepMusic-Adarsh/VanicFeatureExtraction/fileSe.txt'
with open(dirNameFile) as f:
    lines = f.read().splitlines()

featureDict = {}
# source = '/home/nomad/Documents/pyAudioAnalysis/genres_wav/reggae'
# for root, dirs, filenames in os.walk(source):
#     for f in filenames:
#         print f
#         fullpath = os.path.join(source, f)
#         print fullpath
#         [Fs, x] = audioBasicIO.readAudioFile(fullpath)
#         # x = x[:,0]
#         F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
#         featureDict[f] = F


for file in lines:
	print(file)
	[Fs, x] = audioBasicIO.readAudioFile(file)
	x = x[:,1]
	F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
	featureDict[file] = F


with open('featureDictAllFilesTopCharts.pickle', 'wb') as handle:
	pickle.dump(featureDict, handle, protocol=pickle.HIGHEST_PROTOCOL)