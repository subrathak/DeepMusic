import numpy as np
import csv
import sys
path = sys.argv[1]
list_data = []
for line in csv.reader(open(path), delimiter=","):
	if line:
		list_data.append(line)
x = np.asarray(list_data)



electrodesLabel= ['3 ','7 ','9 ','12 ','16 ']
eegLength = x.shape[0]/5
electrodeWiseData = np.zeros([6,6, eegLength])

for index,electrode in enumerate(electrodesLabel):
	electrodes = []
	thetaValue = []
	alphaValue = []
	low_betaValue = []
	high_betaValue = []
	gammaValue = []
	time = []
	for row in x:
		if row[0] == electrode:
			thetaValue.append(row[1])
			alphaValue.append(row[2])
			low_betaValue.append(row[3])
			high_betaValue.append(row[4])
			gammaValue.append(row[5])
			time.append(row[6])
	thetaValue = np.around(np.asarray(thetaValue).astype(float),decimals=4)
	alphaValue = np.around(np.asarray(alphaValue).astype(float),decimals=4)
	low_betaValue = np.around(np.asarray(low_betaValue).astype(float),decimals=4)
	high_betaValue = np.around(np.asarray(high_betaValue).astype(float),decimals=4)
	gammaValue = np.around(np.asarray(gammaValue).astype(float),decimals=4)
	time = np.asarray(time).astype(float)
	allData = np.vstack((time,thetaValue,alphaValue,low_betaValue,high_betaValue,gammaValue))
	# allData = allData.tolist()
	electrodeWiseData[index] = allData
	# print len(allData)
	# electrodeWiseData = electrodeWiseData.append(allData)

allElectrodeEngagement = []
for vals in electrodeWiseData:
	alpha = vals[2]
	theta = vals[1]
	beta = vals[3]
	alphabeta = alpha + theta
	eng = beta/alphabeta
	where_are_NaNs = np.isnan(eng)
	eng[where_are_NaNs] = 0
	allElectrodeEngagement.append(eng)
allElectrodeEngagement = np.asarray(allElectrodeEngagement)
allElectrodeEngagement[5] = time

np.save('./Data/engagementVsTimeGuru.npy',allElectrodeEngagement)