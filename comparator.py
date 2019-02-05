import numpy as np
import pickle
from datetime import datetime, timedelta
import bisect

def timeComparator(start,end):
	startTimeObj = datetime_from_millis(start)
	endTimeObj = datetime_from_millis(end)



def datetime_from_millis(millis, epoch=datetime(1970, 1, 1)):
    """Return UTC time that corresponds to milliseconds since Epoch."""
    return epoch + timedelta(milliseconds=millis)


engTime = np.load('./Data/engagementVsTimeGuru.npy')
# segData = np.load('segmentationData.npy')
times = engTime[5]*1000
ser = []
for time in times:
	ser.append(datetime_from_millis(time))

segmentEngagement = {}

with open('./Data/segmentationDataGuru.pickle', 'rb') as handle:
    segmentationData = pickle.load(handle)


for track in segmentationData:
	startTime = segmentationData[track]['startTime']
	endTime = segmentationData[track]['endTime']
	startTime = np.float(startTime)
	endTime = np.float(endTime)
	startIndex = bisect.bisect_right(ser,datetime_from_millis(startTime))
	endIndex = bisect.bisect_right(ser,datetime_from_millis(endTime))
	segmentEngagement[track] = engTime[:,startIndex:endIndex]

with open('./Data/segmentEngagementGuru.pickle', 'wb') as handle:
    pickle.dump(segmentEngagement, handle, protocol=pickle.HIGHEST_PROTOCOL)

