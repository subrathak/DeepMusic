from datetime import datetime
from pymongo import MongoClient
import numpy as np
client = MongoClient()
db = client.streamer
dbUsers = db.subjects
dbAudios = db.audios
dbResults = db.results

userName = "" #subject name

subjectObj = dbUsers.find_one({"name":userName})

subjectAllResponses = dbResults.find({"subject":subjectObj['_id']})

segmentationData = {}

for response in subjectAllResponses:
	startTime = response['startTime']
	endTime = response['endTime']
	trackId =  response['track']
	track = dbAudios.find_one({"_id":trackId})
	trackName = track['name']
	segment = {'startTime':startTime,'endTime':endTime}
	segmentationData[trackName] = segment

# np.save('segmentationData.npy',segmentationData)

import pickle

with open('./Data/segmentationDataGuru.pickle', 'wb') as handle:
    pickle.dump(segmentationData, handle, protocol=pickle.HIGHEST_PROTOCOL)
