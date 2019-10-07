<img src="https://github.com/subrathak/DeepMusic/blob/master/engagement.png" alt="Engagement forecast sample" height="400">
# Retrain the model:
1. mongoToDataStripper.py : for a mentioned username in the code, gets the starting and ending times of all the tracks listened.

2.noiseCleaner.py : given eeg csv file created from krishna.py{emotive}, generates a numpy file containing engagement , can be 
modified to store all band powers.

3. comparator.py : takes the mongoToDataSStripper pickle and noiseCleaner numpy file and generates dictinary of track:engagement.

4. VanicFeatureExtracion : contains the extracted features of the tracks. 

Good Luck !
