from bvp import BVPExtractor
import numpy as np

import warnings
warnings.simplefilter("ignore")
 
files = ["channel_data/1_55BPM.mp4channels.pkl", "channel_data/2_67BPM.mp4channels.pkl", "channel_data/3_56BPM.mp4channels.pkl", "channel_data/4_78BPM.mp4channels.pkl", "channel_data/6_59BPM.mp4channels.pkl"] 
correct = [55, 67, 56, 78, 59]
step = 0.1

#4 parameters we need to check direction of [++++, +++-, ++-+, ++--, +-++, +-+-, +--+, +---, -+++, -++-, -+-+, -+--, --++, --+-, ---+, ----]
stepArrangements = [[1,1,1,1], [1,1,1,-1], [1,1,-1,1], [1,1,-1,-1], [1,-1,1,1], [1,-1,1,-1], [1,-1,-1,1], [1,-1,-1,-1], [-1,1,1,1], [-1,1,1,-1], [-1,1,-1,1], [-1,1,-1,-1], [-1,-1,1,1], [-1,-1,1,-1], [-1,-1,-1,1], [-1,-1,-1,-1]]

#lambda, window, cutLow, cutHigh
values = [299.9, 4.8, .5, 3]

bestSteps = [0, 0, 0, 0]
bestError = float('Inf')
keepGoing = True
while(keepGoing):
    
    values[0] += bestSteps[0]
    values[1] += bestSteps[1]
    values[2] += bestSteps[2]
    values[3] += bestSteps[3]
    print(values)
    print(bestError)

    keepGoing = False
    
    for arrangement in stepArrangements:
        found = []
        exctractor = BVPExtractor(values[0] + step*arrangement[0], values[1] + step*arrangement[1], [values[2] + step*arrangement[2], values[3] + step*arrangement[3]])
        
        for file in files:
            bvp, fs = exctractor.get_BVP_signal(None, False, file)
            found.append(exctractor.find_heartrate(bvp, fs)['bpm'])
            
        tempError = np.sqrt((np.square(np.array(found) - np.array(correct))).mean())
        if(tempError < bestError):
            keepGoing = True
            bestError = tempError
            bestSteps = [step*arrangement[0], step*arrangement[1], step*arrangement[2], step*arrangement[3]]
        
    
    #-----
    
print("Values", values)
print("Error", bestError)