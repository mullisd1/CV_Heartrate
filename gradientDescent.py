from bvp import BVPExtractor

files = ["channel_data/1_55BPM.mp4channels.pkl", "channel_data/2_67BPM.mp4channels.pkl", "channel_data/3_56BPM.mp4channels.pkl", "channel_data/4_78BPM.mp4channels.pkl", "channel_data/5_75BPM.mp4channels.pkl", "channel_data/6_59BPM.mp4channels.pkl", "channel_data/7_54BPM.mp4channels.pkl"] 
correct = [55, 67, 56, 78, 75, 59, 54]
step = 0.1

#4 parameters we need to check direction of [++++, +++-, ++-+, ++--, +-++, +-+-, +--+, +---, -+++, -++-, -+-+, -+--, --++, --+-, ---+, ----]
stepArrangements = [[1,1,1,1], [1,1,1,-1], [1,1,-1,1], [1,1,-1,-1], [1,-1,1,1], [1,-1,1,-1], [1,-1,-1,1], [1,-1,-1,-1], [-1,1,1,1], [-1,1,1,-1], [-1,1,-1,1], [-1,1,-1,-1], [-1,-1,1,1], [-1,-1,1,-1], [-1,-1,-1,1], [-1,-1,-1,-1]]

#lambda, window, cutLow, cutHigh
values = [10, 5, .7, 2]

bestSteps = [0, 0, 0, 0]
bestError = float('Inf')
keepGoing = True
while(keepGoing):
    values[0] += bestSteps[0]
    values[1] += bestSteps[1]
    values[2] += bestSteps[2]
    values[3] += bestSteps[3]

    keepGoing = False
    
    for arrangement in stepArrangements:
        found = []
        exctractor = BVPExtractor(values[0] + step*arrangement[0], values[1] + step*arrangement[1], [values[2] + step*arrangement[2], values[3] + step*arrangement[3]])
        
        for file in files:
            bvp, fs = exctractor.get_BVP_signal(None, False, file)
            found.push(find_heartrate(bvp, fs)['bpm'])
            
        tempError = np.sqrt(((np.array(found) - np.array(correct)) ** 2).mean())
        if(tempError < bestError):
            keepGoing = True
            bestError = tempError
            bestSteps = [step*arrangement[0], step*arrangement[1], step*arrangement[2], step*arrangement[3]]
        
    
    #-----
    
print("Values", values)
print("Error", bestError)