def movingAverageFilter(data, numPoints, doKeepSameLength=False):
    if(len(data) < 5):
        return data
    result = []
    
    isMid = numPoints % 2

        
    buffer = int(numPoints/2)
    # print(buffer)
    # print(len(data)-buffer)
    for i in range(buffer, len(data) - buffer + int(not isMid)):
        result.append(sum(data[i-buffer: i+buffer+isMid])/numPoints)
        # print(data[i-buffer: i+buffer+isMid])

    return result


from scipy import signal
#Requires a 1-D array input data
#I don't know how to test if this works correctly
def bandpassFilter(data, fs, cutoff):
    #cutOff = [0.7, 4]
    #fs=256
    window='hamming'
    filt = signal.firwin(128, cutOff, window='hamming', pass_zero=False, scale=False, fs=fs)
    return signal.lfilter(filt, 1, data)    


from scipy import interpolate
#interpolate signal with cubuc spline at sampling freq of 256
#I am unsure if the 256 Hz req. ultimately means we need 256Hz of data points for timestamps
def interpolateData(timeStamps, data):
    return inerpolate.CubicSpline(timeStamps, data)

