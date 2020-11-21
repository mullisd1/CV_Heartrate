def movingAverageFilter(data, numPoints, doKeepSameLength=False):
    if(len(data) < 5):
        return data
    result = []
    
    isMid = numPoints % 2

        
    buffer = int(numPoints/2)
#     print(buffer)
#     print(len(data)-buffer)
    for i in range(buffer, len(data) - buffer + int(not isMid)):
#         print(i)
#         print(buffer)
#         print(i-buffer)
#         print(int(i+buffer+isMid))
#         print(data[i-buffer: int(i+buffer+isMid)])
        result.append(sum(data[i-buffer: int(i+buffer+isMid)])/numPoints)
        # print(data[i-buffer: i+buffer+isMid])

    return result


from scipy import signal
#Requires a 1-D array input data
#I don't know how to test if this works correctly
def bandpassFilter(data, fs, cutOff=[0.7, 0.4], window='hamming'):    
    filt = signal.firwin(128, cutOff, window=window, pass_zero=False, scale=False, fs=fs)
    return signal.lfilter(filt, 1, data)    


from scipy import interpolate
#interpolate signal with cubuc spline at sampling freq of 256
#I am unsure if the 256 Hz req. ultimately means we need 256Hz of data points for timestamps
def interpolateData(timeStamps, data):
    return interpolate.CubicSpline(timeStamps, data)

