#!/usr/bin/env

# Remote plehtysysmosngraphy
# David Haas, Hogan Pope, Spencer Mullinix

# science
from matplotlib.pyplot import tight_layout
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal
from sklearn.decomposition import FastICA
import cv2
import numpy as np
import dlib
import imutils


# utilities
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import glob
import re

# custom code
from filteringData import movingAverageFilter, bandpassFilter

class BVPExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')

    def parse_for_fs(self, session_folder):
        text = open(session_folder + "session.xml").read()
        return float(re.search(r'vidRate="(\S+)"', text).group(1))


    def sample_video(self, session_folder, draw=False):
        def get_video(folder_path): return glob.glob(f"{folder_path}*.avi")[0]
        
        video_stream = cv2.VideoCapture(get_video(session_folder))
        fs = self.parse_for_fs(session_folder)
        nframes = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        Y = np.zeros((nframes,3))  # Holds the data for b,g,r channels
        for i in trange(nframes):
            good, frame = video_stream.read()       
            Y[i,:] = self.get_face_sample(frame, draw=draw)

            if draw:
                cv2.imshow('Press Q to quit', frame)
                if cv2.waitKey(int(1)) & 0xFF == ord('q'):
                    print("Quitting")
                    exit()
            
        video_stream.release()
        cv2.destroyAllWindows()
        
        return Y, fs
    
    def point_inside_polygon(self,x,y,poly):
        n = len(poly)
        inside = False
        p2x = 0.0
        p2y = 0.0
        xints = 0.0
        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    def get_face_sample(self, image, draw=False, bbox_shrink=0.4):
        rects = self.detector(image, 1)

        if rects is None:
            print('No face detected')
            return False

        #Get the facial landmarks
        shape = self.predictor(image, rects[0])

        #Get the coords of the facial landmarks we care about

        #left cheek
        x = shape.part(0).x
        y = shape.part(0).y
        w = shape.part(50).x - x
        h = shape.part(50).y - y
        
        
        # Shrink bounding box to get only face skin
        x1l,y1l = int(x + w*bbox_shrink/2), int(y + h*bbox_shrink/2)
        x2l,y2l = int((x + w) - w*bbox_shrink/2), int((y+h) - h*bbox_shrink/2)

        #right cheek
        x = shape.part(16).x
        y = shape.part(16).y
        w = shape.part(52).x - x
        h = shape.part(52).y - y
        
        x1r,y1r = int(x + w*bbox_shrink/2), int(y + h*bbox_shrink/2)
        x2r,y2r = int((x + w) - w*bbox_shrink/2), int((y+h) - h*bbox_shrink/2)
        
        # Extract and filter bounding box data to one measurement per channel

        totals = [0, 0, 0]
        totalCnt = 0
        for i in range(y1l, y2l):
            for j in range(x1l, x2l):
                totals[0] += image[i][j][0]
                totals[1] += image[i][j][1]
                totals[2] += image[i][j][2]
                totalCnt += 1
        
        for i in range(y1r, y2r):
            for j in range(x1r, x2r):
                totals[0] += image[i][j][0]
                totals[1] += image[i][j][1]
                totals[2] += image[i][j][2]
                totalCnt += 1


        
        channel_averages = [totals[0]/totalCnt, totals[1]/totalCnt, totals[2]/totalCnt]
        #roi = image[y1:y2][x1:x2][:]
        #print(y1, y2, ";", x1,x2)
        #channel_averages = np.mean(roi, axis=(0,1))  # mean of each channel
        
        

        if draw:
            cv2.rectangle(image, (x1l,y1l), (x2l,y2l), (0, 0, 255), 2)
            cv2.rectangle(image, (x1r,y1r), (x2r,y2r), (0, 0, 255), 2)
            cv2.putText(image, f'blue signal: {round(channel_averages[0],2)}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(image, f'green signal: {round(channel_averages[1],2)}',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f'red signal: {round(channel_averages[2],2)}',
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


        return channel_averages


    def detrend_traces(self, channels, λ=10):
        K = channels.shape[0] - 1
        I = scipy.sparse.eye(K)
        D2 = scipy.sparse.spdiags((np.ones((K,1)) * [1,-2,1]).T ,[0,1,2], K-2, K)

        detrended = np.zeros((K, channels.shape[1]))
        for idx in range(channels.shape[1]):  # iterates thru each channel (b,g,r)
            #TODO I'm not sure if Poh et al used the difference array as z, like 
            # the did in the detrending paper... i dont think they do
            z = channels[:K,idx]
            # z = channels[1:,idx] - channels[:-1, idx]  # create the interval 

            term = scipy.sparse.csc_matrix(I + λ**2 * D2.T * D2)
            z_stationary = (I - scipy.sparse.linalg.inv(term)) * z
            detrended[:, idx] = z_stationary

        return detrended


    def z_normalize(self, data):
        return (data - data.mean(axis=0)) / data.std(axis=0)


    def ica_decomposition(self, data):
        ica = FastICA(n_components=data.shape[1])
        return ica.fit_transform(data)


    def select_component(self, components, fs):
        largest_psd_peak = -1e10
        best_component = None
        for i in range(components.shape[1]):
            x = components[:,i]
            f, psd = scipy.signal.periodogram(x, fs)
            if max(psd) > largest_psd_peak:
                largest_psd_peak = max(psd)
                best_component = x
            
            
        return best_component


    def get_BVP_signal(self, video_path, draw=False):
        Y, fs = self.sample_video(video_path, draw=draw)  # Get color samples from video
        # Y, fs = np.load('channel_data.npy'), 60.9708 
        
        detrended_data = self.detrend_traces(Y)
        cleaned_data = self.z_normalize(detrended_data)
        np.save('detrended_signals.npy', detrended_data)
        source_signals = self.ica_decomposition(cleaned_data)
        np.save('ica_signals.npy', source_signals)
        bvp_signal = self.select_component(source_signals, fs)

        return bvp_signal


    def find_heartrate(self, bvp_signal):
        # 1st paragraph of sec 3c from Poe et al.
        averaged = movingAverageFilter(bvp_signal,5)

        bp = bandpassFilter(averaged)
        np.save('filtered_bvp.npy', bp)
        plt.title("Bandpass filtered BVP signal")
        plt.plot(bp)
        plt.show()

def plot_figures():
    # load in data
    raw = np.load('channel_data.npy')
    detrended = np.load('detrended_signals.npy')
    ica = np.load('ica_signals.npy')
    bvp = np.load('bvp_signal.npy')

    # Plot raw color data
    rgb_fig, rgb_ax = plt.subplots(3,1, tight_layout={'pad': 1})
    rgb_fig.set_size_inches(8, 6)
    plt.subplots_adjust(wspace=None, hspace=1)

    for c, name in enumerate(['Blue', 'Green', 'Red']):  # plot each component
        rgb_ax[c].set_title(f"{name} channel")
        rgb_ax[c].plot(raw[:,c])

    # Plot Detrended Data
    det_fig, det_ax = plt.subplots(3,1, tight_layout={'pad': 1})
    det_fig.set_size_inches(8, 6)
    plt.subplots_adjust(wspace=None, hspace=1)

    for c, name in enumerate(['Blue', 'Green', 'Red']):  # plot each component
        det_ax[c].set_title(f"{name} signal - detrended")
        det_ax[c].plot(detrended[:,c])

    # Plot ICA
    ica_fig, ica_ax = plt.subplots(3,1, tight_layout={'pad': 1})
    ica_fig.set_size_inches(8, 6)
    plt.subplots_adjust(wspace=None, hspace=1)

    for c in range(ica.shape[1]):  # plot each component
        ica_ax[c].set_title(f"Component {c+1}")
        ica_ax[c].plot(ica[:,c])

    # Plot BVP
    bvp_fig = plt.figure(tight_layout={'pad': 1})
    bvp_fig.set_size_inches(8, 3)
    plt.plot(bvp)
    plt.title("Selected BVP Component")


    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', action="store", type=str)
    parser.add_argument('--draw', '-d', action="store_true", default=False)
    parser.add_argument('--plot', '-p', help="Plot figures from code", action="store_true", default=False)
    parser.add_argument('--hr', help="Calculate heart rate from bvp signal", action="store_true", default=False)
    args = parser.parse_args()

    exctractor = BVPExtractor()

    if args.plot:
        print("plotting figures...")
        plot_figures()
    elif args.hr:
        hr = exctractor.find_heartrate(np.load('bvp_signal.npy'))
    else:
        print("Running algorithm...")
        bvp_signal = exctractor.get_BVP_signal('Sessions/1/', draw=args.draw)
        np.save('bvp_signal.npy', bvp_signal)


if __name__ == "__main__":
    main()

