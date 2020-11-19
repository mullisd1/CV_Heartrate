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
import heartpy


# utilities
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import glob
import re
import os
import pickle

# custom code
from filteringData import movingAverageFilter, bandpassFilter

class BVPExtractor:
    def __init__(self, smoothing, avg_window=5, freq_cutoff=[0.7,3]):
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.coords = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.smoothing = smoothing
        self.avg_window_size = avg_window
        self.freq_cutoff = freq_cutoff
        
    
    def get_video(self, path):
        if os.path.isdir(path):
            path = glob.glob(f"{path}*.avi")[0]
        return cv2.VideoCapture(path)


    def sample_video(self, video_path, draw=False):
        if os.path.isdir(video_path):
            video_path = glob.glob(f"{video_path}*.avi")[0]
        
        video_stream = cv2.VideoCapture(video_path)
        fs = video_stream.get(cv2.CAP_PROP_FPS)
        print(fs)

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

    def get_face_sample(self, image, draw=False, bbox_shrink=0.5):
        # if(self.coords == None):
        rects = self.detector(image, 1)

        if rects is None:
            print('No face detected')
            return False

        #Get the facial landmarks
        shape = self.predictor(image, rects[0])

        #Get the coords of the facial landmarks we care about

        #left cheek - 0 - 50
        
        #Main face
        x = shape.part(1).x
        y = shape.part(1).y
        w = shape.part(13).x - x
        h = shape.part(13).y - y
        
        
        # Shrink bounding box to get only face skin
        x1l,y1l = int(x + w*bbox_shrink/2), int(y + h*bbox_shrink/2)
        x2l,y2l = int((x + w) - w*bbox_shrink/2), int((y+h) - h*bbox_shrink/2)
        
        self.coords = [x1l, y1l, x2l, y2l]

        #right cheek
        #x = shape.part(16).x
        #y = shape.part(16).y
        #w = shape.part(52).x - x
        #h = shape.part(52).y - y
        
        #x1r,y1r = int(x + w*bbox_shrink/2), int(y + h*bbox_shrink/2)
        #x2r,y2r = int((x + w) - w*bbox_shrink/2), int((y+h) - h*bbox_shrink/2)
        
        # Extract and filter bounding box data to one measurement per channel

        totals = [0, 0, 0]
        totalCnt = 0
        for i in range(self.coords[1], self.coords[3]):
            for j in range(self.coords[0], self.coords[2]):
                totals[0] += image[i][j][0]
                totals[1] += image[i][j][1]
                totals[2] += image[i][j][2]
                totalCnt += 1
        
        #for i in range(y1r, y2r):
        #    for j in range(x1r, x2r):
        #        totals[0] += image[i][j][0]
        #        totals[1] += image[i][j][1]
        #        totals[2] += image[i][j][2]
        #        totalCnt += 1
        
        channel_averages = [totals[0]/totalCnt, totals[1]/totalCnt, totals[2]/totalCnt]
        #roi = image[y1:y2][x1:x2][:]
        #print(y1, y2, ";", x1,x2)
        #channel_averages = np.mean(roi, axis=(0,1))  # mean of each channel
        
        

        if draw:
            cv2.rectangle(image, (self.coords[0],self.coords[1]), (self.coords[2],self.coords[3]), (0, 0, 255), 2)
            #cv2.rectangle(image, (x1r,y1r), (x2r,y2r), (0, 0, 255), 2)
            cv2.putText(image, f'blue signal: {round(channel_averages[0],2)}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(image, f'green signal: {round(channel_averages[1],2)}',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f'red signal: {round(channel_averages[2],2)}',
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


        return channel_averages


    def remove_outliers(self, signals, num_stds=1):
        for idx in range(signals.shape[1]):
            x = signals[:, idx]
            k = 2
            winsize = 20 # samples
            for i, num in enumerate(x):
                l_window = x[max(i-winsize,0) : i]
                r_window = x[i+1 : min(i+winsize,len(x))]
                neighbors = np.hstack((l_window, r_window))
                mu, sig = neighbors.mean(), neighbors.std()
                upper_bound = mu + k*sig
                lower_bound = mu - k*sig
                if num > upper_bound or num < lower_bound:
                    x[i] = mu

        return signals


    def detrend_traces(self, channels):
        λ = self.smoothing
        K = channels.shape[0] - 1
        I = scipy.sparse.eye(K)
        D2 = scipy.sparse.spdiags((np.ones((K,1)) * [1,-2,1]).T ,[0,1,2], K-2, K)

        detrended = np.zeros((K, channels.shape[1]))
        for idx in range(channels.shape[1]):  # iterates thru each channel (b,g,r)
            z = channels[:K,idx]
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
            x = bandpassFilter(x, fs, self.freq_cutoff)  # McDuff et al.
            f, psd = scipy.signal.periodogram(x, fs)
            if max(psd) > largest_psd_peak:
                largest_psd_peak = max(psd)
                best_component = components[:,i]
            
        return best_component


    def get_BVP_signal(self, video_path, draw=False):
        if video_path is None:
            stored = pickle.load(open('channel_data.pkl', 'rb'))
            Y, fs = stored['data'], stored['fs']
        else:
            Y, fs = self.sample_video(video_path, draw=draw)  # Get color samples from video
        
            pickle.dump({'data': Y, 'fs': fs}, open('channel_data.pkl', 'wb'))
        
        Y = self.remove_outliers(Y)
        detrended_data = self.detrend_traces(Y)
        pickle.dump(detrended_data, open('detrended_signals.pkl', 'wb'))

        cleaned_data = self.z_normalize(detrended_data)
        pickle.dump(detrended_data, open('cleaned_signals.pkl', 'wb'))
        
        source_signals = self.ica_decomposition(cleaned_data)
        pickle.dump(source_signals, open('ica_signals.pkl', 'wb'))

        bvp_signal = self.select_component(source_signals, fs)
        pickle.dump({'data': bvp_signal, 'fs': fs}, open('bvp_signal.pkl', 'wb'))

        return bvp_signal, fs


    def find_heartrate(self, bvp_signal, fs):
        # 1st paragraph of sec 3c from Poe et al.
        averaged = movingAverageFilter(bvp_signal, self.avg_window_size)

        bp = bandpassFilter(averaged, fs, self.freq_cutoff)

        pickle.dump({'data': bp, 'fs': fs}, open('filtered_bvp.pkl', 'wb'))
        working_data, measures = heartpy.process(bp, fs)
        print(measures)

        plt.title("Bandpass filtered BVP signal")
        plt.plot(bp)
        plt.show()

        return measures


def plot_figures():
    # load in data
    raw = pickle.load(open('channel_data.pkl', 'rb'))['data']
    detrended = pickle.load(open('detrended_signals.pkl', 'rb'))
    clean = pickle.load(open('cleaned_signals.pkl', 'rb'))
    ica = pickle.load(open('ica_signals.pkl', 'rb'))
    bvp = pickle.load(open('bvp_signal.pkl', 'rb'))['data']

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

    # Plot cleaned signals
    clean_fig, clean_ax = plt.subplots(3,1, tight_layout={'pad': 1})
    clean_fig.set_size_inches(8, 6)
    plt.subplots_adjust(wspace=None, hspace=1)

    for c, name in enumerate(['Blue', 'Green', 'Red']):  # plot each component
        clean_ax[c].set_title(f"{name} signal - cleaned")
        clean_ax[c].plot(clean[:,c])

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
    parser.add_argument('--source', '-s', action="store", type=str, default='Sessions/1/')
    parser.add_argument('--load', '-l', action="store_true", default=False, help="Use the most recent channel data")
    parser.add_argument('--draw', '-d', action="store_true", default=False)

    parser.add_argument('--plot', '-p', help="Plot figures from code", action="store_true", default=False)
    parser.add_argument('--hr', help="Calculate heart rate from bvp signal", action="store_true", default=False)
    parser.add_argument('--extract', '-e', help="Extract channel data from multiple videos", action="store", type=str)

    parser.add_argument('--smooth', action='store', type=float, default=300, help="Smoothing parameter for detrending")
    parser.add_argument('--avg', '-a', action='store', type=float, default=5, help="Window size for average filter")
    parser.add_argument('--cutoff', '-c', action='append', nargs=2, default=[0.7,3], help="Lower and upper bandpass frequencies")
    args = parser.parse_args()

    exctractor = BVPExtractor(args.smooth, args.avg, args.cutoff)

    if args.plot:
        print("plotting figures...")
        plot_figures()
    elif args.hr:
        bvp_data = pickle.load(open('bvp_signal.pkl', 'rb'))
        hr = exctractor.find_heartrate(bvp_data['data'], bvp_data['fs'])
    elif args.extract:
        output_folder = 'channel_data'
        for video_path in glob.glob(args.extract + '*'):
            filename = video_path[video_path.rfind('/')+1:]
            print(filename)
            Y, fs = exctractor.sample_video(video_path)
            pickle.dump({'data': Y, 'fs': fs}, open(f'{output_folder}/{filename}channels.pkl', 'wb'))

    else:
        print("Running algorithm...")
        bvp_signal, fs = exctractor.get_BVP_signal(args.source if not args.load else None, draw=args.draw)
        exctractor.find_heartrate(bvp_signal, fs)


if __name__ == "__main__":
    main()

