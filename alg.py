# Remote plehtysysmosngraphy
# David Haas, Hogan Pope, Spencer Mullinix

# science
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sklearn.decomposition import FastICA
import cv2

# utilities
from tqdm import tqdm, trange
import glob


class PulsePolice:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')


    def get_face_sample(self, image, draw=False, bbox_shrink=0.6):
        # Detect the faces
        faces = self.face_cascade.detectMultiScale(image, 1.1, 4)
        if faces is None:
            print('No face detected')
            return False

        # Get bounding box
        x, y, w, h = faces[0]

        # Shrink bounding box to get only face skin
        x1,y1 = int(x + w*bbox_shrink/2), int(y + h*bbox_shrink/2)
        x2,y2 = int((x + w) - w*bbox_shrink/2), int((y+h) - h*bbox_shrink/2)
        if draw:
            cv2.rectangle(image, (x1,y1), (x2,y2), (0, 0, 255), 2)
        
        # Extract and filter bounding box data to one measurement per channel
        roi = image[y1:y2,x1:x2,:]
        channel_averages = np.mean(roi, axis=(0,1))  # mean of each channel
        # print(channel_averages)
        return channel_averages


    def detrend_traces(self, channels, λ=10):
        K = channels.shape[0] - 1
        I = scipy.sparse.eye(K)
        D2 = scipy.sparse.spdiags((np.ones((K,1)) * [1,-2,1]).T ,[0,1,2], K-2, K)

        detrended = np.zeros((K, channels.shape[1]))
        for idx in range(channels.shape[1]):  # iterates thru each channel (b,g,r)
            #TODO I'm not sure if Poh et al used the difference array as z, like 
            # the did in the detrending paper
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


    def select_component(self, components):
        # Select the signal with the highest PSD peak
        pass


    def sample_video(self, video_path, draw=False):
        video_stream = cv2.VideoCapture(video_path)
        nframes = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)/10)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print(nframes / fps)

        Y = np.zeros((nframes,3))  # Holds the data for b,g,r channels

        for i in trange(nframes):
            good, frame = video_stream.read()       
            Y[i,:] = self.get_face_sample(frame, draw=draw)

            if draw:
                cv2.imshow('Press Q to quit', frame)
                if cv2.waitKey(int(1)) & 0xFF == ord('q'):
                    break
            
        video_stream.release()
        cv2.destroyAllWindows()

        import code; code.interact(local=locals())

        return Y, fps


    def get_BVP_signal(self, video_path, draw=True):
        # Y, fs = self.sample_video(video_path)  # Get color sampels from video
        Y = np.load('channel_data.npy')    
        
        detrended_data = self.detrend_traces(Y)
        cleaned_data = self.z_normalize(detrended_data)
        source_signals = self.ica_decomposition(cleaned_data)
        import code; code.interact(local=locals())
        # bvp_signals = self.select_component(source_signals)


def main():
    import argparse
    def get_video(folder_path): return glob.glob(f"{folder_path}*.avi")[0]
    video_path = get_video('Sessions/1/')

    pp = PulsePolice()
    pp.get_BVP_signal(video_path, draw=False)


if __name__ == "__main__":
    main()
