#import sys
#import glob,os
#import json
import pandas as pd
#import tensorflow as tf
import logging
#import zipfile
#import re
#import datetime
import numpy as np
#import os
#import glob
#import matplotlib
import math
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import datetime
#import tensorflow as tf
#import sklearn as sk
from tensorflow.python import keras as ks
from pprint import pprint
import re
#import cv2
import json


import sys, glob, os
import json
import pandas as pd
#import tensorflow as tf
# Check versions
#assert tf.__version__ == '1.8.0'

import logging
import zipfile
#import re
import datetime
#import cv2
#import shutil
#import json
#from tabulate import tabulate
import tqdm
#from IPython import get_ipython
#import cv2


import glob
#import json
#import pandas as pd
import tensorflow as tf
# Check versions
assert tf.__version__ == '1.8.0'

import logging
#import zipfile
#import re
#import datetime
#import cv2
#import shutil
#import json
#from tabulate import tabulate
#import tqdm
#from IPython import get_ipython
import cv2

#%%
class NoPlots:
    def __enter__(self):
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.ioff()
    def __exit__(self, type, value, traceback):
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.ion()


#%% Logging
#>>> import warnings
#>>> image = np.array([0, 0.5, 1], dtype=float)
#>>> with warnings.catch_warnings():
#...     warnings.simplefilter("ignore")
#...     img_as_ubyte(image)




#%% LOGGING for Spyder! Disable for production. 
logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.DEBUG)

# Create formatter
#FORMAT = "%(asctime)s - %(levelno)s - %(module)-15s - %(funcName)-15s - %(message)s"
#FORMAT = "%(asctime)s L%(levelno)s: %(message)s"
FORMAT = "%(asctime)s - %(levelname)s - %(funcName) -20s: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.critical("Logging started")

class LoggerCritical:
    def __enter__(self):
        my_logger = logging.getLogger()
        my_logger.setLevel("CRITICAL")
    def __exit__(self, type, value, traceback):
        my_logger = logging.getLogger()
        my_logger.setLevel("DEBUG")

#
#
#
#
#
#class LoggerCritical:
#    def __enter__(self):
#        my_logger = logging.getLogger()
#        my_logger.setLevel("CRITICAL")
#    def __exit__(self, type, value, traceback):
#        my_logger = logging.getLogger()
#        my_logger.setLevel("DEBUG")
#
#
#import logging
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)
#logging.debug("test")
#
#with LoggerCritical():
#    logging.debug("test block")
#%%
def remove_outliers(this_series):
    """Given a pd.Series, return a new Series with no outliers
    """
    no_outlier_mask = np.abs(this_series-this_series.mean()) <= (3*this_series.std())
    return this_series[no_outlier_mask]


#%%

def mm2inch(value):
    return value/25.4
PAPER_A3_LAND = (mm2inch(420),mm2inch(297))
PAPER_A4_LAND = (mm2inch(297),mm2inch(210))
PAPER_A5_LAND = (mm2inch(210),mm2inch(148))
#%%
    

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts    


#%%

class VideoWriter:
    """From a folder containing jpg images - create a video. 
    
    
    """
    
    def __init__(self,jpg_folder, path_vid_out, fps):
        self.jpg_folder = jpg_folder
        self.path_vid_out = path_vid_out
        self.fps = fps
        assert os.path.exists(self.jpg_folder), "{} does not exist".format(self.jpg_folder)
        
        jpg_files = glob.glob(os.path.join(self.jpg_folder,'*.jpg'))
        
        logging.debug("{} video JPG frames found in {}".format(len(jpg_files),self.jpg_folder))

        self.jpg_files = self.sort_jpgs(jpg_files)
        
        logging.debug("Output video set to {} at {} fps".format(self.path_vid_out,self.fps))

        self.height, self.width = self.get_dimensions(self.jpg_files[0])

    def get_dimensions(self,this_jpg_path):
        # Load a single frame to get dimensions
        img_arr = mpl.image.imread(this_jpg_path)
        frames_height = img_arr.shape[0]
        frames_width = img_arr.shape[1]
        logging.debug("Dimensions: {} x {} pixels (Height x Width)".format(frames_height,frames_width))
        
        return frames_height,frames_width

    def sort_jpgs(self,jpg_files):
        """Sort on file name (timestamp)
        
        """
        frame_paths = list()
        for this_img_path in jpg_files:
            this_frame_dict = dict()
            _, this_img_fname = os.path.split(this_img_path)
            timestamp = os.path.splitext(this_img_fname)[0]
            this_frame_dict['timestamp'] = timestamp
            this_frame_dict['path'] = this_img_path
            frame_paths.append(this_frame_dict)

        # Sort the frames!
        frame_paths_sorted = sorted(frame_paths, key=lambda k: k['timestamp'])
        logging.debug("Sorted {} video frame image paths".format(len(frame_paths_sorted)))
        
        return [fd['path'] for fd in frame_paths_sorted]
    
    def write_video(self, num_frames=None, overwrite=False):
        """From a list of frame JPG paths, generate a MP4. 
        
        Optionally specify the length of the video in num_frames (good for testing)
        """
        
        if os.path.exists(self.path_vid_out) and not overwrite:
            logging.debug("{} video already exists, skip unless overwrite=True".format(self.path_vid_out))
            
            return
        
        # This is extremely picky, and can fail (create empty file) with no warning !!
        writer = cv2.VideoWriter(self.path_vid_out, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (self.width,self.height))
        if not num_frames:
            frames_to_write = self.jpg_files
        else:
            frames_to_write = self.jpg_files[0:num_frames]
        with NoPlots(), LoggerCritical():
            for this_jpg_path in tqdm.tqdm(frames_to_write):
                img_arr = mpl.image.imread(this_jpg_path)
                RGB_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

                writer.write(RGB_img) # Write out frame to video
        
        logging.debug("Wrote {} frames to {}".format(len(frames_to_write),self.path_vid_out))
        
        writer.release()
        cv2.destroyAllWindows()
    
    def test_write(self):
        """To test cv2 import, dimensions (cv2 is very picky), etc.
        
        Write some random noise to the video out path. 
        """
        
        # Dimensions, for testing purposes
        H = 480
        W = 640
        writer = cv2.VideoWriter(self.path_vid_out, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W,H))
        for frame in tqdm.tqdm(range(400)):
            this_frame = np.random.randint(0, 255, (H,W,3)).astype('uint8')
            writer.write(this_frame)
        writer.release()        
        logging.debug("Wrote test video to {}".format(self.path_vid_out))

