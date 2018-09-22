"""
Writes video from a folder of JPG
"""
#%%
#import glob
#import os
#import logging
#import sys
#import cv2
#import tqdm

#%%
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
assert os.path.exists(LOCAL_PROJECT_PATH)
#THIS_DATASET = "20180829 194519"
THIS_DATASET_ID = "20180907 191303 BENCHMARK1 AUTO"
#THIS_FRAMES_DIR = "Video Frames"
THIS_MODEL_ID = 'model 20180907 190147'
THIS_FRAMES_DIR = "frames_saliency_boosted"
OUTPUT_VID_NAME = "frames_saliency_boosted.mp4"



PATH_INPUT_JPGS = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET_ID,THIS_MODEL_ID,THIS_FRAMES_DIR)
PATH_OUTPUT_FILE = os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,OUTPUT_VID_NAME)

#PATH_INPUT_JPGS = '/home/batman/MULE DATA/20180904 192907/model 20180906 165918/frames_saliency_boosted'
#PATH_OUTPUT_FILE = '/home/batman/MULE DATA/20180904 192907/model 20180906 165918/frames_saliency_boosted.mp4'


#%%
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


class NoPlots:
    def __enter__(self):
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.ioff()
    def __exit__(self, type, value, traceback):
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.ion()


#%% Initialize the video writer object

vidwriter = VideoWriter(PATH_INPUT_JPGS,PATH_OUTPUT_FILE,fps=8)
#vidwriter.test_write()
#%%
if False: 
    vidwriter.write_video()
