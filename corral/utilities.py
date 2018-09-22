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
import matplotlib.pyplot as plt
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
        pass
        #get_ipython().run_line_magic('matplotlib', 'qt')
        #plt.ioff()
    def __exit__(self, type, value, traceback):
        pass
        #get_ipython().run_line_magic('matplotlib', 'inline')
        #plt.ion()


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
PAPER = {
    PAPER_A3_LAND : (mm2inch(420),mm2inch(297))
    PAPER_A4_LAND : (mm2inch(297),mm2inch(210))
    PAPER_A5_LAND : (mm2inch(210),mm2inch(148))
}

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

