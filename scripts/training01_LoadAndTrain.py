import glob
import os
#%%
"""
Manage and analyze the Data Set directory.
Iterate over each Data Set and check which data elements have been created. 

"""
from corral.dataset import AIDataSet
from corral.datagen import MuleDataGenerator
from corral.utilities import NoPlots
from corral.plotter import DataSetPlotter
#%% Instantiate and load the dataset
LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
DATASET_ID = "20180907 184100 BENCHMARK1 TRG"
DATASET_ID = "20180910 202846 AUTO"

# USE THESE DATASETS
DATASET_ID = "20180910 181306"
DATASET_ID = "20180910 192658"
DATASET_ID = "20180910 193908"
DATASET_ID = "20180912 120000 FINAL SET"
DATASET_ID = "20180912 161937"

assert os.path.exists(LOCAL_PROJECT_PATH)


WRITE_JPG_RGB = 0
WRITE_JPG_Y = 0

WRITE_FRAMES_RGB = 0
WRITE_FRAMES_Y = 0

#
mask_sections = (
        ('1536596037921','1536596044125'),
        
        )
        



#%% ===========================================================================
# Load dataset
# =============================================================================
ds = AIDataSet(LOCAL_PROJECT_PATH,DATASET_ID)
ds.process_time_steps()
if WRITE_JPG_RGB: 
    ds.write_jpgs(dir_jpgs="jpg_images", overwrite=False)
if WRITE_JPG_Y:     
    ds.write_jpgs_bw(dir_jpgs="jpg_images_Y", overwrite=False)

#%% ===========================================================================
# Mask values
# =============================================================================

ds.mask_first_Ns(2)
ds.mask_last_Ns(2)
ds.mask_null_throttle(0.1)   

#%% ===========================================================================
# Summary plots
# =============================================================================
with NoPlots():
    plotter = DataSetPlotter()
    plotter.histogram_steering(ds)
    plotter.histogram_throttle(ds)
    with LoggerCritical():
        if WRITE_JPG_RGB: 
            plotter.plot_sample_frames(ds)
        if WRITE_JPG_Y:     
            plotter.plot_sample_frames_bw(ds)
        
    plotter.boxplots_time(ds)

#%% ===========================================================================
# Frames: to /Video Frames and /Video Frames.mp4
# =============================================================================
if WRITE_FRAMES_RGB:
    ds.write_frames(overwrite=False)
    
    PATH_INPUT_JPGS = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames')
    PATH_OUTPUT_FILE = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames.mp4')
    
    vidwriter = VideoWriter(PATH_INPUT_JPGS,PATH_OUTPUT_FILE,fps=24)
    vidwriter.write_video()

if WRITE_FRAMES_Y:
    ds.write_frames(output_dir_name="Video Frames Y", overwrite=False,blackwhite=True,cmap='bwr',gui_color='black')
    
    PATH_INPUT_JPGS = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames Y')
    PATH_OUTPUT_FILE = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames Y.mp4')
    
    vidwriter = VideoWriter(PATH_INPUT_JPGS,PATH_OUTPUT_FILE,fps=24)
    vidwriter.write_video()

#%% Start CUDA and Training
ks.backend.clear_session()
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
for dev in devices:
    print(dev.name)

#%% ===========================================================================
# Train model
# =============================================================================

# New Model_ID
THIS_MODEL_ID = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
#THIS_MODEL_ID = '20180909 160218'
dsm=ModelledData(ds,THIS_MODEL_ID)
dsm.model_folder_empty
dsm.generate_partitions()
#trained_dataset.list_models()
#dsm.instantiate_generators(MuleDataGeneratorBlackWhite)
#dsm.instantiate_model(model_name="blackwhite_steering_model")

dsm.instantiate_generators(MuleDataGenerator)
dsm.instantiate_model(model_name="baseline_steering_model")

dsm.model.summary()
#raise
dsm.instantiate_callbacks()
dsm.callback_list
dsm.train_model(50)
dsm.make_predictions()


#%% TEST GEN
if 0:
    generator_params = {'dim': (160,120),
              'batch_size': 64,
              'n_classes': 15,
              'n_channels': 3,
              'shuffle': True,
              #'path_frames':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'),
              #'path_records':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'),
             }
    
    training_generator = MuleDataGeneratorBlackWhite(dsm.partition['train'], dsm.ds, **generator_params)
    first_batch = training_generator[0]
    last_batch = training_generator[len(training_generator)]