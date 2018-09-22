"""
Manage and analyze the Data Set directory.
Iterate over each Data Set and check which data elements have been created. 

NOT Standalone script. 

"""
#%% Instantiate and load the dataset

LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
#DATASET_ID = "20180907 193306"
DATASET_ID = "20180907 191303 BENCHMARK1 AUTO"
#DATASET_ID = "20180907 193757"
THIS_MODEL_ID = 'model 20180907 190147'

assert os.path.exists(LOCAL_PROJECT_PATH)

#%% ===========================================================================
# Load dataset
# =============================================================================
ds = AIDataSet(LOCAL_PROJECT_PATH,DATASET_ID)
ds.process_time_steps()
if True: 
    ds.write_jpgs(dir_jpgs="jpg_images", overwrite=False)

#%% ===========================================================================
# Frames: to /Video Frames and /Video Frames.mp4
# =============================================================================
ds.write_frames(overwrite=False)

PATH_INPUT_JPGS = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames')
PATH_OUTPUT_FILE = os.path.join(LOCAL_PROJECT_PATH,DATASET_ID,'Video Frames.mp4')

vidwriter = VideoWriter(PATH_INPUT_JPGS,PATH_OUTPUT_FILE,fps=8)
vidwriter.write_video()

#%% Start CUDA and Training
#ks.backend.clear_session()
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
for dev in devices:
    print(dev.name)

#%% 1) Load the DataSet, and train the predictions AGAIN
trds = ModelledData(ds,THIS_MODEL_ID)
trds.load_best_model()
trds.model.summary()
trds.make_predictions()

#%% 2) Saliency
this_saliency = SaliencyGen(trds)
this_saliency.gen_pure_CNN()
this_saliency.path_saliency_jpgs
##this_saliency.modelled_dataset.model.layers
#this_saliency.modelled_dataset.model.summary()
this_saliency.get_layers()
this_saliency.saliency_tf_function()
this_saliency.get_kernels()
if False: # This takes a while!
    this_saliency.write_saliency_mask_jpgs()

#%% HUD Frames
this_saliency.create_HUD_frames()
raise

#%%
#this_strength = 4
#this_blur = 3
this_saliency.blend_simple(2,1.5,num_frames=None)
#blend_simple(self,blur_rad,strength,num_frames = None)
raise



#%%
this_map = 'viridis'
this_map = 'seismic'
this_map = 'summer'
this_map = 'magma'
this_map = 'hot'
this_strength = 4
this_blur = 3
this_saliency.blend_PIL(this_blur,this_map,this_strength,this_map)
