#%% DATAGEN
class MuleDataGenerator(ks.utils.Sequence):
     """Generates data for Keras"""
     def __init__(self, indices, dataset, 
                  batch_size=32, dim=None, n_channels=None, n_classes=15, shuffle=True):
         """Keras data generator
         
         Aggregates the AIDataSet class
         
         Attributes:
             indices (str): The allowed timestamps for data generation
             dataset (AIDataSet): The dataset object with it's df and npz
             batch_size : 
             dim : 
             n_channels : 
             n_classes :
             shuffle :
         """
         self.indices = indices
         self.dataset = dataset
         self.batch_size = batch_size
         self.dim = dim
         self.n_channels = n_channels
         self.n_classes = n_classes
         self.shuffle = shuffle
         self.on_epoch_end()
         
         logging.debug("** Initialize datagen **".format())
         logging.debug("Data folder: {}".format(dataset.data_folder))
         
         logging.debug("{} of {} total records used for generation".format(len(self.indices), len(self.dataset.df)))
         #logging.debug("Frames NPZ located at: {}".format(self.dataset.path_frames_npz))
         logging.debug("{} samples over batch size {} yields {} batches".format(len(self.indices),
                                                                                    self.batch_size,
                                                                                    math.ceil(len(self.indices)/self.batch_size),))
         
     def __len__(self):
         """Keras generator method - Denotes the number of batches per epoch
         """        
         return int(np.floor(len(self.indices) / self.batch_size))
     
     # GET A BATCH!
     def __getitem__(self, index): 
         """Keras generator method - Generate one batch of data
         """         
         logging.debug("Generating batch {}".format(index))
         
         # Generate indexes of the batch
         batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
 
         # Generate data by selecting these IDs
         X, y = self.__data_generation(batch_indices)
 
         return X, y
 
     def on_epoch_end(self):
         """Keras generator method - Shuffles indices after each epoch
         """
         #self.indexes = np.arange(len(self.indices))
         if self.shuffle == True:
             # Shuffle is in-place! 
             np.random.shuffle(self.indices)
             
     def __get_npy_arrays(self,batch_indices):
         """Custom method - get the X input arrays
         
         Open the npz file and load n frames into memory
         """
         # This is a pointer to the file
         npz_file=np.load(self.dataset.path_frames_npz)
         
         frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
         logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))
         
         return frames_array
     
     def __get_records(self,batch_indices):
         """Custom method - get the y labels
         """
         this_batch_df = self.dataset.df.loc[batch_indices]
         steering_values = this_batch_df['steering_signal'].values
         steering_records_array = self.dataset.bin_Y(steering_values)
         logging.debug("Generating {} records {}:".format(steering_records_array.shape[0],steering_records_array.shape))
         return steering_records_array
         
     def __data_generation(self, batch_indices):
         """Keras generator method - Generates data containing batch_size samples
         """
 
         X = self.__get_npy_arrays(batch_indices)
         y = self.__get_records(batch_indices)
 
         return X, y


class MuleDataGeneratorBlackWhite(ks.utils.Sequence):
     """Generates data for Keras"""
     def __init__(self, indices, dataset, 
                  batch_size=32, dim=None, n_channels=None, n_classes=15, shuffle=True):
         """Keras data generator
         
         Aggregates the AIDataSet class
         
         Attributes:
             indices (str): The allowed timestamps for data generation
             dataset (AIDataSet): The dataset object with it's df and npz
             batch_size : 
             dim : 
             n_channels : 
             n_classes :
             shuffle :
         """
         self.indices = indices
         self.dataset = dataset
         self.batch_size = batch_size
         self.dim = dim
         self.n_channels = n_channels
         self.n_classes = n_classes
         self.shuffle = shuffle
         self.on_epoch_end()
         
         logging.debug("** Initialize datagen **".format())
         logging.debug("Data folder: {}".format(dataset.data_folder))
         
         logging.debug("{} of {} total records used for generation".format(len(self.indices), len(self.dataset.df)))
         #logging.debug("Frames NPZ located at: {}".format(self.dataset.path_frames_npz))
         logging.debug("{} samples over batch size {} yields {} batches".format(len(self.indices),
                                                                                    self.batch_size,
                                                                                    math.ceil(len(self.indices)/self.batch_size),))
         
     def __len__(self):
         """Keras generator method - Denotes the number of batches per epoch
         """        
         return int(np.floor(len(self.indices) / self.batch_size))
     
     # GET A BATCH!
     def __getitem__(self, index): 
         """Keras generator method - Generate one batch of data
         """         
         logging.debug("Generating batch {}".format(index))
         
         # Generate indexes of the batch
         batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
 
         # Generate data by selecting these IDs
         X, y = self.__data_generation(batch_indices)
 
         return X, y
 
     def on_epoch_end(self):
         """Keras generator method - Shuffles indices after each epoch
         """
         #self.indexes = np.arange(len(self.indices))
         if self.shuffle == True:
             # Shuffle is in-place! 
             np.random.shuffle(self.indices)
             
     def __get_npy_arrays(self,batch_indices):
         """Custom method - get the X input arrays
         
         Open the npz file and load n frames into memory
         """
         # This is a pointer to the file
         npz_file=np.load(self.dataset.path_frames_npz)
         
         frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
         
         #BLACK AND WHITE
         #TODO: Structure the datagen someway, modularize
         
         
         #print(frames_array.shape)
         frames_array_bw = np.mean(frames_array,axis=3)
         frames_array_bw = np.expand_dims(frames_array_bw,3)
         #print(frames_array_bw.shape)
         #raise
         logging.debug("Generating {} frames: {}".format(frames_array_bw.shape[0], frames_array_bw.shape))
         
         return frames_array_bw
     
     def __get_records(self,batch_indices):
         """Custom method - get the y labels
         """
         this_batch_df = self.dataset.df.loc[batch_indices]
         steering_values = this_batch_df['steering_signal'].values
         steering_records_array = self.dataset.bin_Y(steering_values)
         logging.debug("Generating {} records {}:".format(steering_records_array.shape[0],steering_records_array.shape))
         return steering_records_array
         
     def __data_generation(self, batch_indices):
         """Keras generator method - Generates data containing batch_size samples
         """
 
         X = self.__get_npy_arrays(batch_indices)
         y = self.__get_records(batch_indices)
 
         return X, y




#%% Data set class
class AIDataSet():
    """A single datafolder object
    
    Description text
    
    Attributes
    ----------
    df : pandas.DataFrame
        The dataframe object, with the 'timestamp' column for indexing. 
    path_frames_npz : str
        Path to frames numpy zip object. Can be accessed directly by np.load.
    """
    
    def __init__(self,path_data,data_folder):
        # Check the data folder
        self.path_data = path_data
        assert os.path.exists(self.path_data)
        self.data_folder = data_folder
        self.path_dataset = os.path.join(self.path_data,self.data_folder)
        assert os.path.exists(self.path_dataset)
        
        logging.debug("Data set {}, recorded on {}".format(self.data_folder,self.datetime_string))

        # Check the raw records zip, load to DataFrame
        self.path_records_zip = os.path.join(self.path_dataset,"json_records.zip")
        assert os.path.exists(self.path_records_zip)   
        self.df = self.load_records_df()
        logging.debug("Records {}".format(len(self.df)))
        self.df['steering_signal_catnum'] = self.signal_to_category_number('steering_signal')

        
        # Check the raw frames zip, no need to unzip
        self.path_frames_npz = os.path.join(self.path_dataset,"camera_numpy.zip")
        assert os.path.exists(self.path_frames_npz)
        frames_timestamps = self.get_frames_timesteps() 
        logging.debug("Frames npz is {:0.2f} MB, {} records".format(self.frames_size,len(frames_timestamps)))
        
        # Assert timestep alignment
        assert all(self.df.index == frames_timestamps), "Misaligned timestamps"
        
        # JPG folder
        JPG_FOLDER_NAME = "jpg_images"
        self.path_jpgs_dir = os.path.join(self.path_dataset,JPG_FOLDER_NAME)
        
        
        self.mask = pd.Series(True, index=self.df.index)
        
        # Predictions
        #self.predicted_model = None
        #self.
        self.augment_df_datetime()

    # =============================================================================
    #--- Query
    # =============================================================================
    @property
    def datetime_string(self):
        p = re.compile("^\d+ \d+")
        folder_dt = p.findall(self.data_folder)[0]
        dt_obj = datetime.datetime.strptime(folder_dt, '%Y%m%d %H%M%S')
        return dt_obj.strftime("%A %d %b %H:%M")

    @property
    def datetime_string_iso(self):
        p = re.compile("^\d+ \d+")
        folder_dt = p.findall(self.data_folder)[0]
        dt_obj = datetime.datetime.strptime(folder_dt, '%Y%m%d %H%M%S')
        return dt_obj.isoformat()

    @property
    def frames_size(self):
        """Size of frames npz array in MB
        """
        #print(self.path_frames_npz)
        return os.path.getsize(self.path_frames_npz)/1000/1000
    
    def get_frames_timesteps(self):
        """Get timestamps from zipped NPY files. Return sorted pd.Series. 
        """
        # Open zip
        with zipfile.ZipFile(self.path_frames_npz, "r") as f:
            # Get the file names
            fnames = (os.path.splitext(name) for name in f.namelist()) 
            # Split and save
            timestamps, extensions = zip(*fnames)
        assert all(ext == '.npy' for ext in extensions)
        
        # Convert to datetime
        #datetime_stamps = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
        # SORT!
        #datetime_stamps.sort()
        
        # Sorted and reindexed! 
        return pd.Series(timestamps).sort_values().reset_index(drop=True)
    
    def __str__(self):
        return "Dataset {} at {} with {} records".format(self.data_folder, self.path_data, len(self.df))

    @property
    def elapsed_minutes(self):
        elapsed_time = self.df['datetime'].iloc[-1] - self.df['datetime'].iloc[0]
        elapsed_time_min = elapsed_time.total_seconds() / 60
        return elapsed_time_min
    
    @property
    def num_records(self):
        return len(self.df)
    
    @property
    def int_index(self,timestamp):
        # Helper to swap timestamp string <> integer index on df
        return self.df[self.df['timestamp']==timestamp]


    @property
    def timestamp(self,int_index):
        # Helper to swap timestamp string <> integer index on df
        return self.df[int_index]['timestamp']
                
    
    # =============================================================================
    #--- Utility
    # =============================================================================
    @property
    def mask_cover_pct(self):
        return 100 - sum(self.mask)/len(self.mask)*100
    
    def mask_first_Ns(self,numsecs = 3):
        ds = self
        first_datetime = datetime.datetime.fromtimestamp(int(ds.df.index[0])/1000)
        assert ds.df['datetime'][0] == first_datetime
        # Get a timedelta (days,seconds)
        tdelta = datetime.timedelta(0,numsecs)
        second_datetime = ds.df['datetime'][0] + tdelta
        truemask = (ds.df['datetime'] >= first_datetime) & (ds.df['datetime'] <= second_datetime)
        this_mask = ~truemask
        self.mask = self.mask & this_mask
        logging.debug("Masked {} timesteps from {} to {}, current cover: {:0.1f}%".format(sum(this_mask),first_datetime,second_datetime, self.mask_cover_pct))
        
    def mask_last_Ns(self,numsecs = 2):
        ds = self
        last_datetime = datetime.datetime.fromtimestamp(int(ds.df.index[-1])/1000)
        assert ds.df['datetime'][-1] == last_datetime
        # Get a timedelta (days,seconds)
        tdelta = datetime.timedelta(0,numsecs)
        start_datetime = ds.df['datetime'][-1] - tdelta
        truemask = (ds.df['datetime'] >= start_datetime) & (ds.df['datetime'] <= last_datetime)
        this_mask = ~truemask
        self.mask = self.mask & this_mask
        logging.debug("Masked {} timesteps from {} to {}, current cover: {:0.1f}%".format(sum(truemask),start_datetime,last_datetime,self.mask_cover_pct))

    def mask_null_throttle(self,cutoff=0.1):
        ds = self
        truemask = ds.df['throttle_signal'] <= cutoff
        this_mask = ~truemask 
        self.mask = self.mask & this_mask
        logging.debug("Masked {} timesteps throttle<{}, current cover: {:0.1f}%".format(sum(truemask),cutoff,self.mask_cover_pct))


    def mask_(self,first_ts,second_ts):
        ds = self
        #first_datetime = datetime.datetime.fromtimestamp(int(ds.df.index[0])/1000)
        #assert ds.df['datetime'][0] == first_datetime
        # Get a timedelta (days,seconds)
        #tdelta = datetime.timedelta(0,numsecs)
        #second_datetime = ds.df['datetime'][0] + tdelta
        truemask = (ds.df.index >= first_ts) & (ds.df.index >= second_ts)
        this_mask = ~truemask
        self.mask = self.mask & this_mask
        logging.debug("Masked {} timesteps from {} to {}, current cover: {:0.1f}%".format(sum(this_mask),first_datetime,second_datetime, self.mask_cover_pct))
        
        
    
    # Conversion between categorical and floating point steering
    def linear_bin(self,a):
        a = a + 1
        b = round(a / (2 / 14))
        arr = np.zeros(15)
        arr[int(b)] = 1
        return arr
    
    
    def linear_unbin(self,arr):
        if not len(arr) == 15:
            raise ValueError('Illegal array length, must be 15')
        b = np.argmax(arr)
        a = b * (2 / 14) - 1
        return a
    
    
    def bin_Y(self,Y):
        d = [ self.linear_bin(y) for y in Y ]
        return np.array(d)
    
    
    def unbin_Y(self,Y):
        d = [ self.linear_unbin(y) for y in Y ]
        return np.array(d)    

    def signal_to_category_number(self,column_name):
        """Break the floating point [-1,1] signal into bins
        """
        cats = self.bin_Y(self.df[column_name])
        # Get the category number
        return np.argmax(cats,axis=1)
        
    
    # =============================================================================
    #--- Load into memory
    # =============================================================================
    def load_records_df(self):
        """Get DataFrame from zipped JSON records. Return sorted pd.DataFrame. 
        
        All record columns created
        Timestamp column added (mtime)
        Sort the DF on timestamp
        Reindex        
        """        
        json_records = list()
        with zipfile.ZipFile(self.path_records_zip, "r") as f:
            json_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] =='.json']
            # Each record is a seperate json file
            for json_file in json_file_paths:
                this_fname = os.path.splitext(json_file)[0] 
                this_timestep = this_fname.split('_')[1]
                d = f.read(json_file)
                d = json.loads(d.decode("utf-8"))
                d['timestamp'] = this_timestep
                json_records.append(d)
        # Sorted and reindexed!
        this_df = pd.DataFrame(json_records).sort_values(by='timestamp')
        this_df.index = this_df['timestamp']
        #.reset_index(drop=True)
        this_df['steering_signal'] = this_df['steering_signal'].apply(lambda x: x*-1)
        logging.debug("Steering signal inverterted - WHY?".format())

        return this_df
        #return pd.DataFrame(json_records).sort_values(by='timestamp').reset_index(drop=True)
    

            
    # =============================================================================
    #--- Timestep analysis and processing
    # =============================================================================
    def augment_df_datetime(self):
        def convert_datetime(x):
            return datetime.datetime.fromtimestamp(int(x)/1000)
            #return datetime.datetime.strptime(x, '%Y%m%d %H%M%S')
        self.df['datetime'] = self.df['timestamp'].apply(convert_datetime)
        logging.debug("Augmented df with 'datetime' column".format())
        
    def process_time_steps(self):
        """Analysis of timestamps. Add some attributes to the class. 
        """
        assert 'datetime' in self.df.columns
        # Analysis of timesteps
        self.elapsed_time = self.df['datetime'].iloc[-1] - self.df['datetime'].iloc[0]
        self.elapsed_time_min = self.elapsed_time.total_seconds() / 60
        
        # Analysis of delta-times
        ts_deltas = (self.df['datetime']-self.df['datetime'].shift()).fillna(0)
        
        self.df['ts_deltas_ms'] = ts_deltas.apply(lambda x : x.total_seconds() * 1000)
        
        
        logging.debug("ts_deltas_ms column added".format())
        
        stats = ts_deltas[0:-1].describe()
        
        self.ts_deltas_mean = stats['mean'].total_seconds() * 1000
        self.ts_deltas_std = stats['std'].total_seconds() * 1000
        
        logging.debug("{:0.2f} minutes elapsed between start and stop".format(self.elapsed_time_min))

        logging.debug("Timestep analysis: {:0.0f} +/- {:0.0f} ms".format(
                      self.ts_deltas_mean,
                      self.ts_deltas_std
                      ))        
    
    # =============================================================================
    #--- Video
    # =============================================================================
    def gen_record_frame(self, ts_string_index, source_jpg_folder='jpg_images', source_ext = '.jpg',cmap=None,gui_color='green'):
        """From a timestamp, create a single summary figure of that timestep. 
        
        The figure has no border (full image)
        
        Show a data box with throttle and steering values. 
        Show also the predicted values, if available. 
        
        Show a steering widget to visualize the current steering signal. 
        Show also the predicted value, if available. 
        
        """
        rec = self.df.loc[ts_string_index]
        # Settings ############################################################
        font_label_box = {
            'color':gui_color,
            'size':16,
        }
        font_steering = {'family': 'monospace',
                #'color':  'darkred',
                'weight': 'normal',
                'size': 45,
                }
        SCALE = 50
        HEIGHT_INCHES = 160*2.54/SCALE
        WIDTH_INCHES =  120*2.54/SCALE
        
        # Figure ##############################################################
        fig = plt.figure(frameon=False,figsize=(HEIGHT_INCHES,WIDTH_INCHES))
        ax = mpl.axes.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Main Image ##########################################################
        jpg_path = os.path.join(self.path_dataset,source_jpg_folder,ts_string_index+source_ext)
        #print(self.path_dataset)
        #print(source_jpg_folder)
        #print(ts_string_index+source_ext)
        assert os.path.exists(os.path.join(self.path_dataset,source_jpg_folder)), "Does not exist: {}".format(os.path.join(self.path_dataset,source_jpg_folder))
        img = mpl.image.imread(jpg_path)
        ax.imshow(img,cmap)
        #raise
        
        
        #ax.axes.get_xaxis().set_visible(False)
        #ax.axes.get_yaxis().set_visible(False)
        
        # Data box ########################################################
        timestamp_string = rec['datetime'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['datetime'].microsecond))
        if 'steering_pred_signal' in self.df.columns:
            this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,
                          rec['steering_signal'],rec['steering_pred_signal'],rec['throttle_signal'])
        else: 
            this_label = "{}\n{:0.2f}/ steering \n{:0.2f} throttle".format(timestamp_string,rec['steering_signal'],rec['throttle_signal'])
        t1 = ax.text(2,15,this_label,fontdict=font_label_box)
        t1.set_bbox(dict(facecolor='white', alpha=0.7,edgecolor='none'))
        # Steering widget HUD #################################################
        # Steering HUD: Actual steering signal
        steer_actual = ''.join(['|' if v else '-' for v in self.linear_bin(rec['steering_signal'])])
        text_steer = ax.text(80,105,steer_actual,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color=gui_color)
        # Steering HUD: Predicted steering angle
        if 'steering_pred_signal' in self.df.columns:
            steer_pred = ''.join(['◈' if v else ' ' for v in self.linear_bin(rec['steering_pred_signal'])])
            text_steer_pred = ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')
        
        return fig

    def get_one_frame(self,index_ts):
        npz_objs = np.load(self.path_frames_npz)
        return npz_objs[index_ts]

    # =============================================================================
    # Process frames to JPG
    # =============================================================================
    def write_jpgs(self, dir_jpgs=None, overwrite = False):
        """Write pure JPGs to disk from numpy zip file
        
        """
        if not dir_jpgs: dir_jpgs = self.path_jpgs_dir
        
        
        jpg_files = glob.glob(os.path.join(self.path_dataset,dir_jpgs,'*.jpg'))
        if len(jpg_files) == len(self.df) and not overwrite:
            logging.debug("{} jpg files already exist, skip unless overwrite=True".format(len(self.df)))
            return
        
        # Open zip
        arrays = np.load(self.path_frames_npz)
        timestamps = [k for k in arrays.keys()]
        timestamps.sort()
        
        # Create a directory for the JPEGs
        path_jpg = os.path.join(self.path_dataset, dir_jpgs)
        if not os.path.exists(path_jpg):
            os.mkdir(path_jpg)
        
        # Print to .jpg
        for k in tqdm.tqdm(timestamps):
            img = arrays[k]
            arrays[k]
            out_path = os.path.join(path_jpg,'{}.jpg'.format(k))
            cv2.imwrite(out_path, img)
        logging.debug("Wrote {} .jpg to {}".format(len(timestamps),path_jpg))
        #return path_jpg


    def write_jpgs_bw(self, dir_jpgs=None, overwrite = False):
        """Write first channel JPGs to disk from numpy zip file
        
        """
        if not dir_jpgs: dir_jpgs = self.path_jpgs_dir
        
        
        jpg_files = glob.glob(os.path.join(self.path_dataset,dir_jpgs,'*.jpg'))
        if len(jpg_files) == len(self.df) and not overwrite:
            logging.debug("{} jpg files already exist, skip unless overwrite=True".format(len(self.df)))
            return
        
        # Open zip
        arrays = np.load(self.path_frames_npz)
        timestamps = [k for k in arrays.keys()]
        timestamps.sort()
        
        # Create a directory for the JPEGs
        path_jpg = os.path.join(self.path_dataset, dir_jpgs)
        if not os.path.exists(path_jpg):
            os.mkdir(path_jpg)
        
        # Print to .jpg
        for k in tqdm.tqdm(timestamps):
            img = arrays[k]
            #arrays[k]
            img_Y = img[:,:,0]
            out_path = os.path.join(path_jpg,'{}.jpg'.format(k))
            cv2.imwrite(out_path, img_Y)
        logging.debug("Wrote {} .jpg to {}".format(len(timestamps),path_jpg))
        #return path_jpg
    
    def write_frames(self, output_dir_name = 'Video Frames', overwrite=False, blackwhite=False,cmap=None,gui_color='green'):
        """From a JPG image, overlay information with matplotlib, save to disk.
        
        Skip if directory already full. 
        """
        
        OUT_PATH=os.path.join(self.path_dataset,output_dir_name)
        if not os.path.exists(OUT_PATH):
            os.mkdir(OUT_PATH)
        
        jpg_files = glob.glob(os.path.join(OUT_PATH,'*.jpg'))
        if len(jpg_files) == len(self.df) and not overwrite:
            logging.debug("{} jpg files already exist here, skip unless overwrite=True".format(len(self.df)))
            return
        
        logging.debug("Writing frames to {}".format(OUT_PATH))            

        with LoggerCritical(),NoPlots():
            for idx in tqdm.tqdm(self.df.index):
                # Get the frame figure
                if blackwhite:
                    frame_figure = self.gen_record_frame(idx,source_jpg_folder='jpg_images_Y',cmap=cmap,gui_color=gui_color)
                elif not blackwhite:
                    frame_figure = self.gen_record_frame(idx)
            
                # Save it to jpg
                path_jpg = os.path.join(OUT_PATH,idx + '.jpg')
                frame_figure.savefig(path_jpg)
                
        logging.debug("Wrote {} jpg files to {}".format(len(self.df),OUT_PATH))

    
    def zip_jpgs(path_jpg, target_path):
        raise
        jpg_files = glob.glob(os.path.join(path_jpg,'*.jpg'))
        
        with zipfile.ZipFile(target_path, 'w') as myzip:
            for f in jpg_files:
                name = os.path.basename(f)
                myzip.write(f,name)
                os.remove(f)
        logging.debug("Zipped {} to {}".format(len(jpg_files),target_path))
        
    def delete_jpgs(path_jpg):
        raise
        jpg_files = glob.glob(os.path.join(path_jpg,'*.jpg'))
        
        # Remove all .npy files, confirm
        [os.remove(f) for f in jpg_files]
        
        jpg_files = glob.glob(os.path.join(path_jpg,'*.jpg'))
        assert len(jpg_files) == 0
        os.rmdir(path_jpg)
        logging.debug("Deleted all .jpg files".format())

#%% Plotter
class DataSetPlotter:
    def __init__(self):
        pass

    def boxplots_time(self,dataset):
        fig=plt.figure(figsize=PAPER_A4_LAND,facecolor='white')
        fig, axes = plt.subplots(figsize=PAPER_A4_LAND,facecolor='white',nrows=1, ncols=3)

        median = dataset.df['ts_deltas_ms'].median()
        hertz = 1/(median/1000)        
        
        title_str = "{:0.1f} ms, {:0.1f} Hertz (median) for dataset {}".format(median,hertz,dataset.data_folder,dataset.num_records, dataset.elapsed_minutes)
        
        fig.suptitle(title_str,fontsize=20)
        # First plot, get the column as a series
        dataset.df['ts_deltas_ms'].plot.box(ax=axes[0])
        axes[0].yaxis.grid(True)
        axes[0].set_title("Raw time deltas")
        axes[0].set_ylabel("Timestep [ms]")
        
        # Second, remove outliers
        ts_no_outliers1 = remove_outliers(dataset.df['ts_deltas_ms'])
        #time_df['ts_no_outliers1'] = ts_no_outliers1
        ts_no_outliers1.plot.box(ax=axes[1])
        axes[1].set_title("Outliers (3σ) removed")
        axes[1].yaxis.grid(True)
        
        ts_no_outliers2 = remove_outliers(ts_no_outliers1)
        axes[2].set_title("Outliers (3σ) removed again")
        #time_df['ts_no_outliers2'] = ts_no_outliers2
        ts_no_outliers2.plot.box(ax=axes[2])
        axes[2].yaxis.grid(True)
        
        outpath=os.path.join(dataset.path_dataset,'Timestep analysis.png')
        fig.savefig(outpath)
        logging.debug("Wrote boxplots_time figure to {}".format(outpath))
        
        

    
    def histogram_steering(self,dataset):
        fig=plt.figure(figsize=PAPER_A5_LAND,facecolor='white')
        hist_steering = dataset.df['steering_signal'].hist()
        
        title_str = "Histogram of steering signals"
        subtitle_str = "Dataset: {}, {} records over {:0.1f} minutes ".format(dataset.data_folder,dataset.num_records, dataset.elapsed_minutes)
        
        
        hist_steering.set_title(subtitle_str)
        
        fig.suptitle(title_str, fontsize=20)
        #fig.title(subtitle_string, fontsize=10)

        outpath=os.path.join(dataset.path_dataset,'Steering Histogram.png')
        fig.savefig(outpath)
        logging.debug("Wrote histogram_steering figure to {}".format(outpath))
   
    def histogram_throttle(self,dataset):
        fig=plt.figure(figsize=PAPER_A5_LAND,facecolor='white')
        hist_throttle = dataset.df['throttle_signal'].hist()
        title_str = "Histogram of throttle signals"
        subtitle_str = "Dataset: {}, {} records over {:0.1f} minutes ".format(dataset.data_folder,dataset.num_records, dataset.elapsed_minutes)
        hist_throttle.set_title(subtitle_str)
        
        fig.suptitle(title_str, fontsize=20)
        #fig.title(subtitle_string, fontsize=10)

        outpath=os.path.join(dataset.path_dataset,'Throttle Histogram.png')
        fig.savefig(outpath)
        logging.debug("Wrote histogram_throttle figure to {}".format(outpath))
        
    def plot12(self,dataset,ts_string_indices, source_jpg_folder='jpg_images',extension='jpg', rows=3, cols=4, outfname='Sample Frames.png',cmap=None,gui_color='green'):
        """
        Render N records to analysis
        """
        # Settings ############################################################
        font_label_box = {
            'color':'green',
            'size':16,
        }
        font_steering = {'family': 'monospace',
                #'color':  'darkred',
                'weight': 'normal',
                'size': 20,
                }
        ROWS = rows
        COLS = cols
        NUM_IMAGES = ROWS * COLS
        
        # Figure ##############################################################
        # figsize = [width, height]
        fig=plt.figure(figsize=PAPER_A3_LAND,facecolor='white')
        fig.suptitle("Sample frames, Dataset: {}".format(dataset.data_folder), fontsize=20)
        
        
        for i,ts_string_index in enumerate(ts_string_indices):
            rec = dataset.df.loc[ts_string_index]

            timestamp_string = rec['datetime'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['datetime'].microsecond))
            
            if 'steering_pred_signal' in dataset.df.columns:
                this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,
                              rec['steering_signal'],rec['steering_pred_signal'],rec['throttle_signal'])
            else: 
                this_label = "{}\n{:0.2f}/ steering \n{:0.2f} throttle".format(timestamp_string,rec['steering_signal'],rec['throttle_signal'])
                
            ax = fig.add_subplot(ROWS,COLS,i+1)

            # Main Image ##########################################################
            jpg_path = os.path.join(dataset.path_dataset,source_jpg_folder,ts_string_index+'.'+extension)
            assert os.path.exists(jpg_path), "{} does not exist".format(jpg_path)
            img = mpl.image.imread(jpg_path)
            ax.imshow(img,cmap=cmap)
            #plt.title(str_label)
            
            # Data box ########################################################
            
            #ax.axes.get_xaxis().set_visible(False)
            #ax.axes.get_yaxis().set_visible(False)
            t = ax.text(5,25,this_label,color=gui_color,alpha=1)
            #t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
            t.set_bbox(dict(facecolor='white', alpha=0.7,edgecolor='none'))
            
            # Steering widget HUD #################################################
            # Steering HUD: Actual steering signal
            steer_actual = ''.join(['|' if v else '-' for v in dataset.linear_bin(rec['steering_signal'])])
            text_steer = ax.text(80,105,steer_actual,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color=gui_color)
            # Steering HUD: Predicted steering angle
            if 'steering_pred_signal' in dataset.df.columns:
                steer_pred = ''.join(['◈' if v else ' ' for v in dataset.linear_bin(rec['steering_pred_signal'])])
                text_steer_pred = ax.text(80,95,steer_pred,fontdict=font_steering,horizontalalignment='center',verticalalignment='center',color='red')


        outpath=os.path.join(dataset.path_dataset,outfname)
        fig.savefig(outpath)
        logging.debug("Wrote Sample Frames figure to {}".format(outpath))
    
    def plot_sample_frames(self,dataset):
        
        # Right turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] > 0.9)
        these_indices = dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Left turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] < -0.9)
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Straight
        this_mask = dataset.mask & ((dataset.df['steering_signal']  > -0.1) & (dataset.df['steering_signal']  < 0.1))
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()
        
        #return these_indices
        self.plot12(dataset,these_indices)

        # This is a pointer to the file
        #npz_file=np.load(dataset.path_frames_npz)
        
        #frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
        #logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))
        
        #return frames_array


    def plot_sample_frames_bw(self,dataset):
        
        # Right turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] > 0.9)
        these_indices = dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Left turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] < -0.9)
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Straight
        this_mask = dataset.mask & ((dataset.df['steering_signal']  > -0.1) & (dataset.df['steering_signal']  < 0.1))
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()
        
        #return these_indices
        self.plot12(dataset=dataset,ts_string_indices=these_indices,source_jpg_folder='jpg_images_Y',extension='jpg', rows=3, cols=4, outfname='Sample Frames Y.png',cmap='bwr',gui_color='black')
        
        # This is a pointer to the file
        #npz_file=np.load(dataset.path_frames_npz)
        
        #frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
        #logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))
        
        #return frames_array
    
    def __get_records(self,batch_indices):
        """Custom method - get the y labels
        """
        this_batch_df = self.dataset.df.loc[batch_indices]
        steering_values = this_batch_df['steering_signal'].values
        steering_records_array = self.dataset.bin_Y(steering_values)
        logging.debug("Generating {} records {}:".format(steering_records_array.shape[0],steering_records_array.shape))
        return steering_records_array
        
    def __data_generation(self, batch_indices):
        """Keras generator method - Generates data containing batch_size samples
        """

        X = self.__get_npy_arrays(batch_indices)
        y = self.__get_records(batch_indices)

        return X, y
  


#%%

class ModelledData():
    """Augment a dataset with training
    
    ds - The dataset object, as defined
    ds.df - The dataframe
    ds.mask - The mask
    """
    
    def __init__(self,dataset,model_folder):
        
        #super().__init__(path_data,data_folder)
        self.ds = dataset
        
        self.model_folder = model_folder
        
        # Create the model folder, or reference it
        self.path_model_dir = os.path.join(self.ds.path_data,self.ds.data_folder,self.model_folder)
        if not os.path.exists(self.path_model_dir):
            os.makedirs(self.path_model_dir)
            logging.debug("Created a NEW model folder at {}/{}".format(self.ds.data_folder,self.model_folder))
        else:
            logging.debug("Model folder at {}/{}".format(self.ds.data_folder,model_folder))
            
        #assert not os.listdir(self.path_model_dir), "{} not empty".format(self.path_model_dir)
        #logging.debug("This model exists in {}".format(model_dir))
        
        self.callback_list = None
        
        if not self.model_folder_empty:
            self.list_models()
            
    @classmethod
    def from_dataset(self,dataset,model_folder):
        self = dataset
        
        self.model_folder = model_folder

        
        # Create the model folder, or reference it
        self.path_model_dir = os.path.join(self.path_data,self.data_folder,self.model_folder)
        if not os.path.exists(self.path_model_dir):
            os.makedirs(self.path_model_dir)
            logging.debug("Created a NEW model folder at {}/{}".format(self.data_folder,self.model_folder))
        else:
            logging.debug("Model folder at {}/{}".format(self.data_folder,model_folder))

        
        return self
    
    def list_models(self):
        search_str = os.path.join(self.path_model_dir,'*.h5')
        #print(search_str)
        paths_weights = glob.glob(search_str)
        logging.debug("{} weights found".format(len(paths_weights)))
        model_files = list()
        for this_wt_path in paths_weights:
            _,fname = os.path.split(this_wt_path)
            basename, ext = os.path.splitext(fname)
            #print(basename)
            loss_string = re.search(r"Loss [-+]?[0-9]*\.?[0-9]+",basename)[0]
            loss_num = float(re.search("[-+]?[0-9]*\.?[0-9]+",loss_string)[0])            
            #print(loss_num)
            model_files.append({'path':this_wt_path, 'loss':loss_num, 'fname':basename})
        
        model_files = sorted(model_files, key=lambda k: k['loss'])
        for mf in model_files:
            #print(mf['fname'],mf['loss'])
            pass
        return model_files
            
    @property
    def model_folder_empty(self):
        if os.listdir(self.path_model_dir): return False
        else:  return True
    
    @property
    def has_predictions(self):
        return 'steering_pred_signal' in self.ds.df.columns
    
    def generate_partitions(self, split=0.8):
        logging.debug("Partitioning  train/val to {:0.0f}/{:0.0f}%".format(split*100, (1-split)*100))
        
        # The split mask
        msk = np.random.rand(len(self.ds.df)) < split
        
        # Aggregate the split with the overall mask
        mask_tr = msk & self.ds.mask
        mask_val = ~msk & self.ds.mask
        
        self.partition = dict()
        self.partition['train'] = self.ds.df.index[mask_tr].values
        self.partition['validation'] = self.ds.df.index[mask_val].values
        
        tr_pct = len(self.partition['train'])/(len(self.partition['train'])+len(self.partition['validation']))*100
        val_pct = len(self.partition['validation'])/(len(self.partition['train'])+len(self.partition['validation']))*100
        logging.debug("Actual split: {:0.1f}/{:0.1f}% over {:0.1f}% of the total records".format(tr_pct,val_pct,self.ds.mask_cover_pct))
    
    def instantiate_generators(self,generator_class, generator_params = None):
        if not generator_params:
            generator_params = {'dim': (160,120),
                      'batch_size': 64,
                      'n_classes': 15,
                      'n_channels': 3,
                      'shuffle': True,
                      #'path_frames':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'),
                      #'path_records':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'),
                     }
       
        self.training_generator = generator_class(self.partition['train'], self.ds, **generator_params)
        self.validation_generator = generator_class(self.partition['validation'], self.ds, **generator_params)        
        logging.debug("training_generator and validation_generator instantiated".format())

    def instantiate_model(self,model_name="baseline_steering_model"):
        def baseline_steering_model():
            model = ks.models.Sequential()
            model.add(ks.layers.Conv2D(24, (5,5), strides=(2, 2), activation = "relu", input_shape=(120,160,3)))
            model.add(ks.layers.Conv2D(32, (5,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (5,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3,3), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3,3), strides=(1, 1), activation = "relu"))
            model.add(ks.layers.Flatten()) # This is just a reshape!
            model.add(ks.layers.Dense(100,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(50,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(15, activation='softmax', name='angle_out'))
            return model
        
        def blackwhite_steering_model():
            model = ks.models.Sequential()
            model.add(ks.layers.Conv2D(24, (5,5), strides=(2, 2), activation = "relu", input_shape=(120,160,1)))
            model.add(ks.layers.Conv2D(32, (5,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (5,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3,3), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3,3), strides=(1, 1), activation = "relu"))
            model.add(ks.layers.Flatten()) # This is just a reshape!
            model.add(ks.layers.Dense(100,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(50,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(15, activation='softmax', name='angle_out'))
            return model
                
        # The library of models
        #TODO: Move this outside? 
        MODEL_ARCHITECTURE_MAP = {
                "baseline_steering_model":baseline_steering_model,
                "blackwhite_steering_model":blackwhite_steering_model}
        model_architecture = MODEL_ARCHITECTURE_MAP[model_name]
        optimizer = ks.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        model = model_architecture()
        model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=[ks.metrics.categorical_accuracy]
                       )
        
        self.model = model
        logging.debug("Model created".format())
        
    def instantiate_callbacks(self):
        # Save checkpoint
        weight_filename="weights Loss {val_loss:.2f} Epoch {epoch:02d}.h5"
        weight_path = os.path.join(self.path_model_dir,weight_filename)
        callback_wts = ks.callbacks.ModelCheckpoint(weight_path, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='min')
        # Early stopping
        callback_stopping = ks.callbacks.EarlyStopping(monitor='val_loss', 
                                                       min_delta=0.0005, 
                                                       patience=5, # number of epochs with no improvement after which training will be stopped.
                                                       verbose=1, 
                                                       mode='auto')
        
        # Logger
        this_log_path = os.path.join(self.path_model_dir,"history.log".format())
        callback_logger = ks.callbacks.CSVLogger(this_log_path,separator=',', append=True)
        
        
        class MyCallback(ks.callbacks.Callback):
            def __init__(self,model_folder_path):
                self.model_folder_path = model_folder_path
                
            def on_train_begin(self, logs={}):
                logging.info("Started training {}".format(self.model))
                self.losses = []
                return 
                
            def on_train_end(self, logs={}):
                logging.info("Finished training {}".format(self.model))
                return
         
        
            def on_epoch_begin(self, epoch, logs={}):
                logging.info("Epoch {} {}".format(epoch,logs))
                return
         
            def on_batch_end(self, batch, logs={}):
                #self.losses.append(logs.get('loss'))
                logging.debug("\tBatch {} {}".format(batch,logs))
                pass
        
                
            def on_epoch_end(self, epoch, logs={}):
                #print("HISTORY:{}".format(self.model.history.history))
                #print("EPOCH:", epoch)
                #print("LOGS:",logs)
                
                
                this_path = os.path.join(self.model_folder_path,"History epoch {:02d}.json".format(epoch))
                with open(this_path, 'w') as fh:
                    json.dump(self.model.history.history, fh)
                print("Saved history to {}".format(this_path))

                
        my_history_callback = MyCallback(self.path_model_dir)
        self.callback_list = [callback_wts,callback_stopping,callback_logger]        
        
        logging.debug("{} callbacks created".format(len(self.callback_list)))

    def train_model(self, epochs=10):
        assert self.model_folder_empty, "The model folder {} is not empty, instantiate a new model!".format(self.model_folder)        
        with LoggerCritical():
            history = self.model.fit_generator(
                    generator=self.training_generator,
                    validation_data=self.validation_generator,
                    use_multiprocessing=True,
                    workers=6,
                    epochs=epochs,
                    verbose=1,
                    callbacks=self.callback_list)
        
        self.history_dict = history.__dict__
        
        #this_timestamp = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
        logging.debug("Finished training model {}".format(self.model_folder))

    def load_best_model(self):
        model_def = self.list_models()[0]
        path_model = model_def['path']
        assert os.path.exists(path_model)
        
        self.model = ks.models.load_model(path_model)
        
        logging.debug("Loaded weights {} with loss {}".format(model_def['fname'],model_def['loss']))

    def load_spec_model(self,model_id=None):
        # Load a specific weights file
        raise
        logging.debug("Loading model {} from {}".format(model_id, path_model))
        path_model = os.path.join(self.path_model_dir,model_id)
        pred_model = ks.models.load_model(path_model)
        
    #def make_predictions(self,model_id=None):
    #    logging.debug("Predicting over self.model {}".format(self.model))
        #self.make_model_predictions(self.model)
        
    def make_predictions(self):
        """Augment the df_records with the predictions
        """
        #print(self.df.head())
        #this_df_records['steering_pred_cats'] = pd.Series(dtype=object)
        #df_records['steering_pred_argmax'] = 
        
        # get all the X array (all numpy arrays), in *proper* order
        
        #
        logging.debug("Predicting over self.model {}".format(self.model))

        npz_file = np.load(self.ds.path_frames_npz)
        #frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
        frames_array = np.stack([npz_file[idx] for idx in self.ds.df.index], axis=0)
        #print(arrays)
        logging.debug("All images loaded as 1 numpy array {}".format(frames_array.shape))
        logging.debug("Starting predictions ...".format(frames_array.shape))
        
        #predictions_cats = self.model.predict(frames_array,verbose=1)
        predictions_cats = self.model.predict(frames_array,verbose=1)
        
        logging.debug("Predictions complete, shape: {}".format(predictions_cats.shape))
        #logging.debug("Saved categories to column steering_pred_signal_cats".format())
        
        predictions = self.ds.unbin_Y(predictions_cats)
        #logging.debug("Predictions unbinned, shape: {}".format(predictions.shape))
        
        self.ds.df['steering_pred_signal'] = predictions
        logging.debug("Predictions added to df in column {}".format('steering_pred_signal'))
        
        # Get the category of this steering signal
        self.ds.df['steering_pred_signal_catnum'] = self.ds.signal_to_category_number('steering_pred_signal')
        
        self.raw_accuracy =  sum(self.ds.df[self.ds.mask]['steering_signal_catnum'] == self.ds.df[self.ds.mask]['steering_pred_signal_catnum'])/len(self.ds.df[self.ds.mask])
        logging.debug("Raw accuracy {:0.2f}%".format(self.raw_accuracy*100))
        
        #return predictions

    def save_predictions(self,path_out):
        assert 'steering_pred_signal' in self.ds.df.columns 
        assert 'steering_pred_signal_catnum' in self.ds.df.columns 
        pass



#%%
#these_records = these_records[0:100]



class SaliencyGen():
    """Aggregates the ModelledDataSet class, and operates to produce frames
    """
    def __init__(self,modelled_dataset):
        # Get the model and dataset (modelleddataset)
        self.modelled_dataset = modelled_dataset
        assert modelled_dataset.has_predictions
        self.model_folder = modelled_dataset.model_folder
        self.path_model_dir = modelled_dataset.path_model_dir
        logging.debug("Saliency gen for model ID: {}".format(self.model_folder))
        logging.debug("Loaded model accuracy: {:0.1f}%".format(self.modelled_dataset.raw_accuracy*100))
        
        # Original raw images
        self.path_jpgs_dir = modelled_dataset.ds.path_jpgs_dir
        logging.debug("Source orginal raw images are in folder: {}".format(self.path_jpgs_dir))
        files = glob.glob(os.path.join(self.path_jpgs_dir,'*.jpg'))
        logging.debug("{} jpgs found".format(len(files)))
        
        # New saliency mask jpgs
        self.path_saliency_jpgs = os.path.join(self.path_model_dir,'imgs_saliency_masks')
        if not os.path.exists(self.path_saliency_jpgs): 
            os.makedirs(self.path_saliency_jpgs)
        logging.debug("Saliency JPG output folder: {}".format(self.path_saliency_jpgs))
        files = glob.glob(os.path.join(self.path_saliency_jpgs,'*.*'))
        logging.debug("{} files found".format(len(files)))        
        
        # Boosted saliency mask jpgs
        self.path_boosted_saliency_jpgs = os.path.join(self.path_model_dir,'imgs_saliency_masks_boosted')
        if not os.path.exists(self.path_boosted_saliency_jpgs): 
            os.makedirs(self.path_boosted_saliency_jpgs)
        logging.debug("BOOSTED Saliency JPG output folder: {}".format(self.path_boosted_saliency_jpgs))
        files = glob.glob(os.path.join(self.path_boosted_saliency_jpgs,'*.*'))
        logging.debug("{} files found".format(len(files)))        
        
        # New saliency frames
        self.path_frames_jpgs = os.path.join(self.path_model_dir,'frames_saliency_boosted')
        if not os.path.exists(self.path_frames_jpgs): 
            os.makedirs(self.path_frames_jpgs)        
        logging.debug("Combined HUD frames output folder: {}".format(self.path_frames_jpgs))
        files = glob.glob(os.path.join(self.path_frames_jpgs,'*.*'))
        logging.debug("{} files found".format(len(files)))
        
    def gen_pure_CNN(self):
        # Get a pure convolutional model, no dropout or other layers
        img_in = ks.layers.Input(shape=(120, 160, 3), name='img_in')
        x = img_in
        x = ks.layers.Convolution2D(24, (5,5), strides=(2,2), activation='relu', name='conv1')(x)
        x = ks.layers.Convolution2D(32, (5,5), strides=(2,2), activation='relu', name='conv2')(x)
        x = ks.layers.Convolution2D(64, (5,5), strides=(2,2), activation='relu', name='conv3')(x)
        x = ks.layers.Convolution2D(64, (3,3), strides=(2,2), activation='relu', name='conv4')(x)
        conv_5 = ks.layers.Convolution2D(64, (3,3), strides=(1,1), activation='relu', name='conv5')(x)
        convolution_part = ks.models.Model(inputs=[img_in], outputs=[conv_5])
        self.convolutional_model = convolution_part
        logging.debug("Generated a pure CNN {}".format(self.convolutional_model))

    def get_layers(self):
        # Get each layer of the pure Conv model
        # Assign the weights from the trained model
        logging.debug("Retreiving and copying weights from the model, to the pure CNN".format())
        for layer_num in ('1', '2', '3', '4', '5'):
            
            this_pureconv_layer = self.convolutional_model.get_layer('conv' + layer_num)
            this_layer_name = 'conv2d_' + layer_num
            #print("Copied weights from loaded", this_layer_name, "to the pure CNN")
            these_weights = self.modelled_dataset.model.get_layer(this_layer_name).get_weights()
            this_pureconv_layer.set_weights(these_weights)
        logging.debug("Assigned trained model weights to all convolutional layers".format())


    def saliency_tf_function(self):
        inp = self.convolutional_model.input                                           # input placeholder
        outputs = [layer.output for layer in self.convolutional_model.layers]          # all layer outputs
        saliency_function = ks.backend.function([inp], outputs)
        self.saliency_function = saliency_function
        logging.debug("Created tensorflow pipeliine (saliency_function) from weighted convolutional layers".format())

    def get_kernels(self):
        """ Recreate the kernels and strides for each layer
        """
        kernel_3x3 = tf.constant(np.array([
                [[[1]], [[1]], [[1]]], 
                [[[1]], [[1]], [[1]]], 
                [[[1]], [[1]], [[1]]]
        ]), tf.float32)
        
        kernel_5x5 = tf.constant(np.array([
                [[[1]], [[1]], [[1]], [[1]], [[1]]], 
                [[[1]], [[1]], [[1]], [[1]], [[1]]], 
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[1]], [[1]], [[1]], [[1]]]
        ]), tf.float32)
        
        self.layers_kernels = {5: kernel_3x3, 4: kernel_3x3, 3: kernel_5x5, 2: kernel_5x5, 1: kernel_5x5}
        
        self.layers_strides = {5: [1, 1, 1, 1], 4: [1, 2, 2, 1], 3: [1, 2, 2, 1], 2: [1, 2, 2, 1], 1: [1, 2, 2, 1]}
            
        logging.debug("Assigned layers_kernels and layers_strides".format())
        
    def write_saliency_mask_jpgs(self,number=None):
        frames_npz = np.load(self.modelled_dataset.ds.path_frames_npz)
        if not number:
            number = len(self.modelled_dataset.ds.df)
            
        with LoggerCritical(), NoPlots():
            for idx in tqdm.tqdm(self.modelled_dataset.ds.df.index[0:number]):
                rec = self.modelled_dataset.ds.df.loc[idx]
                path_out = os.path.join(self.path_saliency_jpgs,rec['timestamp']+'.png')
                if os.path.exists(path_out): continue

                #print(idx,rec)
                
                # Get a frame array, and shape it to 4D
                img_array = frames_npz[idx]
                img_array = np.expand_dims(img_array, axis=0)
                activations = self.saliency_function([img_array])
                
                # The upscaled activation changes each loop (layer)
                upscaled_activation = np.ones((3, 6))
                for layer in [5, 4, 3, 2, 1]:
                    averaged_activation = np.mean(activations[layer], axis=3).squeeze(axis=0) * upscaled_activation
                    output_shape = (activations[layer - 1].shape[1], activations[layer - 1].shape[2])
                    x = tf.constant(
                        np.reshape(averaged_activation, (1,averaged_activation.shape[0],averaged_activation.shape[1],1)),
                        tf.float32
                    )
                    conv = tf.nn.conv2d_transpose(
                        x, self.layers_kernels[layer],
                        output_shape=(1,output_shape[0],output_shape[1], 1), 
                        strides=self.layers_strides[layer], 
                        padding='VALID'
                    )
                    with tf.Session() as session:
                        result = session.run(conv)
                    upscaled_activation = np.reshape(result, output_shape)
                    
                    salient_mask = (upscaled_activation - np.min(upscaled_activation))/(np.max(upscaled_activation) - np.min(upscaled_activation))
                    
                    # Make an RGB 3-channel image            
                    salient_mask_stacked = np.dstack((salient_mask,salient_mask,salient_mask))
                    
                    # Save it to JPG
                    plt.imsave(path_out, salient_mask_stacked)
                    
                    #ks.backend.clear_session()

                
    def blend_simple(self,blur_rad,strength,num_frames = None):
        #
        logging.debug("blur_rad {}, strength {}".format(blur_rad,strength))
        
        source_folder = os.path.split(self.path_saliency_jpgs)[1]
        target_folder = os.path.split(self.path_boosted_saliency_jpgs)[1]
        jpg_files = glob.glob(os.path.join(self.path_saliency_jpgs,'*.png'))
        logging.debug("Boosting {} frames at {} to {}".format(len(jpg_files),source_folder,target_folder))
        
        frames_npz = np.load(self.modelled_dataset.ds.path_frames_npz)
        
        # For testing, write a sample
        if not num_frames:
            num_frames = len(jpg_files)
            
        with LoggerCritical(), NoPlots():
            for img_path in tqdm.tqdm(jpg_files[0:num_frames]):
                
                #print(img_path)
                _,fname = os.path.split(img_path)
                path_out = os.path.join(self.path_boosted_saliency_jpgs,fname)
                if os.path.exists(path_out): continue

                timestamp,_ = os.path.splitext(fname)
                #print(timestamp)
                saliency_frame = plt.imread(img_path)[:,:,:3]
                raw_frame = frames_npz[timestamp]
                
                if 0: # Try adjusting brightness and contrast...
                    b = 0. # brightness
                    c = 64.  # contrast
                
                    #call addWeighted function, which performs:
                    #    dst = src1*alpha + src2*beta + gamma
                    # we use beta = 0 to effectively only operate on src1
                    saliency_frame = cv2.addWeighted(saliency_frame, 1. + c/127., saliency_frame, 0, b-c)                                
                
                saliency_frame.setflags(write=1)
                saliency_frame[:,:,0] = saliency_frame[:,:,0] * 1.5
                saliency_frame[:,:,1] = saliency_frame[:,:,1] * 0
                saliency_frame[:,:,2] = saliency_frame[:,:,2] * 0
                blur_kernel = np.ones((blur_rad,blur_rad),np.float32) * strength
                saliency_frame_blurred = cv2.filter2D(saliency_frame,-1,blur_kernel)


                #saliency_frame_blurred = cv2.GaussianBlur(saliency_frame,(1,1),0)
                
                alpha = 0.004
                beta = 1.0 - alpha
                gamma = 0.0
                try:
                    blend = cv2.addWeighted(raw_frame.astype(np.float32), alpha, saliency_frame_blurred, beta, gamma)
                except:
                    print("BAD IMAGE?", img_path)
                    raise
                #plt.imshow(blend)
                
                
                
                plt.imsave(path_out, blend)

        # Raw masks
        pass
    
    def create_HUD_frames(self):
        source_folder_jpg = os.path.split(self.path_saliency_jpgs)[1]
        target_folder = os.path.split(self.path_frames_jpgs)[1]
        jpg_files = glob.glob(os.path.join(self.path_boosted_saliency_jpgs,'*.png'))
        logging.debug("Creating {} HUD frames from {} to {}".format(len(jpg_files),source_folder_jpg,target_folder))        

        with LoggerCritical(), NoPlots():
            for img_path in tqdm.tqdm(glob.glob(self.path_boosted_saliency_jpgs + r"/*.png")):
                #print(img_path)
                _,fname = os.path.split(img_path)

                index,_ = os.path.splitext(fname)
                pathpart_source_imgs = self.model_folder + r"/" + 'imgs_saliency_masks_boosted'
                path_jpg = os.path.join(self.path_frames_jpgs, index + ".jpg")
                if os.path.exists(path_jpg): continue

                frame_figure = this_saliency.modelled_dataset.ds.gen_record_frame(index, source_jpg_folder=pathpart_source_imgs, source_ext = '.png')
                ks.backend.clear_session()

                # Save it to jpg
                #path_jpg = os.path.join(OUT_PATH,idx + '.jpg')
                frame_figure.savefig(path_jpg)
        logging.debug("Wrote frames to {}".format(self.path_frames_jpgs))
        

    def blend_PIL(self,blur_rad,map_name,strength):
        """A more advanced boosting pipeline
        """
        logging.debug("blur_rad {}, map_name {}, strength {}".format(blur_rad,map_name,strength))

        pass
