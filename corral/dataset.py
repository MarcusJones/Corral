# %% Data set class
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

    def __init__(self, path_data, data_folder):
        # Check the data folder
        self.path_data = path_data
        assert os.path.exists(self.path_data)
        self.data_folder = data_folder
        self.path_dataset = os.path.join(self.path_data, self.data_folder)
        assert os.path.exists(self.path_dataset)

        logging.debug("Data set {}, recorded on {}".format(self.data_folder, self.datetime_string))

        # Check the raw records zip, load to DataFrame
        self.path_records_zip = os.path.join(self.path_dataset, "json_records.zip")
        assert os.path.exists(self.path_records_zip)
        self.df = self.load_records_df()
        logging.debug("Records {}".format(len(self.df)))
        self.df['steering_signal_catnum'] = self.signal_to_category_number('steering_signal')

        # Check the raw frames zip, no need to unzip
        self.path_frames_npz = os.path.join(self.path_dataset, "camera_numpy.zip")
        assert os.path.exists(self.path_frames_npz)
        frames_timestamps = self.get_frames_timesteps()
        logging.debug("Frames npz is {:0.2f} MB, {} records".format(self.frames_size, len(frames_timestamps)))

        # Assert timestep alignment
        assert all(self.df.index == frames_timestamps), "Misaligned timestamps"

        # JPG folder
        JPG_FOLDER_NAME = "jpg_images"
        self.path_jpgs_dir = os.path.join(self.path_dataset, JPG_FOLDER_NAME)

        self.mask = pd.Series(True, index=self.df.index)

        # Predictions
        # self.predicted_model = None
        # self.
        self.augment_df_datetime()

    # =============================================================================
    # --- Query
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
        # print(self.path_frames_npz)
        return os.path.getsize(self.path_frames_npz) / 1000 / 1000

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
        # datetime_stamps = [datetime.datetime.fromtimestamp(int(ts)/1000) for ts in timestamps]
        # SORT!
        # datetime_stamps.sort()

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
    def int_index(self, timestamp):
        # Helper to swap timestamp string <> integer index on df
        return self.df[self.df['timestamp'] == timestamp]

    @property
    def timestamp(self, int_index):
        # Helper to swap timestamp string <> integer index on df
        return self.df[int_index]['timestamp']

    # =============================================================================
    # --- Utility
    # =============================================================================
    @property
    def mask_cover_pct(self):
        return 100 - sum(self.mask) / len(self.mask) * 100

    def mask_first_Ns(self, numsecs=3):
        ds = self
        first_datetime = datetime.datetime.fromtimestamp(int(ds.df.index[0]) / 1000)
        assert ds.df['datetime'][0] == first_datetime
        # Get a timedelta (days,seconds)
        tdelta = datetime.timedelta(0, numsecs)
        second_datetime = ds.df['datetime'][0] + tdelta
        truemask = (ds.df['datetime'] >= first_datetime) & (ds.df['datetime'] <= second_datetime)
        this_mask = ~truemask
        self.mask = self.mask & this_mask
        logging.debug(
            "Masked {} timesteps from {} to {}, current cover: {:0.1f}%".format(sum(this_mask), first_datetime,
                                                                                second_datetime, self.mask_cover_pct))

    def mask_last_Ns(self, numsecs=2):
        ds = self
        last_datetime = datetime.datetime.fromtimestamp(int(ds.df.index[-1]) / 1000)
        assert ds.df['datetime'][-1] == last_datetime
        # Get a timedelta (days,seconds)
        tdelta = datetime.timedelta(0, numsecs)
        start_datetime = ds.df['datetime'][-1] - tdelta
        truemask = (ds.df['datetime'] >= start_datetime) & (ds.df['datetime'] <= last_datetime)
        this_mask = ~truemask
        self.mask = self.mask & this_mask
        logging.debug("Masked {} timesteps from {} to {}, current cover: {:0.1f}%".format(sum(truemask), start_datetime,
                                                                                          last_datetime,
                                                                                          self.mask_cover_pct))

    def mask_null_throttle(self, cutoff=0.1):
        ds = self
        truemask = ds.df['throttle_signal'] <= cutoff
        this_mask = ~truemask
        self.mask = self.mask & this_mask
        logging.debug("Masked {} timesteps throttle<{}, current cover: {:0.1f}%".format(sum(truemask), cutoff,
                                                                                        self.mask_cover_pct))

    def mask_(self, first_ts, second_ts):
        ds = self
        # first_datetime = datetime.datetime.fromtimestamp(int(ds.df.index[0])/1000)
        # assert ds.df['datetime'][0] == first_datetime
        # Get a timedelta (days,seconds)
        # tdelta = datetime.timedelta(0,numsecs)
        # second_datetime = ds.df['datetime'][0] + tdelta
        truemask = (ds.df.index >= first_ts) & (ds.df.index >= second_ts)
        this_mask = ~truemask
        self.mask = self.mask & this_mask
        logging.debug(
            "Masked {} timesteps from {} to {}, current cover: {:0.1f}%".format(sum(this_mask), first_datetime,
                                                                                second_datetime, self.mask_cover_pct))

    # Conversion between categorical and floating point steering
    def linear_bin(self, a):
        a = a + 1
        b = round(a / (2 / 14))
        arr = np.zeros(15)
        arr[int(b)] = 1
        return arr

    def linear_unbin(self, arr):
        if not len(arr) == 15:
            raise ValueError('Illegal array length, must be 15')
        b = np.argmax(arr)
        a = b * (2 / 14) - 1
        return a

    def bin_Y(self, Y):
        d = [self.linear_bin(y) for y in Y]
        return np.array(d)

    def unbin_Y(self, Y):
        d = [self.linear_unbin(y) for y in Y]
        return np.array(d)

    def signal_to_category_number(self, column_name):
        """Break the floating point [-1,1] signal into bins
        """
        cats = self.bin_Y(self.df[column_name])
        # Get the category number
        return np.argmax(cats, axis=1)

    # =============================================================================
    # --- Load into memory
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
            json_file_paths = [name for name in f.namelist() if os.path.splitext(name)[1] == '.json']
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
        # .reset_index(drop=True)
        this_df['steering_signal'] = this_df['steering_signal'].apply(lambda x: x * -1)
        logging.debug("Steering signal inverterted - WHY?".format())

        return this_df
        # return pd.DataFrame(json_records).sort_values(by='timestamp').reset_index(drop=True)

    # =============================================================================
    # --- Timestep analysis and processing
    # =============================================================================
    def augment_df_datetime(self):
        def convert_datetime(x):
            return datetime.datetime.fromtimestamp(int(x) / 1000)
            # return datetime.datetime.strptime(x, '%Y%m%d %H%M%S')

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
        ts_deltas = (self.df['datetime'] - self.df['datetime'].shift()).fillna(0)

        self.df['ts_deltas_ms'] = ts_deltas.apply(lambda x: x.total_seconds() * 1000)

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

    # --- Video
    # =============================================================================
    def gen_record_frame(self, ts_string_index, source_jpg_folder='jpg_images', source_ext='.jpg', cmap=None,
                         gui_color='green'):
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
            'color': gui_color,
            'size': 16,
        }
        font_steering = {'family': 'monospace',
                         # 'color':  'darkred',
                         'weight': 'normal',
                         'size': 45,
                         }
        SCALE = 50
        HEIGHT_INCHES = 160 * 2.54 / SCALE
        WIDTH_INCHES = 120 * 2.54 / SCALE

        # Figure ##############################################################
        fig = plt.figure(frameon=False, figsize=(HEIGHT_INCHES, WIDTH_INCHES))
        ax = mpl.axes.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Main Image ##########################################################
        jpg_path = os.path.join(self.path_dataset, source_jpg_folder, ts_string_index + source_ext)
        # print(self.path_dataset)
        # print(source_jpg_folder)
        # print(ts_string_index+source_ext)
        assert os.path.exists(os.path.join(self.path_dataset, source_jpg_folder)), "Does not exist: {}".format(
            os.path.join(self.path_dataset, source_jpg_folder))
        img = mpl.image.imread(jpg_path)
        ax.imshow(img, cmap)
        # raise

        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)

        # Data box ########################################################
        timestamp_string = rec['datetime'].strftime("%D %H:%M:%S.") + "{:.2}".format(str(rec['datetime'].microsecond))
        if 'steering_pred_signal' in self.df.columns:
            this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,
                                                                                  rec['steering_signal'],
                                                                                  rec['steering_pred_signal'],
                                                                                  rec['throttle_signal'])
        else:
            this_label = "{}\n{:0.2f}/ steering \n{:0.2f} throttle".format(timestamp_string, rec['steering_signal'],
                                                                           rec['throttle_signal'])
        t1 = ax.text(2, 15, this_label, fontdict=font_label_box)
        t1.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))
        # Steering widget HUD #################################################
        # Steering HUD: Actual steering signal
        steer_actual = ''.join(['|' if v else '-' for v in self.linear_bin(rec['steering_signal'])])
        text_steer = ax.text(80, 105, steer_actual, fontdict=font_steering, horizontalalignment='center',
                             verticalalignment='center', color=gui_color)
        # Steering HUD: Predicted steering angle
        if 'steering_pred_signal' in self.df.columns:
            steer_pred = ''.join(['◈' if v else ' ' for v in self.linear_bin(rec['steering_pred_signal'])])
            text_steer_pred = ax.text(80, 95, steer_pred, fontdict=font_steering, horizontalalignment='center',
                                      verticalalignment='center', color='red')

        return fig

    def get_one_frame(self, index_ts):
        npz_objs = np.load(self.path_frames_npz)
        return npz_objs[index_ts]

    # =============================================================================
    # Process frames to JPG
    # =============================================================================
    def write_jpgs(self, dir_jpgs=None, overwrite=False):
        """Write pure JPGs to disk from numpy zip file

        """
        if not dir_jpgs: dir_jpgs = self.path_jpgs_dir

        jpg_files = glob.glob(os.path.join(self.path_dataset, dir_jpgs, '*.jpg'))
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
            out_path = os.path.join(path_jpg, '{}.jpg'.format(k))
            cv2.imwrite(out_path, img)
        logging.debug("Wrote {} .jpg to {}".format(len(timestamps), path_jpg))
        # return path_jpg

    def write_jpgs_bw(self, dir_jpgs=None, overwrite=False):
        """Write first channel JPGs to disk from numpy zip file

        """
        if not dir_jpgs: dir_jpgs = self.path_jpgs_dir

        jpg_files = glob.glob(os.path.join(self.path_dataset, dir_jpgs, '*.jpg'))
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
            # arrays[k]
            img_Y = img[:, :, 0]
            out_path = os.path.join(path_jpg, '{}.jpg'.format(k))
            cv2.imwrite(out_path, img_Y)
        logging.debug("Wrote {} .jpg to {}".format(len(timestamps), path_jpg))
        # return path_jpg

    def write_frames(self, output_dir_name='Video Frames', overwrite=False, blackwhite=False, cmap=None,
                     gui_color='green'):
        """From a JPG image, overlay information with matplotlib, save to disk.

        Skip if directory already full.
        """

        OUT_PATH = os.path.join(self.path_dataset, output_dir_name)
        if not os.path.exists(OUT_PATH):
            os.mkdir(OUT_PATH)

        jpg_files = glob.glob(os.path.join(OUT_PATH, '*.jpg'))
        if len(jpg_files) == len(self.df) and not overwrite:
            logging.debug("{} jpg files already exist here, skip unless overwrite=True".format(len(self.df)))
            return

        logging.debug("Writing frames to {}".format(OUT_PATH))

        with LoggerCritical(), NoPlots():
            for idx in tqdm.tqdm(self.df.index):
                # Get the frame figure
                if blackwhite:
                    frame_figure = self.gen_record_frame(idx, source_jpg_folder='jpg_images_Y', cmap=cmap,
                                                         gui_color=gui_color)
                elif not blackwhite:
                    frame_figure = self.gen_record_frame(idx)

                # Save it to jpg
                path_jpg = os.path.join(OUT_PATH, idx + '.jpg')
                frame_figure.savefig(path_jpg)

        logging.debug("Wrote {} jpg files to {}".format(len(self.df), OUT_PATH))

    def zip_jpgs(path_jpg, target_path):
        raise
        jpg_files = glob.glob(os.path.join(path_jpg, '*.jpg'))

        with zipfile.ZipFile(target_path, 'w') as myzip:
            for f in jpg_files:
                name = os.path.basename(f)
                myzip.write(f, name)
                os.remove(f)
        logging.debug("Zipped {} to {}".format(len(jpg_files), target_path))

    def delete_jpgs(path_jpg):
        raise
        jpg_files = glob.glob(os.path.join(path_jpg, '*.jpg'))

        # Remove all .npy files, confirm
        [os.remove(f) for f in jpg_files]

        jpg_files = glob.glob(os.path.join(path_jpg, '*.jpg'))
        assert len(jpg_files) == 0
        os.rmdir(path_jpg)
        logging.debug("Deleted all .jpg files".format())