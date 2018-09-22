import matplotlib.pyplot as plt
# %% Plotter
class DataSetPlotter:
    def __init__(self):
        pass

    def boxplots_time(self ,dataset):
        fig =plt.figure(figsize=PAPER_A4_LAND ,facecolor='white')
        fig, axes = plt.subplots(figsize=PAPER_A4_LAND ,facecolor='white' ,nrows=1, ncols=3)

        median = dataset.df['ts_deltas_ms'].median()
        hertz = 1/ (median / 1000)

        title_str = "{:0.1f} ms, {:0.1f} Hertz (median) for dataset {}".format(median, hertz, dataset.data_folder,
                                                                               dataset.num_records,
                                                                               dataset.elapsed_minutes)

        fig.suptitle(title_str, fontsize=20)
        # First plot, get the column as a series
        dataset.df['ts_deltas_ms'].plot.box(ax=axes[0])
        axes[0].yaxis.grid(True)
        axes[0].set_title("Raw time deltas")
        axes[0].set_ylabel("Timestep [ms]")

        # Second, remove outliers
        ts_no_outliers1 = remove_outliers(dataset.df['ts_deltas_ms'])
        # time_df['ts_no_outliers1'] = ts_no_outliers1
        ts_no_outliers1.plot.box(ax=axes[1])
        axes[1].set_title("Outliers (3σ) removed")
        axes[1].yaxis.grid(True)

        ts_no_outliers2 = remove_outliers(ts_no_outliers1)
        axes[2].set_title("Outliers (3σ) removed again")
        # time_df['ts_no_outliers2'] = ts_no_outliers2
        ts_no_outliers2.plot.box(ax=axes[2])
        axes[2].yaxis.grid(True)

        outpath = os.path.join(dataset.path_dataset, 'Timestep analysis.png')
        fig.savefig(outpath)
        logging.debug("Wrote boxplots_time figure to {}".format(outpath))

    def histogram_steering(self, dataset):
        fig = plt.figure(figsize=PAPER_A5_LAND, facecolor='white')
        hist_steering = dataset.df['steering_signal'].hist()

        title_str = "Histogram of steering signals"
        subtitle_str = "Dataset: {}, {} records over {:0.1f} minutes ".format(dataset.data_folder, dataset.num_records,
                                                                              dataset.elapsed_minutes)

        hist_steering.set_title(subtitle_str)

        fig.suptitle(title_str, fontsize=20)
        # fig.title(subtitle_string, fontsize=10)

        outpath = os.path.join(dataset.path_dataset, 'Steering Histogram.png')
        fig.savefig(outpath)
        logging.debug("Wrote histogram_steering figure to {}".format(outpath))

    def histogram_throttle(self, dataset):
        fig = plt.figure(figsize=PAPER_A5_LAND, facecolor='white')
        hist_throttle = dataset.df['throttle_signal'].hist()
        title_str = "Histogram of throttle signals"
        subtitle_str = "Dataset: {}, {} records over {:0.1f} minutes ".format(dataset.data_folder, dataset.num_records,
                                                                              dataset.elapsed_minutes)
        hist_throttle.set_title(subtitle_str)

        fig.suptitle(title_str, fontsize=20)
        # fig.title(subtitle_string, fontsize=10)

        outpath = os.path.join(dataset.path_dataset, 'Throttle Histogram.png')
        fig.savefig(outpath)
        logging.debug("Wrote histogram_throttle figure to {}".format(outpath))

    def plot12(self, dataset, ts_string_indices, source_jpg_folder='jpg_images', extension='jpg', rows=3, cols=4,
               outfname='Sample Frames.png', cmap=None, gui_color='green'):
        """
        Render N records to analysis
        """
        # Settings ############################################################
        font_label_box = {
            'color': 'green',
            'size': 16,
        }
        font_steering = {'family': 'monospace',
                         # 'color':  'darkred',
                         'weight': 'normal',
                         'size': 20,
                         }
        ROWS = rows
        COLS = cols
        NUM_IMAGES = ROWS * COLS

        # Figure ##############################################################
        # figsize = [width, height]
        fig = plt.figure(figsize=PAPER_A3_LAND, facecolor='white')
        fig.suptitle("Sample frames, Dataset: {}".format(dataset.data_folder), fontsize=20)

        for i, ts_string_index in enumerate(ts_string_indices):
            rec = dataset.df.loc[ts_string_index]

            timestamp_string = rec['datetime'].strftime("%D %H:%M:%S.") + "{:.2}".format(
                str(rec['datetime'].microsecond))

            if 'steering_pred_signal' in dataset.df.columns:
                this_label = "{}\n{:0.2f}/{:0.2f} steering \n{:0.2f} throttle".format(timestamp_string,
                                                                                      rec['steering_signal'],
                                                                                      rec['steering_pred_signal'],
                                                                                      rec['throttle_signal'])
            else:
                this_label = "{}\n{:0.2f}/ steering \n{:0.2f} throttle".format(timestamp_string, rec['steering_signal'],
                                                                               rec['throttle_signal'])

            ax = fig.add_subplot(ROWS, COLS, i + 1)

            # Main Image ##########################################################
            jpg_path = os.path.join(dataset.path_dataset, source_jpg_folder, ts_string_index + '.' + extension)
            assert os.path.exists(jpg_path), "{} does not exist".format(jpg_path)
            img = mpl.image.imread(jpg_path)
            ax.imshow(img, cmap=cmap)
            # plt.title(str_label)

            # Data box ########################################################

            # ax.axes.get_xaxis().set_visible(False)
            # ax.axes.get_yaxis().set_visible(False)
            t = ax.text(5, 25, this_label, color=gui_color, alpha=1)
            # t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))

            # Steering widget HUD #################################################
            # Steering HUD: Actual steering signal
            steer_actual = ''.join(['|' if v else '-' for v in dataset.linear_bin(rec['steering_signal'])])
            text_steer = ax.text(80, 105, steer_actual, fontdict=font_steering, horizontalalignment='center',
                                 verticalalignment='center', color=gui_color)
            # Steering HUD: Predicted steering angle
            if 'steering_pred_signal' in dataset.df.columns:
                steer_pred = ''.join(['◈' if v else ' ' for v in dataset.linear_bin(rec['steering_pred_signal'])])
                text_steer_pred = ax.text(80, 95, steer_pred, fontdict=font_steering, horizontalalignment='center',
                                          verticalalignment='center', color='red')

        outpath = os.path.join(dataset.path_dataset, outfname)
        fig.savefig(outpath)
        logging.debug("Wrote Sample Frames figure to {}".format(outpath))

    def plot_sample_frames(self, dataset):

        # Right turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] > 0.9)
        these_indices = dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Left turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] < -0.9)
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Straight
        this_mask = dataset.mask & ((dataset.df['steering_signal'] > -0.1) & (dataset.df['steering_signal'] < 0.1))
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # return these_indices
        self.plot12(dataset, these_indices)

        # This is a pointer to the file
        # npz_file=np.load(dataset.path_frames_npz)

        # frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
        # logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))

        # return frames_array

    def plot_sample_frames_bw(self, dataset):

        # Right turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] > 0.9)
        these_indices = dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Left turn
        this_mask = dataset.mask & (dataset.df['steering_signal'] < -0.9)
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # Straight
        this_mask = dataset.mask & ((dataset.df['steering_signal'] > -0.1) & (dataset.df['steering_signal'] < 0.1))
        these_indices += dataset.df[this_mask].sample(4)['timestamp'].tolist()

        # return these_indices
        self.plot12(dataset=dataset, ts_string_indices=these_indices, source_jpg_folder='jpg_images_Y', extension='jpg',
                    rows=3, cols=4, outfname='Sample Frames Y.png', cmap='bwr', gui_color='black')

        # This is a pointer to the file
        # npz_file=np.load(dataset.path_frames_npz)

        # frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
        # logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))

        # return frames_array

    def __get_records(self, batch_indices):
        """Custom method - get the y labels
        """
        this_batch_df = self.dataset.df.loc[batch_indices]
        steering_values = this_batch_df['steering_signal'].values
        steering_records_array = self.dataset.bin_Y(steering_values)
        logging.debug("Generating {} records {}:".format(steering_records_array.shape[0], steering_records_array.shape))
        return steering_records_array

    def __data_generation(self, batch_indices):
        """Keras generator method - Generates data containing batch_size samples
        """

        X = self.__get_npy_arrays(batch_indices)
        y = self.__get_records(batch_indices)

        return X, y

