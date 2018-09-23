import os
from tensorflow.python import keras as ks
import logging
import numpy as np
# %%

class ModelledData():
    """Augment a dataset with training

    ds - The dataset object, as defined
    ds.df - The dataframe
    ds.mask - The mask
    """

    def __init__(self ,dataset ,model_folder):

        # super().__init__(path_data,data_folder)
        self.ds = dataset

        self.model_folder = model_folder

        # Create the model folder, or reference it
        self.path_model_dir = os.path.join(self.ds.path_data ,self.ds.data_folder ,self.model_folder)
        if not os.path.exists(self.path_model_dir):
            os.makedirs(self.path_model_dir)
            logging.debug("Created a NEW model folder at {}/{}".format(self.ds.data_folder ,self.model_folder))
        else:
            logging.debug("Model folder at {}/{}".format(self.ds.data_folder ,model_folder))

        # assert not os.listdir(self.path_model_dir), "{} not empty".format(self.path_model_dir)
        # logging.debug("This model exists in {}".format(model_dir))

        self.callback_list = None

        if not self.model_folder_empty:
            self.list_models()

    @classmethod
    def from_dataset(self ,dataset ,model_folder):
        self = dataset

        self.model_folder = model_folder


        # Create the model folder, or reference it
        self.path_model_dir = os.path.join(self.path_data ,self.data_folder ,self.model_folder)
        if not os.path.exists(self.path_model_dir):
            os.makedirs(self.path_model_dir)
            logging.debug("Created a NEW model folder at {}/{}".format(self.data_folder ,self.model_folder))
        else:
            logging.debug("Model folder at {}/{}".format(self.data_folder ,model_folder))


        return self

    def list_models(self):
        search_str = os.path.join(self.path_model_dir ,'*.h5')
        # print(search_str)
        paths_weights = glob.glob(search_str)
        logging.debug("{} weights found".format(len(paths_weights)))
        model_files = list()
        for this_wt_path in paths_weights:
            _ ,fname = os.path.split(this_wt_path)
            basename, ext = os.path.splitext(fname)
            # print(basename)
            loss_string = re.search(r"Loss [-+]?[0-9]*\.?[0-9]+" ,basename)[0]
            loss_num = float(re.search("[-+]?[0-9]*\.?[0-9]+" ,loss_string)[0])
            # print(loss_num)
            model_files.append({'path' :this_wt_path, 'loss' :loss_num, 'fname' :basename})

        model_files = sorted(model_files, key=lambda k: k['loss'])
        for mf in model_files:
            # print(mf['fname'],mf['loss'])
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
        logging.debug("Partitioning  train/val to {:0.0f}/{:0.0f}%".format(split *100, ( 1 -split ) *100))

        # The split mask
        msk = np.random.rand(len(self.ds.df)) < split

        # Aggregate the split with the overall mask
        mask_tr = msk & self.ds.mask
        mask_val = ~msk & self.ds.mask

        self.partition = dict()
        self.partition['train'] = self.ds.df.index[mask_tr].values
        self.partition['validation'] = self.ds.df.index[mask_val].values

        tr_pct = len(self.partition['train'] ) /(len(self.partition['train'] ) +len(self.partition['validation']) ) *100
        val_pct = len(self.partition['validation'] ) / \
                    (len(self.partition['train'] ) +len(self.partition['validation']) ) *100
        logging.debug("Actual split: {:0.1f}/{:0.1f}% over {:0.1f}% of the total records".format(tr_pct ,val_pct
                                                                                                 ,self.ds.mask_cover_pct))

    def instantiate_generators(self ,generator_class, generator_params = None):
        if not generator_params:
            generator_params = {'dim': (160 ,120),
                                'batch_size': 64,
                                'n_classes': 15,
                                'n_channels': 3,
                                'shuffle': True,
                                # 'path_frames':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'camera_numpy.zip'),
                                # 'path_records':os.path.join(LOCAL_PROJECT_PATH,THIS_DATASET,'df_record.pck'),
                                }

        self.training_generator = generator_class(self.partition['train'], self.ds, **generator_params)
        self.validation_generator = generator_class(self.partition['validation'], self.ds, **generator_params)
        logging.debug("training_generator and validation_generator instantiated".format())

    def instantiate_model(self ,model_name="baseline_steering_model"):
        def baseline_steering_model():
            model = ks.models.Sequential()
            model.add(ks.layers.Conv2D(24, (5 ,5), strides=(2, 2), activation = "relu", input_shape=(120 ,160 ,3)))
            model.add(ks.layers.Conv2D(32, (5 ,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (5 ,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3 ,3), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3 ,3), strides=(1, 1), activation = "relu"))
            model.add(ks.layers.Flatten()) # This is just a reshape!
            model.add(ks.layers.Dense(100 ,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(50 ,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(15, activation='softmax', name='angle_out'))
            return model

        def blackwhite_steering_model():
            model = ks.models.Sequential()
            model.add(ks.layers.Conv2D(24, (5 ,5), strides=(2, 2), activation = "relu", input_shape=(120 ,160 ,1)))
            model.add(ks.layers.Conv2D(32, (5 ,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (5 ,5), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3 ,3), strides=(2, 2), activation = "relu"))
            model.add(ks.layers.Conv2D(64, (3 ,3), strides=(1, 1), activation = "relu"))
            model.add(ks.layers.Flatten()) # This is just a reshape!
            model.add(ks.layers.Dense(100 ,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(50 ,activation="relu"))
            model.add(ks.layers.Dropout(0.1))
            model.add(ks.layers.Dense(15, activation='softmax', name='angle_out'))
            return model

        # The library of models
        # TODO: Move this outside?
        MODEL_ARCHITECTURE_MAP = {
            "baseline_steering_model" :baseline_steering_model,
            "blackwhite_steering_model" :blackwhite_steering_model}
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
        weight_filename ="weights Loss {val_loss:.2f} Epoch {epoch:02d}.h5"
        weight_path = os.path.join(self.path_model_dir ,weight_filename)
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
        this_log_path = os.path.join(self.path_model_dir ,"history.log".format())
        callback_logger = ks.callbacks.CSVLogger(this_log_path ,separator=',', append=True)


        class MyCallback(ks.callbacks.Callback):
            def __init__(self ,model_folder_path):
                self.model_folder_path = model_folder_path

            def on_train_begin(self, logs={}):
                logging.info("Started training {}".format(self.model))
                self.losses = []
                return

            def on_train_end(self, logs={}):
                logging.info("Finished training {}".format(self.model))
                return


            def on_epoch_begin(self, epoch, logs={}):
                logging.info("Epoch {} {}".format(epoch ,logs))
                return

            def on_batch_end(self, batch, logs={}):
                # self.losses.append(logs.get('loss'))
                logging.debug("\tBatch {} {}".format(batch ,logs))
                pass


            def on_epoch_end(self, epoch, logs={}):
                # print("HISTORY:{}".format(self.model.history.history))
                # print("EPOCH:", epoch)
                # print("LOGS:",logs)


                this_path = os.path.join(self.model_folder_path ,"History epoch {:02d}.json".format(epoch))
                with open(this_path, 'w') as fh:
                    json.dump(self.model.history.history, fh)
                print("Saved history to {}".format(this_path))


        my_history_callback = MyCallback(self.path_model_dir)
        self.callback_list = [callback_wts ,callback_stopping ,callback_logger]

        logging.debug("{} callbacks created".format(len(self.callback_list)))

    def train_model(self, epochs=10):
        assert self.model_folder_empty, "The model folder {} is not empty, instantiate a new model!".format \
            (self.model_folder)
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

        # this_timestamp = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
        logging.debug("Finished training model {}".format(self.model_folder))

    def load_best_model(self):
        model_def = self.list_models()[0]
        path_model = model_def['path']
        assert os.path.exists(path_model)

        self.model = ks.models.load_model(path_model)

        logging.debug("Loaded weights {} with loss {}".format(model_def['fname'] ,model_def['loss']))

    def load_spec_model(self ,model_id=None):
        # Load a specific weights file
        raise
        logging.debug("Loading model {} from {}".format(model_id, path_model))
        path_model = os.path.join(self.path_model_dir ,model_id)
        pred_model = ks.models.load_model(path_model)

    # def make_predictions(self,model_id=None):
    #    logging.debug("Predicting over self.model {}".format(self.model))
    # self.make_model_predictions(self.model)

    def make_predictions(self):
        """Augment the df_records with the predictions
        """
        # print(self.df.head())
        # this_df_records['steering_pred_cats'] = pd.Series(dtype=object)
        # df_records['steering_pred_argmax'] =

        # get all the X array (all numpy arrays), in *proper* order

        #
        logging.debug("Predicting over self.model {}".format(self.model))

        npz_file = np.load(self.ds.path_frames_npz)
        # frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
        frames_array = np.stack([npz_file[idx] for idx in self.ds.df.index], axis=0)
        # print(arrays)
        logging.debug("All images loaded as 1 numpy array {}".format(frames_array.shape))
        logging.debug("Starting predictions ...".format(frames_array.shape))

        # predictions_cats = self.model.predict(frames_array,verbose=1)
        predictions_cats = self.model.predict(frames_array ,verbose=1)

        logging.debug("Predictions complete, shape: {}".format(predictions_cats.shape))
        # logging.debug("Saved categories to column steering_pred_signal_cats".format())

        predictions = self.ds.unbin_Y(predictions_cats)
        # logging.debug("Predictions unbinned, shape: {}".format(predictions.shape))

        self.ds.df['steering_pred_signal'] = predictions
        logging.debug("Predictions added to df in column {}".format('steering_pred_signal'))

        # Get the category of this steering signal
        self.ds.df['steering_pred_signal_catnum'] = self.ds.signal_to_category_number('steering_pred_signal')

        self.raw_accuracy =  sum(self.ds.df[self.ds.mask]['steering_signal_catnum'] == self.ds.df[self.ds.mask]
            ['steering_pred_signal_catnum'] ) /len(self.ds.df[self.ds.mask])
        logging.debug("Raw accuracy {:0.2f}%".format(self.raw_accuracy *100))

        # return predictions

    def save_predictions(self ,path_out):
        assert 'steering_pred_signal' in self.ds.df.columns
        assert 'steering_pred_signal_catnum' in self.ds.df.columns
        pass

