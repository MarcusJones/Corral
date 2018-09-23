from tensorflow.python import keras as ks
import math
import numpy as np
import logging
# %% DATAGEN
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
        # logging.debug("Frames NPZ located at: {}".format(self.dataset.path_frames_npz))
        logging.debug("{} samples over batch size {} yields {} batches".format(len(self.indices),
                                                                               self.batch_size,
                                                                               math.ceil(len(
                                                                                   self.indices) / self.batch_size), ))

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
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data by selecting these IDs
        X, y = self.__data_generation(batch_indices)

        return X, y

    def on_epoch_end(self):
        """Keras generator method - Shuffles indices after each epoch
        """
        # self.indexes = np.arange(len(self.indices))
        if self.shuffle == True:
            # Shuffle is in-place!
            np.random.shuffle(self.indices)

    def __get_npy_arrays(self, batch_indices):
        """Custom method - get the X input arrays

        Open the npz file and load n frames into memory
        """
        # This is a pointer to the file
        npz_file = np.load(self.dataset.path_frames_npz)

        frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)
        logging.debug("Generating {} frames: {}".format(frames_array.shape[0], frames_array.shape))

        return frames_array

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
        # logging.debug("Frames NPZ located at: {}".format(self.dataset.path_frames_npz))
        logging.debug("{} samples over batch size {} yields {} batches".format(len(self.indices),
                                                                               self.batch_size,
                                                                               math.ceil(len(
                                                                                   self.indices) / self.batch_size), ))

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
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data by selecting these IDs
        X, y = self.__data_generation(batch_indices)

        return X, y

    def on_epoch_end(self):
        """Keras generator method - Shuffles indices after each epoch
        """
        # self.indexes = np.arange(len(self.indices))
        if self.shuffle == True:
            # Shuffle is in-place!
            np.random.shuffle(self.indices)

    def __get_npy_arrays(self, batch_indices):
        """Custom method - get the X input arrays

        Open the npz file and load n frames into memory
        """
        # This is a pointer to the file
        npz_file = np.load(self.dataset.path_frames_npz)

        frames_array = np.stack([npz_file[idx] for idx in batch_indices], axis=0)

        # BLACK AND WHITE
        # TODO: Structure the datagen someway, modularize

        # print(frames_array.shape)
        frames_array_bw = np.mean(frames_array, axis=3)
        frames_array_bw = np.expand_dims(frames_array_bw, 3)
        # print(frames_array_bw.shape)
        # raise
        logging.debug("Generating {} frames: {}".format(frames_array_bw.shape[0], frames_array_bw.shape))

        return frames_array_bw

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



