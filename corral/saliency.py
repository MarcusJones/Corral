class SaliencyGen():
    """Aggregates the ModelledDataSet class, and operates to produce frames
    """

    def __init__(self, modelled_dataset):
        # Get the model and dataset (modelleddataset)
        self.modelled_dataset = modelled_dataset
        assert modelled_dataset.has_predictions
        self.model_folder = modelled_dataset.model_folder
        self.path_model_dir = modelled_dataset.path_model_dir
        logging.debug("Saliency gen for model ID: {}".format(self.model_folder))
        logging.debug("Loaded model accuracy: {:0.1f}%".format(self.modelled_dataset.raw_accuracy * 100))

        # Original raw images
        self.path_jpgs_dir = modelled_dataset.ds.path_jpgs_dir
        logging.debug("Source orginal raw images are in folder: {}".format(self.path_jpgs_dir))
        files = glob.glob(os.path.join(self.path_jpgs_dir, '*.jpg'))
        logging.debug("{} jpgs found".format(len(files)))

        # New saliency mask jpgs
        self.path_saliency_jpgs = os.path.join(self.path_model_dir, 'imgs_saliency_masks')
        if not os.path.exists(self.path_saliency_jpgs):
            os.makedirs(self.path_saliency_jpgs)
        logging.debug("Saliency JPG output folder: {}".format(self.path_saliency_jpgs))
        files = glob.glob(os.path.join(self.path_saliency_jpgs, '*.*'))
        logging.debug("{} files found".format(len(files)))

        # Boosted saliency mask jpgs
        self.path_boosted_saliency_jpgs = os.path.join(self.path_model_dir, 'imgs_saliency_masks_boosted')
        if not os.path.exists(self.path_boosted_saliency_jpgs):
            os.makedirs(self.path_boosted_saliency_jpgs)
        logging.debug("BOOSTED Saliency JPG output folder: {}".format(self.path_boosted_saliency_jpgs))
        files = glob.glob(os.path.join(self.path_boosted_saliency_jpgs, '*.*'))
        logging.debug("{} files found".format(len(files)))

        # New saliency frames
        self.path_frames_jpgs = os.path.join(self.path_model_dir, 'frames_saliency_boosted')
        if not os.path.exists(self.path_frames_jpgs):
            os.makedirs(self.path_frames_jpgs)
        logging.debug("Combined HUD frames output folder: {}".format(self.path_frames_jpgs))
        files = glob.glob(os.path.join(self.path_frames_jpgs, '*.*'))
        logging.debug("{} files found".format(len(files)))

    def gen_pure_CNN(self):
        # Get a pure convolutional model, no dropout or other layers
        img_in = ks.layers.Input(shape=(120, 160, 3), name='img_in')
        x = img_in
        x = ks.layers.Convolution2D(24, (5, 5), strides=(2, 2), activation='relu', name='conv1')(x)
        x = ks.layers.Convolution2D(32, (5, 5), strides=(2, 2), activation='relu', name='conv2')(x)
        x = ks.layers.Convolution2D(64, (5, 5), strides=(2, 2), activation='relu', name='conv3')(x)
        x = ks.layers.Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', name='conv4')(x)
        conv_5 = ks.layers.Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv5')(x)
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
            # print("Copied weights from loaded", this_layer_name, "to the pure CNN")
            these_weights = self.modelled_dataset.model.get_layer(this_layer_name).get_weights()
            this_pureconv_layer.set_weights(these_weights)
        logging.debug("Assigned trained model weights to all convolutional layers".format())

    def saliency_tf_function(self):
        inp = self.convolutional_model.input  # input placeholder
        outputs = [layer.output for layer in self.convolutional_model.layers]  # all layer outputs
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

    def write_saliency_mask_jpgs(self, number=None):
        frames_npz = np.load(self.modelled_dataset.ds.path_frames_npz)
        if not number:
            number = len(self.modelled_dataset.ds.df)

        with LoggerCritical(), NoPlots():
            for idx in tqdm.tqdm(self.modelled_dataset.ds.df.index[0:number]):
                rec = self.modelled_dataset.ds.df.loc[idx]
                path_out = os.path.join(self.path_saliency_jpgs, rec['timestamp'] + '.png')
                if os.path.exists(path_out): continue

                # print(idx,rec)

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
                        np.reshape(averaged_activation,
                                   (1, averaged_activation.shape[0], averaged_activation.shape[1], 1)),
                        tf.float32
                    )
                    conv = tf.nn.conv2d_transpose(
                        x, self.layers_kernels[layer],
                        output_shape=(1, output_shape[0], output_shape[1], 1),
                        strides=self.layers_strides[layer],
                        padding='VALID'
                    )
                    with tf.Session() as session:
                        result = session.run(conv)
                    upscaled_activation = np.reshape(result, output_shape)

                    salient_mask = (upscaled_activation - np.min(upscaled_activation)) / (
                                np.max(upscaled_activation) - np.min(upscaled_activation))

                    # Make an RGB 3-channel image
                    salient_mask_stacked = np.dstack((salient_mask, salient_mask, salient_mask))

                    # Save it to JPG
                    plt.imsave(path_out, salient_mask_stacked)

                    # ks.backend.clear_session()

    def blend_simple(self, blur_rad, strength, num_frames=None):
        #
        logging.debug("blur_rad {}, strength {}".format(blur_rad, strength))

        source_folder = os.path.split(self.path_saliency_jpgs)[1]
        target_folder = os.path.split(self.path_boosted_saliency_jpgs)[1]
        jpg_files = glob.glob(os.path.join(self.path_saliency_jpgs, '*.png'))
        logging.debug("Boosting {} frames at {} to {}".format(len(jpg_files), source_folder, target_folder))

        frames_npz = np.load(self.modelled_dataset.ds.path_frames_npz)

        # For testing, write a sample
        if not num_frames:
            num_frames = len(jpg_files)

        with LoggerCritical(), NoPlots():
            for img_path in tqdm.tqdm(jpg_files[0:num_frames]):

                # print(img_path)
                _, fname = os.path.split(img_path)
                path_out = os.path.join(self.path_boosted_saliency_jpgs, fname)
                if os.path.exists(path_out): continue

                timestamp, _ = os.path.splitext(fname)
                # print(timestamp)
                saliency_frame = plt.imread(img_path)[:, :, :3]
                raw_frame = frames_npz[timestamp]

                if 0:  # Try adjusting brightness and contrast
                    b = 0.  # brightness
                    c = 64.  # contrast

                    # call addWeighted function, which performs:
                    #    dst = src1*alpha + src2*beta + gamma
                    # we use beta = 0 to effectively only operate on src1
                    saliency_frame = cv2.addWeighted(saliency_frame, 1. + c / 127., saliency_frame, 0, b - c)

                saliency_frame.setflags(write=1)
                saliency_frame[:, :, 0] = saliency_frame[:, :, 0] * 1.5
                saliency_frame[:, :, 1] = saliency_frame[:, :, 1] * 0
                saliency_frame[:, :, 2] = saliency_frame[:, :, 2] * 0
                blur_kernel = np.ones((blur_rad, blur_rad), np.float32) * strength
                saliency_frame_blurred = cv2.filter2D(saliency_frame, -1, blur_kernel)

                # saliency_frame_blurred = cv2.GaussianBlur(saliency_frame,(1,1),0)

                alpha = 0.004
                beta = 1.0 - alpha
                gamma = 0.0
                try:
                    blend = cv2.addWeighted(raw_frame.astype(np.float32), alpha, saliency_frame_blurred, beta, gamma)
                except:
                    print("BAD IMAGE?", img_path)
                    raise
                # plt.imshow(blend)

                plt.imsave(path_out, blend)

        # Raw masks
        pass

    def create_HUD_frames(self):
        source_folder_jpg = os.path.split(self.path_saliency_jpgs)[1]
        target_folder = os.path.split(self.path_frames_jpgs)[1]
        jpg_files = glob.glob(os.path.join(self.path_boosted_saliency_jpgs, '*.png'))
        logging.debug("Creating {} HUD frames from {} to {}".format(len(jpg_files), source_folder_jpg, target_folder))

        with LoggerCritical(), NoPlots():
            for img_path in tqdm.tqdm(glob.glob(self.path_boosted_saliency_jpgs + r"/*.png")):
                # print(img_path)
                _, fname = os.path.split(img_path)

                index, _ = os.path.splitext(fname)
                pathpart_source_imgs = self.model_folder + r"/" + 'imgs_saliency_masks_boosted'
                path_jpg = os.path.join(self.path_frames_jpgs, index + ".jpg")
                if os.path.exists(path_jpg): continue

                frame_figure = this_saliency.modelled_dataset.ds.gen_record_frame(index,
                                                                                  source_jpg_folder=pathpart_source_imgs,
                                                                                  source_ext='.png')
                ks.backend.clear_session()

                # Save it to jpg
                # path_jpg = os.path.join(OUT_PATH,idx + '.jpg')
                frame_figure.savefig(path_jpg)
        logging.debug("Wrote frames to {}".format(self.path_frames_jpgs))

    def blend_PIL(self, blur_rad, map_name, strength):
        """A more advanced boosting pipeline
        """
        logging.debug("blur_rad {}, map_name {}, strength {}".format(blur_rad, map_name, strength))

        pass