# Corral
Offline training suite for autonomous RC vehicles
- `Corral`, offline vehicle training suite

### Features
1. **DataSet** class;
   * Enforces contract for further offline processing pipeline
   * Strictly aggregates the numpy images and the saved records and signals
   on the timestamped data
   * Transforms underlying records
   * Flexible masking (cleaning) of data set by;
      * Start - stop keyframe markers (datetime stamps)
      * State values i.e. throttle < 0.1
1. **Plotter** class
   * Operates on DataSet to plot summary histograms, charts etc.
   test* Operates on DataSet to generate analysis frames with a HUD overlay
1. **DataGenerator** class operates on DataSet to serve batches to Keras
1. **Trainer** class operatates on DataSet to train a Keras model
1. **Saliency** class operates on DataSet to visualize the network activations
1. **VideoWriter** class uses ffmpeg to write a video from a directory of frames
