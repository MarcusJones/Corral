# Corral
Offline training suite for autonomous RC vehicles
- `Corral`, offline vehicle training suite

## **DataSet** class
Enforces contract for further offline processing pipeline

Strictly aggregates the numpy images and the saved records and signals
   on the timestamped data

Transforms underlying records

Flexible masking (cleaning) of data set by;

`DataSet.mask_last_Ns()` 


`DataSet.mask_null_throttle()`

```
Masked 459 timesteps throttle<0.1, current cover: 7.8%
Masked 6181 timesteps from 2018-09-12 16:23:26.186000 to 2018-09-12 16
```
 * Start - stop keyframe markers (datetime stamps)
 * State values i.e. throttle < 0.1

## **Plotter** class
   * Operates on DataSet to plot summary histograms, charts etc.
   * Operates on DataSet to generate analysis frames with a HUD overlay
   
## **DataGenerator** class
Operates on DataSet to serve batches to Keras

## **Trainer** class 
Operatates on DataSet to train a Keras model

## **Saliency** class 
Operates on DataSet to visualize the network activations

## **VideoWriter** class 
Uses ffmpeg to write a video from a directory of frames
