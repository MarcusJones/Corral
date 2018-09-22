# -*- coding: utf-8 -*-


LOCAL_PROJECT_PATH = glob.glob(os.path.expanduser('~/MULE DATA'))[0]
#DATASET_ID = "20180907 193306"
#DATASET_ID = "20180907 184100 BENCHMARK1 TRG"
#DATASET_ID = "20180907 193757"
DATASET_ID = "20180909 215335"
DATASET_ID = "20180909 215233"
DATASET_ID = "20180907 184100 BENCHMARK1 TRG"

#/home/batman/MULE DATA/
assert os.path.exists(LOCAL_PROJECT_PATH)

DIR_OUT = "/home/batman/MULE_DATA_OUT"

#%% ===========================================================================
# Load dataset
# =============================================================================
ds = AIDataSet(LOCAL_PROJECT_PATH,DATASET_ID)
ds.process_time_steps()

plotter = DataSetPlotter()

#%% Get some images to work with    
# Right turn
this_mask = ds.mask & (ds.df['steering_signal'] > 0.9)
this_mask = ds.mask 
these_indices = ds.df[this_mask].sample(20)['timestamp'].tolist()

#%% Plot only channel 1, 
for idx in these_indices:
    fig=plt.figure(figsize=PAPER_A3_LAND,facecolor='white')
    
    r=ds.df.loc[idx,:]
    #ds.
    img = ds.get_one_frame(idx)
    img_Y = img[:,:,0]
    print("{:3} {:>5.1f} {:>5.1f} {:4}".format(
            img_Y.min(),
          img_Y.mean(),
          np.median(img_Y),
          img_Y.max(),
          ))
    out_path = os.path.join(DIR_OUT,r.timestamp+'.jpg')

    cv2.imwrite(out_path, img_Y)
#%%
jpg_path = "/home/batman/MULE_DATA_OUT/1536605026996.jpg"
img = mpl.image.imread(jpg_path)
fig=plt.figure(figsize=PAPER_A3_LAND,facecolor='white')
ax = fig.add_subplot(1,1,1)
ax.imshow(img,cmap='bwr')

#%% Plot the channels

#fig.suptitle("Sample frames, Dataset: {}".format(dataset.data_folder), fontsize=20)
        
ROWS = 1
COLS = 4
r=ds.df.head()
for idx in [these_indices[0]]:
    fig=plt.figure(figsize=PAPER_A3_LAND,facecolor='white')
    
    r=ds.df.loc[idx,:]
    #ds.
    img = ds.get_one_frame(idx)
    img_grey = np.mean(img, -1)
    
    for i in range(3):
        print(i)
        #img[0]
        
        
        
        ax = fig.add_subplot(ROWS,COLS,1+i)
        #ax.imshow(img[:,:,i],cmap='Greys')
        #ax.imshow(img[:,:,i],cmap='gray')
        #ax.imshow(img[:,:,i],cmap='jet')
        #ax.imshow(img[:,:,i],cmap='hot')
        #ax.imshow(img[:,:,i],cmap='copper')
        #ax.imshow(img[:,:,i],cmap='summer')
        #ax.imshow(img[:,:,i],cmap='bone')
        #ax.imshow(img[:,:,i],cmap='seismic')
        #ax.imshow(img[:,:,i],cmap='coolwarm')
        ax.imshow(img[:,:,i],cmap='bwr')
        #ax.imshow(img[:,:,i],cmap='jet')
    #i+=1
    ax = fig.add_subplot(ROWS,COLS,4)
    ax.imshow(img[:,:,i],cmap='jet')

        #ax.colorbar()
    
    
    
    #plt.imshow(img[:,:,0])
    #plt.imshow(img[:,:,1])
    #plt.imshow(img[:,:,2])
    #lum_img = img
    #plt.imshow(lum_img)
    #plt.imshow()
#plotter.plot12(ds,these_indices)