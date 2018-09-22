import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

#%%
PATH_TEST_DATA = glob.glob(os.path.expanduser('~/MULE_DATA_TEST'))[0]

#test_img_dir = "test_images/"
original_image_names = os.listdir(PATH_TEST_DATA)
paths_original_image_names = list(map(lambda name: os.path.join(PATH_TEST_DATA,name), original_image_names))
print(original_image_names)

FIGSIZE = (12,12)
#%% ===========================================================================
def show_image_list(img_list, cols=2, fig_size=(12, 12), img_labels=original_image_names, show_ticks=True):
    img_count = len(img_list)
    rows = img_count / cols
    cmap = None
    plt.figure(figsize=fig_size)
    for i in range(0, img_count):
        img_name,ext = os.path.splitext(img_labels[i])
        
        plt.subplot(rows, cols, i+1)
        img = img_list[i]
        if len(img.shape) < 3:
            cmap = "gray"
        
        if not show_ticks:
            plt.xticks([])
            plt.yticks([])
            
        #plt.title(img_name[len(test_img_dir):])    
        plt.title(img_name)    
        plt.imshow(img, cmap=cmap)

    plt.tight_layout()
    plt.show()
    
original_images = list(map(lambda img_path: mpimg.imread(img_path), paths_original_image_names))
print("Total image count: ", len(original_images))
show_image_list(original_images)    
r = original_images[0]
#type(r)
#%%
#def to_hsv(img):
#    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#
#def to_hsl(img):
#    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#
#hsv_images = list(map(lambda img: to_hsv(img), original_images))
#hsl_images = list(map(lambda img: to_hsl(img), original_images))
#len(hsv_images)
#r = hsv_images[0]

#%%
#img_count = len(hsv_images)
#interleaved_hsx = list(zip(original_images, hsv_images, hsl_images))
#
#k = 0
#for hsx in interleaved_hsx:
#    img_name = original_image_names[k]
#    show_image_list(hsx, cols=3, fig_size=(12, 12), img_labels=[img_name, img_name, img_name] )
#    k += 1


#%%
# Image should have already been converted to HSL color space
def isolate_white(img,low=50, high=200):
    low_threshold = np.array([low, low, low], dtype=np.uint8)
    high_threshold = np.array([high, high, high], dtype=np.uint8)  
    
    mask = cv2.inRange(img, low_threshold, high_threshold)
    
    return mask

def isolate_white2(img):
    # Caution - OpenCV encodes the data in ****HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([15, 38, 115], dtype=np.uint8)
    # Higher value equivalent pure HSL is (75, 100, 80)
    high_threshold = np.array([35, 204, 255], dtype=np.uint8)  
    
    mask = cv2.inRange(img, low_threshold, high_threshold)
    
    return mask
    
# Image should have already been converted to HSL color space
def isolate_yellow_hsl(img):
    # Caution - OpenCV encodes the data in ****HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([15, 38, 115], dtype=np.uint8)
    # Higher value equivalent pure HSL is (75, 100, 80)
    high_threshold = np.array([35, 204, 255], dtype=np.uint8)  
    
    mask = cv2.inRange(img, low_threshold, high_threshold)
    
    return mask
                            

# Image should have already been converted to HSL color space
def isolate_white_hsl(img):
    # Caution - OpenCV encodes the data in ***HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([0, 200, 0], dtype=np.uint8)
    # Higher value equivalent pure HSL is (360, 100, 100)
    high_threshold = np.array([180, 255, 255], dtype=np.uint8)  
    
    mask = cv2.inRange(img, low_threshold, high_threshold)
    
    return mask

#%% My mask
my_white = list(map(lambda img: isolate_white(img,60,230), original_images))
#my_white2 = list(map(lambda img: isolate_white2(img), original_images))

img_count = len(original_images)
my_interleaved_isolated = list(zip(original_images, my_white))

k = 0
for my_masked in my_interleaved_isolated:
    img_name = original_image_names[k]
    show_image_list(my_masked, cols=2, fig_size=FIGSIZE, img_labels=[img_name, img_name] )
    k += 1

#%%
def combine_mask(img, mask):
    #hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
    return cv2.bitwise_and(img, img, mask=mask)    

def filter_img(img,low,high):
    #hsl_img = to_hsl(img)
    #hsl_yellow = isolate_yellow_hsl(hsl_img)
    img_mask = isolate_white(img,low,high)
    return combine_mask(img, img_mask)

combined_images = list(map(lambda img: filter_img(img,50,190), original_images))

img_count = len(combined_images)
my_masked_imgs_interleave = list(zip(original_images, combined_images))

k = 0
for my_masked in my_masked_imgs_interleave:
    img_name = original_image_names[k]
    show_image_list(my_masked, cols=2, fig_size=FIGSIZE, img_labels=[img_name, img_name] )
    k += 1
    
#%%
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

grayscale_images = list(map(lambda img: grayscale(img), combined_images))
show_image_list(grayscale_images)
#r = my_masked_imgs[0]

#%%
def gaussian_blur(grayscale_img, kernel_size=3):
    return cv2.GaussianBlur(grayscale_img, (kernel_size, kernel_size), 0)
blurred_images1 = list(map(lambda img: gaussian_blur(img, kernel_size=3), grayscale_images))
blurred_images2 = list(map(lambda img: gaussian_blur(img, kernel_size=7), grayscale_images))
blurred_images3 = list(map(lambda img: gaussian_blur(img, kernel_size=11), grayscale_images))

img_count = len(blurred_images1)
interleaved_blur = list(zip(blurred_images1, blurred_images2, blurred_images3))

k = 0
for blurs in interleaved_blur:
    img_name = original_image_names[k]
    show_image_list(blurs, cols=3, fig_size=FIGSIZE, img_labels=[img_name, img_name, img_name] )
    k += 1
#%% Detect edges
def canny_edge_detector(blurred_img, low_threshold, high_threshold):
    return cv2.Canny(blurred_img, low_threshold, high_threshold)


canny_images1 = list(map(lambda img: canny_edge_detector(img, 50, 150), blurred_images2)) 
canny_images2 = list(map(lambda img: canny_edge_detector(img, 0, 10), blurred_images2)) 
canny_images3 = list(map(lambda img: canny_edge_detector(img, 10, 50), blurred_images2))    

img_count = len(canny_images1)
interleaved_canny = list(zip(canny_images1, canny_images2, canny_images3))

k = 0
for cannys in interleaved_canny:
    img_name = original_image_names[k]
    show_image_list(cannys, cols=3, fig_size=FIGSIZE, img_labels=[img_name, img_name, img_name])
    k += 1    


#%% Detect lines
chosen_images = canny_images3
def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)    

rho = 1
# 1 degree
theta = (np.pi/180) * 1
threshold = 15
min_line_length = 30
max_line_gap = 10

##### these settings_work! 
rho = 1.1 # Accuracy
theta = (np.pi/180) * 1 #((np.pi/180) * 1 =  1 degree)
threshold = 30 # Number votes
min_line_length = 20 # Lines shorter than this are rejected
max_line_gap = 10 # maximum gap allowed between line segments to treat them as a single line
###################



##### test
rho = 1.1 # Accuracy
theta = (np.pi/180) * 1 #((np.pi/180) * 1 =  1 degree)
threshold = 30 # Number votes
min_line_length = 10 # Lines shorter than this are rejected
max_line_gap = 20 # maximum gap allowed between line segments to treat them as a single line
###################

hough_lines_per_image = list(map(lambda img: hough_transform(img, rho, theta, threshold, min_line_length, max_line_gap), 
                                 chosen_images))

def draw_lines(img, lines, color=[255, 0, 0], thickness=1, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    return img_copy

img_with_lines = list(map(lambda img, lines: draw_lines(img, lines), original_images, hough_lines_per_image))    
show_image_list(img_with_lines, fig_size=(15, 15))

#%% Detect circles
chosen_images = canny_images3
chosen_images = blurred_images3
circles = cv2.HoughCircles(chosen_images[0], cv2.HOUGH_GRADIENT, 100, 5)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	#cv2.waitKey(0)
    
#%%
#hsl_yellow_images = list(map(lambda img: isolate_yellow_hsl(img), hsl_images))
#hsl_white_images = list(map(lambda img: isolate_white_hsl(img), hsl_images))
#
#img_count = len(hsv_images)
#interleaved_isolated_hsl = list(zip(original_images, hsl_yellow_images, hsl_white_images))
#
#k = 0
#for isolated_hsl in interleaved_isolated_hsl:
#    img_name = original_image_names[k]
#    show_image_list(isolated_hsl, cols=3, fig_size=(12, 12), img_labels=[img_name, img_name, img_name] )
#    k += 1
