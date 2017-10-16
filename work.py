import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle
from scipy.ndimage.measurements import label
from random import shuffle
from moviepy.editor import VideoFileClip

cars = []
notcars = []
# Read in cars
car_images = glob.glob('./data/vehicles/**/*.png', recursive=True)
print('car_images {}'.format(len(car_images)))
for image in car_images:
    cars.append(image)

# Read notcars
notcar_images = glob.glob('./data/non-vehicles/**/*.png', recursive=True)
for image in notcar_images:
    notcars.append(image)

shuffle(car_images)
shuffle(notcar_images)
print(len(car_images), '         images of vehicles')
print(len(notcar_images), '        images of non-vehicles')

plot_car = False
if plot_car:
    # read one car
    fcar = car_images[0]
    print('fcar: {}'.format(fcar))
    car_image = mpimg.imread(fcar)
    plt.subplot(121), plt.imshow(car_image)
    plt.title('Car Image \n{}'.format(fcar))
    # rad one not car
    fncar = notcar_images[0]
    print('fncar: {}'.format(fncar))
    ncar_image = mpimg.imread(fncar)
    plt.subplot(122), plt.imshow(ncar_image)
    plt.title('Not Car Image \n{}'.format(fncar))

    plt.show()

### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 1  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()





# check if mode exist
filename = 'finalized_model.sav'
svc = None
if os.path.isfile(filename):
    dd = pickle.load(open(filename, 'rb'))
    svc = dd['model']
    X_scaler = dd['X_scaler']
    print('load svm model {}'.format(filename))
else:
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    svc = SVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)

    pickle.dump(svc, open(filename, 'wb'))
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    dd = {'model': svc, 'X_scaler': X_scaler}
    pickle.dump(dd, open(filename, 'wb'))
    print('save svm model {}'.format(filename))
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    print('Feature vector length:', len(X_train[0]))

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')



# Check the prediction time for a single sample
t = time.time()

image = mpimg.imread('./test_images/test6.jpg')
draw_image = np.copy(image)

def process_image(image):
    hot_windows, heatmap, labels = process(image,
                                            color_space, svc,
                                            X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                                            hist_bins,
                                            spatial_feat,
                                            hist_feat)
    result = draw_labeled_bboxes(np.copy(image), labels)
    return result

method = 3
if method == 1:
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32) / 255
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    print('search windows {}'.format(len(windows)))
    draw_image_windows = np.copy(image)
    swindow_img = draw_boxes(draw_image_windows, windows, color=(0, 0, 255), thick=6)
    plt.subplot(121), plt.imshow(swindow_img)
    plt.title('Search Windows')

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    print('hot windows {}'.format(len(hot_windows)))
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    plt.subplot(122), plt.imshow(window_img)
    plt.title('BBox')

    plt.show()
elif method == 2:
    images = [
        './test_images/test1.jpg',
        './test_images/test2.jpg',
        './test_images/test3.jpg',
        './test_images/test4.jpg',
        './test_images/test5.jpg',
        './test_images/test6.jpg'
    ]

    idx = 0
    for image_fn in images:
        image = mpimg.imread(image_fn)
        hot_windows, heatmap, labels = process2(image,
                                                    color_space, svc,
                                                     X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                                                     hog_channel,
                                                     hist_bins,
                                                     spatial_feat,
                                                     hist_feat)

        # win_num = 100+len(hot_windows_list2)*10+idx  #131
        draw_image = np.copy(image)
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        plt.subplot(len(images), 3, idx * 3 + 1), plt.imshow(window_img)

        plt.title('Image {} Hot Windows'.format(idx))
        # plot heatmap
        plt.subplot(len(images), 3, idx * 3 + 2), plt.imshow(heatmap, cmap='hot')
        plt.title('heatmap')
        # draw cars bbox
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        plt.subplot(len(images), 3, idx * 3 + 3), plt.imshow(draw_img)
        plt.title('bbox')

        idx += 1

    plt.show()

elif method == 3:
    ystart = 400
    ystop = 700
    images = [
              './test_images/test1.jpg',
              './test_images/test2.jpg',
              './test_images/test3.jpg',
              './test_images/test4.jpg',
              './test_images/test5.jpg',
              './test_images/test6.jpg'
    ]

    idx = 0
    for image_fn in images:
        image = mpimg.imread(image_fn)
        hot_windows_list2, heatmap, labels = process(image,
                color_space, svc,
                X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                hist_bins,
                spatial_feat,
                hist_feat)

        # win_num = 100+len(hot_windows_list2)*10+idx  #131
        draw_image = np.copy(image)
        window_img = draw_boxes(draw_image, hot_windows_list2, color=(0, 0, 255), thick=6)
        plt.subplot(len(images), 3, idx*3+1), plt.imshow(window_img)

        plt.title('Image {} Hot Windows'.format(idx))
        # plot heatmap
        plt.subplot(len(images), 3, idx * 3 + 2), plt.imshow(heatmap, cmap='hot')
        plt.title('heatmap')
        # draw cars bbox
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        plt.subplot(len(images), 3, idx * 3 + 3), plt.imshow(draw_img)
        plt.title('bbox')

        idx += 1

    plt.show()
else:
    white_output = 'output_videos/project_video.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
