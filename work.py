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



# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


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
orient = 8  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 1  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = False  # Spatial features on or off
hist_feat = False  # Histogram features on or off
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

image = mpimg.imread('./test_images/test1.jpg')
draw_image = np.copy(image)

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
    sw_x_limits = [
        [None, None],
        [None, None],
        [None, None]
    ]

    sw_y_limits = [
        [400, 640],
        [400, 600],
        [390, 540]
    ]

    sw_window_size = [
        (128, 128),
        (96, 96),
        (80, 80)
    ]

    sw_overlap = [
        (0.5, 0.5),
        (0.5, 0.5),
        (0.5, 0.5)
    ]

    # create sliding windows
    windows = slide_window(image, x_start_stop=sw_x_limits[0], y_start_stop=sw_y_limits[0],
                           xy_window=sw_window_size[0], xy_overlap=sw_overlap[0])

    windows2 = slide_window(image, x_start_stop=sw_x_limits[1], y_start_stop=sw_y_limits[1],
                            xy_window=sw_window_size[1], xy_overlap=sw_overlap[1])

    windows3 = slide_window(image, x_start_stop=sw_x_limits[2], y_start_stop=sw_y_limits[2],
                            xy_window=sw_window_size[2], xy_overlap=sw_overlap[2])

    # show sliding windows
    sliding_windows = []
    sliding_windows.append(draw_boxes(np.copy(image), windows, color=(0, 0, 0), thick=4))
    sliding_windows.append(draw_boxes(np.copy(image), windows2, color=(0, 0, 0), thick=4))
    sliding_windows.append(draw_boxes(np.copy(image), windows3, color=(0, 0, 0), thick=4))

    # drawing one of sliding windows in blue
    sliding_windows[0] = draw_boxes(sliding_windows[0], [windows[9]], color=(0, 0, 255), thick=8)
    sliding_windows[1] = draw_boxes(sliding_windows[1], [windows2[12]], color=(0, 0, 255), thick=8)
    sliding_windows[2] = draw_boxes(sliding_windows[2], [windows3[5]], color=(0, 0, 255), thick=8)

    windows.extend(windows2)
    windows.extend(windows3)
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    print('hot windows {}'.format(len(hot_windows)))

    plt.subplot(141), plt.imshow(sliding_windows[0])
    plt.title('sliding_windows[0]')
    plt.subplot(142), plt.imshow(sliding_windows[1])
    plt.title('sliding_windows[1]')
    plt.subplot(143), plt.imshow(sliding_windows[2])
    plt.title('sliding_windows[2]')
    plt.subplot(144), plt.imshow(window_img)
    plt.title('result')
    plt.show()
else:

    ystart = 200
    ystop = 700
    scale = 2.0

    out_img = find_cars(image, ystart, ystop, scale, color_space, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins,
                        spatial_feat,
                        hist_feat)

    plt.imshow(out_img)
    plt.show()
