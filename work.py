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
    ystart = 400
    ystop = 700
    images = [
              # './test_images/test1.jpg',
              # './test_images/test2.jpg',
              # './test_images/test3.jpg',
              # './test_images/test4.jpg',
              './test_images/test5.jpg',
              # './test_images/test6.jpg'
    ]

    idx = 0
    for image_fn in images:
        image = mpimg.imread(image_fn)
        hot_windows_list2, heatmap, labels = process(image,
                ystart, ystop, color_space, svc,
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




    # # Check the prediction time for a single sample
    # t = time.time()
    #
    # image = mpimg.imread('./test_images/test1.jpg')
    # draw_image = np.copy(image)
    #

    # scale = 2.0
    # scales = [1.0, 2.0, 2.5]
    # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255)]
    #
    # hot_windows_list = []
    # hot_windows_list2 = []
    # for scale in scales:
    #     tt = time.time()
    #     hot_windows = find_cars(image, ystart, ystop, scale, color_space, svc,
    #                         X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                         hist_bins,
    #                         spatial_feat,
    #                         hist_feat)
    #     tt2 = time.time()
    #     print(round(tt2 - tt, 2), 'Seconds Scale {}'.format(scale))
    #     hot_windows_list.append(hot_windows)
    #     hot_windows_list2.extend(hot_windows)
    #
    # t2 = time.time()
    # print('Total',  round(t2 - t, 2), 'Seconds to Search ...')
    #
    # idx = 0
    # for hot_wins in hot_windows_list:
    #
    #     window_img = draw_boxes(draw_image, hot_wins, color=colors[idx], thick=6)
    #     idx += 1
    #     win_num = 100+len(hot_windows_list)*10+idx
    #     plt.subplot(win_num), plt.imshow(window_img)
    #     plt.title('scale \n{}'.format(scales[idx-1]))
    #
    # plt.show()
    #
    # heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # # Add heat to each box in box list
    # heat = add_heat(heat, hot_windows_list2)
    #
    # # Apply threshold to help remove false positives
    # heat = apply_threshold(heat, 3)
    #
    # # Visualize the heatmap when displaying
    # heatmap = np.clip(heat, 0, 255)
    #
    # # Find final boxes from heatmap using label function
    # labels = label(heatmap)
    # draw_img = draw_labeled_bboxes(np.copy(image), labels)
    #
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # plt.show()


