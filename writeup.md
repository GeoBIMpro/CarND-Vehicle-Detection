
** Vehicle Detection Project **

[//]: # (Image References)
[image1]: ./output_images/car_noncar.png
[image2]: ./output_images/car_hog.png
[image2-2]: ./output_images/car_hog_11_16_1.png
[image3]: ./output_images/search_win_1.png
[image3-1]: ./output_images/search_win_1.5.png
[image3-2]: ./output_images/search_win_2.png
[image4]: ./output_images/test_results.png
[image5]: ./output_images/heatmap_filter.png

###Histogram of Oriented Gradients (HOG)

####1.   HOG features

I use `skimage's hog()` to generate HOG features.

The code for this step is contained in `get_hog_features()` in line 439 of the file `lesson_functions.py`.  

I started by reading in all the `car` and `non-car` images.  Here is an example of one of each of the `car` and `non-car` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:


![alt text][image2]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=1`, `pixels_per_cell=(16, 16)` and `cells_per_block=(1, 1)`:

![alt text][image2-2]



####  2. HOG parameters.

When pixels_per_cell is 8 , the HOG feature has more details than 16, but the search time also longer. with same search windows , pixels_per_cell 8 use 5 times more times.
I use the first set of the parameters as performance consideration, as the this vehicle detection system is running on realtime.

####  3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I shuffle the car and non-car images, and trained a SVM using `svc.fit` in `work.py line 112`.

I also split the train data and test data. the tarining time is around 20 seconds.

###  4. Sliding Window Search

I use size (64,64) to seach small size cars in far position. the seach windows as below.

![alt text][image3]

I also use size(96,96) window as below.

![alt text][image3-1]

The third is size(128,128) window as below.

![alt text][image3-2]

####  5. Test images

Ultimately I searched on 3 scales using RGB 3-channel HOG features without spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

But you can find in test image5 the right white car is not dectected. I guess it is because not enough train data.
---

###  Video Implementation

####  1. Veido Result
Here's a [project_video](https://youtu.be/TOLAfcZU9BU)

Here's a [test_video](https://youtu.be/roQnOUG84kI)

####  2. Heatmap and filter

 I created a heatmap from positive windows and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()`.

I set the threshold 2 in `lesson_functions.py line 197`

![alt text][image5]


###  Discussion

####  1. Tuneing parameters

Most of my time is used on tuneing the parameters: orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat.

I also tried different number of seach windows. I found the svm test accuracy is always good around 0.99.

But the results of test images always have issue. false positive detection sometime happen. or the car not detected.

The reason cause this maybe due to not enough train data.

####  2. Bounding Box size

I found the bounding box not always cover the car actural size, but sometime the bbox is much large than the actural car size.

This may cuase serial problem, for example If the car in the adjacent lane, but we detect the car's bbox occupy ego car's lane, this may cuase ego car slow down or hard stop.

I am not sure how to deal this problem.

####  3. Dectection performance

To get good result the `pix_per_cell` is better small, but the search time is longer.

Current I use `pix_per_cell = 16`, for each image, I use 0.6 second to get bbox. It is already slow. Althrough I can use multi-thread or GPU to speed up the search for each scales size.

the idea detection time is less than 0.1s. But in my testing the detection result is very bad. :(

my conclution: HOG feature + SVM is not good enough for realtime car detection, shall use deep learning.
