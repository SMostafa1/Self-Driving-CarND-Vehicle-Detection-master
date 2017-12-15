##Vehicle Detection

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Figure_1.png
[image2]: ./output_images/test1.jpg
[image3]: ./output_images/test2.jpg
[image4]: ./output_images/test3.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### Writeup / README
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

* I started by reading in all the `vehicle` and `non-vehicle` images.
* I used `get_hog_features` functions found in `Utilities.py` to extract hog features from the images.
* I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
* Here is an example using the `YCrCb` color space and HOG parameters of `orientations=16`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image1] 



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I choose the following
values:colorspace = 'YCrCb' , orient = 11, pix_per_cell = 16, cell_per_block = 2, hog_channel = 'ALL'

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
* After applying `extract_features` found in `Utilities.py` on both arrays that contain vehicle and no-vehicle images, Feature vector is resulted.
* Then scalar is applied on feature vector.
* Labels are defined.
* data is splitted into two sets : train data and validation data.

I trained a linear SVM using LinearSVC classifier from sklearn.svm library with C parameter = 0.001 on Feature vector length of length 1188, it takes 0.93 Seconds to train SVC and I reached
Test Accuracy of SVC =  0.9834


### Finding car in frame

* 'find_cars' function is defined in Utilites.py, this function can extract features using hog sub-sampling and make predictions.
* 'process_frame' is defined in 'MAIN.py':
   * This function use 'find_cars' with different values of ystart , ystop , scale values to adapt near and far
cars.
   * After that it calls 'add_heat' defined in 'Utilities.py',to apply heat map on the predicted rectangles bounding the detected cars.

Found below examples after applying 'process_frame' on test_images:

![alt text][image2]

![alt text][image3] 

![alt text][image4] 

---

### Video Implementation

* I applied 'Process_frame' function on each frame in video.
* The output video detects the cars throughout most of the frames. The video is found in
the package delivered with name ‘output_video\output_Project_video.mp4’
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* It takes a lot of trails to get the working values of `ystart`, `ystop`, and `scale` to correctly find the car in the frame, I believe that it is still need to be more accurate.
* After some trails I found that a scale of less than 1.0 produce a lot of false positives.
* Using colorspace= 'YCrCb' is working well with me when I used 'LUV' it doesn't work, I didn't investigate too much in that.
* C param value of LinearSVC affect a lot results.