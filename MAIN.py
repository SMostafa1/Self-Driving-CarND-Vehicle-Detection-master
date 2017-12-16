################################################Imports#################################################################
import glob
import Utilities
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import time
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import os
from moviepy.editor import *
########################################################################################################################

# Read in cars and notcars
car_images = glob.glob('training_dataset/vehicles/**/*.png')
noncar_images = glob.glob('training_dataset/non-vehicles/**/*.png')

print(len(car_images), len(noncar_images))

car_img = mpimg.imread(car_images[500])
_, car_dst = Utilities.get_hog_features(car_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

noncar_img = mpimg.imread(noncar_images[500])
_, noncar_dst = Utilities.get_hog_features(noncar_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

# Visualize
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
f.subplots_adjust(hspace = .4, wspace=.2)
ax1.imshow(car_img)
ax1.set_title('Car Image', fontsize=16)
ax2.imshow(car_dst, cmap='gray')
ax2.set_title('Car HOG', fontsize=16)
ax3.imshow(noncar_img)
ax3.set_title('Non-Car Image', fontsize=16)
ax4.imshow(noncar_dst, cmap='gray')
ax4.set_title('Non-Car HOG', fontsize=16)
plt.show()



# ########################################Feature extraction parameters#################################################
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

car_features = Utilities.extract_features(car_images, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
notcar_features = Utilities.extract_features(noncar_images, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
########################################################################################################################

################################################Apply scalar and define labels##########################################
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
########################################################################################################################

#####################################Split data into train and test data################################################
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
########################################################################################################################


####################################################Train Classifier####################################################
# Use a linear SVC
svc = LinearSVC(C = 0.001)
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
############Rework############
svc.decision_function(X_train)
svc.predict(X_train)
#############################
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
########################################################################################################################

###########################################Proscess Image###############################################################
#the size and position of cars in the image will be different depending on their distance from the camera,
# `find_cars` will have to be called a few times with different `ystart`, `ystop`, and `scale` values.
# after tuning these parameters thae values below best fit.
#A scale of less than 1.0 produce a lot of false positives.
def process_frame(img):
    Boxes = []

    colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

    ystart = 380
    ystop = 490
    scale = 1.5
    Boxes.append(Utilities.find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 600
    scale = 1.5
    Boxes.append(Utilities.find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 660
    scale = 2.0
    Boxes.append(Utilities.find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    # ystart = 464
    # ystop = 660
    # scale = 3.5
    # Boxes.append(Utilities.find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
    #                             orient, pix_per_cell, cell_per_block, None, None))

    rectangles = [item for sublist in Boxes for item in sublist]

    #####Rework
    ###################Commented for rework#######################
    # heatmap_img = np.zeros_like(img[:, :, 0])
    # heatmap_img = Utilities.add_heat(heatmap_img, rectangles)
    # heatmap_img = Utilities.apply_threshold(heatmap_img, 0.9)
    ####################New code added as per review comments##############################################
    if len(rectangles) > 0:
        det.add_box(rectangles)
    heatmap_img = np.zeros_like(img[:, :, 0])

    for rect_set in det.prev_box:
        heatmap_img = Utilities.add_heat(heatmap_img, rect_set)

    heatmap_img = Utilities.apply_threshold(heatmap_img, 1 + len(det.prev_box) // 2)
    ##################################################################
    labels = label(heatmap_img)
    draw_img, rect = Utilities.draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

####################################Try vehicle detection on test images################################################
# test_images = glob.glob('./test_images/test*.jpg')
#
# fig, axs = plt.subplots(3, 2, figsize=(16,14))
# fig.subplots_adjust(hspace = .004, wspace=.002)
# axs = axs.ravel()
#
# for i, im in enumerate(test_images):
#     axs[i].imshow(process_frame(mpimg.imread(im)))
#     mpimg.imsave('output_images/test'+ str(i+1) + '.jpg',process_frame(mpimg.imread(im)))
#
#     axs[i].axis('off')
#     plt.show()
########################################################################################################################



det = Utilities.Vehicle_Detect()
Video_Output_path = "output_video"
if not os.path.isdir(Video_Output_path):
    os.mkdir(Video_Output_path)

input_path = "project_video.mp4"
print(input_path)
Video_Output_path = "output_video/" + "output_Project_video.mp4"
clip1 = VideoFileClip(input_path)
white_clip = clip1.fl_image(process_frame)
white_clip.write_videofile(Video_Output_path, audio=False)
