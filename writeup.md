## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Undistorted_chessboard_images/undistort_calibration0.png "Undistorted"
[image2]: ./Undistorted_images/202118222823.png "Undistorted"
[image3]: ./Warped_images/202118222826.png "Road Transformed"
[image4]: ./Thresholded/Thresholded_Wrapped1.png "Binary Example"
[image5]: ./output_images/test_image_wrapped1.png "Fit Visual"
[image6]: ./output_images/test_image_original1.png "Output"

[image7]: ./camera_cal/calibration1.jpg "Original Chessboard"
[image8]: ./Thresholded/Thresholded2.png "Road Transformed"

[video1]: ./project_video.mp4 "Video"
[video1]: ./test_videos_output/project_video.mp4 "Video Result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in lines 16 through 73 of the file called `CamCal.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objectp` is just a replicated array of coordinates, and `object_pts` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `image_pts` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `object_pts` and `image_pts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image7]
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 12 through 232 in `Thresholding.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image8]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 109 through 160 in the file `CamCal.py`.  The `warp()` function takes as inputs an image (`img`), internally calculate the source (`trapezium_pts`) and destination (`rectangle_pts`) points.  I chose the hardcode the source and destination points in the following manner:

```python
xdim = undist.shape[1]
ydim = undist.shape[0]
offset = 200
trapezium_pts = np.float32([[xdim/2 -offset/4, ydim * 0.625], #Top left
                            [xdim/2 +offset/4, ydim * 0.625], #Top right
                            [xdim - offset, ydim], #Bottom right
                            [offset, ydim]]) #Bottom left

    rectangle_pts = np.float32([[xdim/4, 0], #Top left
                                [xdim - (offset*3/2), 0], #Top right
                                [xdim - (offset*3/2), ydim], #Bottom right
                                [xdim/4, ydim]]) #Bottom left
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 320, 0        | 
| 690, 450      | 980, 0        |
| 1080, 720     | 980, 720      |
| 200, 720      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial, The code for the second polynomial fit is in the file `polynomial.py` from line 63 through 245 and the steps are as follows:
-Calculate a histogram of the bottom half of the image
-Partition the image into 9 horizontal slices
-Starting from the bottom slice, enclose a 200 pixel wide window around the left peak and right peak of the histogram (split the histogram   in half vertically)
-Go up the horizontal window slices to find pixels that are likely to be part of the left and right lanes, recentering the sliding windows   opportunistically
-Given 2 groups of pixels (left and right lane line candidate pixels), fit a 2nd order polynomial to each group, which represents the         estimated left and right lane lines

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 342 through 380 in my code in `polynomial.py`, using the polynomial fit for the left and right lane lines, I calculated the radius of curvature for each line according to formulas presented during the lessons, then converted the distance units from pixels to meters, assuming 30 meters per 720 pixels in the vertical direction, and 3.7 meters per 700 pixels in the horizontal direction, after that, I averaged the radius of curvature for the left and right lane lines, the value of each curvature was stored in the objects of the lines and the final value was computed during visualization.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 162 through 204 in my code in `CamCal.py` in the function `unwrap()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's the result video ![alt text][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I've done a sanity check over the resulting fits, using 3 values, bottom, middle and top and measauring the distance between the two lines, which should be 700 pixel with a tolerance of 15%.
I've also calculated the distance between the center of the vehicle and the center of the lane, assuming the camera was mounted at the center of the vehicle, and the data was stored on the objects of the lines (left and right)

The Lane detection for the challenging videos have quite some failures, but most can be fixed by shortenning the src points of the wrapping, but this will also ruin the results of the main images/video.

To fix this we need to apply more color thresholds using the LAB and LUV color spaces to better Identify the yellow/white lane colors, and also we might need to adjust the brightness and contrast of the images to get a better clarity.